import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoImageProcessor, AutoModel
import torch.nn.functional as F
import sys
import os

from dino_features_etc import (
    load_dino_model,
    extract_dino_features,
    compute_pairwise_similarities,
    create_soft_labels,
    compute_soft_label_loss,
)

sys.path.append(os.path.abspath(".."))

# Now import modules
from open_clip import get_input_dtype, CLIP, CustomTextCLIP

from open_clip_train.train import AverageMeter
from open_clip_train.distributed import is_master
from open_clip_train.train import unwrap_model
from open_clip_train.zero_shot import zero_shot_eval
from open_clip_train.precision import get_autocast


# Set device to "cuda" if GPU is available, otherwise "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import math
import logging

# Ensure logging works
logging.basicConfig(level=logging.INFO)

from transformers import AutoProcessor

dino_processor = AutoProcessor.from_pretrained("facebook/dinov2-base")  # Example model


# Mock Arguments
class Args:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = "fp32"  # Can be "fp16" if using AMP
    distill = False
    accum_freq = 1
    grad_clip_norm = None
    horovod = False
    skip_scheduler = False
    log_every_n_steps = 1
    batch_size = 2
    world_size = 1
    wandb = False
    rank = 0  # ✅ Add this line


args = Args()


# Dummy Model
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Linear(512, 256)
        self.text_encoder = nn.Linear(512, 256)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, images, texts):
        image_embeds = F.normalize(self.image_encoder(images), dim=1)
        text_embeds = F.normalize(self.text_encoder(texts), dim=1)
        return {
            "image_embeds": image_embeds,
            "text_embeds": text_embeds,
            "logit_scale": self.logit_scale,
        }


model = DummyModel().to(args.device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lambda step: None  # Dummy scheduler


# Dummy DINO Model (Simulated)
class DummyDINO(nn.Module):
    def forward(self, x):
        return torch.randn(x.shape[0], 256).to(args.device)  # Random embeddings


dino_model = DummyDINO().to(args.device)


# Dummy Loss Function
def dummy_loss(**kwargs):
    return {"loss": kwargs["image_embeds"].sum()}  # Example loss


# Dummy Dataset
class DummyDataset:
    def __init__(self, num_samples=10):
        self.num_samples = num_samples
        self.num_batches = num_samples // args.batch_size
        self.dataloader = self

    def __iter__(self):
        for _ in range(self.num_batches):
            yield (
                torch.randn(args.batch_size, 512).to(args.device),  # Images
                torch.randn(args.batch_size, 512).to(args.device),  # Texts
            )

    def set_epoch(self, epoch):
        pass  # No-op for testing


data = {"train": DummyDataset()}


# Dummy helper functions
def extract_dino_features(images, dino_model, processor, device):
    return dino_model(images)


def compute_pairwise_similarities(features):
    return torch.mm(features, features.T)  # Example similarity matrix


def create_soft_labels(similarities):
    return similarities / similarities.max()  # Normalize similarities


def compute_soft_label_loss(pred_sim, soft_labels):
    return ((pred_sim - soft_labels) ** 2).mean()  # Example loss function


def backward(loss, scaler=None):
    loss.backward()


# Import the function
# from train import train_one_epoch  # Replace with actual import


def train_one_epoch(
    model,
    data,
    loss,
    epoch,
    optimizer,
    scaler,
    scheduler,
    dist_model,
    dino_model,  # Add Dino model and processor as arguments
    args,
    tb_writer=None,
):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    # ---------------------------------------

    # Initialize Dino model if not already loaded
    if not hasattr(args, "dino_initialized"):
        dino_model.eval()  # Ensure Dino is in eval mode
        args.dino_initialized = True

    # ---------------------------------------

    model.train()
    if args.distill:
        dist_model.eval()

    data["train"].set_epoch(
        epoch
    )  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data["train"].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, texts = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        # --------------------------------------------------

        with torch.no_grad():
            dino_features = extract_dino_features(
                images, dino_model, dino_processor, device
            )
            dino_similarities = compute_pairwise_similarities(dino_features)
            soft_labels = create_soft_labels(dino_similarities)

        # --------------------------------------------------

        if args.accum_freq == 1:

            # --------------------------------------------------
            # Soft labels with DinoV2
            with autocast():
                model_out = model(images, texts)

                # Compute model's image similarities
                model_image_embeds = model_out["image_embeds"]
                normalized_model_image = F.normalize(model_image_embeds, dim=1)
                model_image_similarities = torch.mm(
                    normalized_model_image, normalized_model_image.T
                )

                # Calculate soft label loss
                soft_label_loss = compute_soft_label_loss(
                    model_image_similarities, soft_labels
                )

                # Original loss calculation
                logit_scale = model_out["logit_scale"]
                if args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(images, texts)
                    model_out.update(
                        {f"dist_{k}": v for k, v in dist_model_out.items()}
                    )

                losses = loss(**model_out, output_dict=True)

                # Combine losses
                # total_loss = sum(losses.values()) + soft_label_loss
                total_loss = soft_label_loss
                losses["soft_label_loss"] = soft_label_loss
                losses["loss"] = total_loss

            # --------------------------------------------------

            # --------------------------------------------------
            # No soft labels
            # with autocast():
            #     model_out = model(images, texts)
            #     logit_scale = model_out["logit_scale"]
            #     if args.distill:
            #         with torch.no_grad():
            #             dist_model_out = dist_model(images, texts)
            #         model_out.update(
            #             {f"dist_{k}": v for k, v in dist_model_out.items()}
            #         )
            #     losses = loss(**model_out, output_dict=True)

            #     total_loss = sum(losses.values())
            #     losses["loss"] = total_loss

            # --------------------------------------------------

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                    # Cache soft labels for the current batch
                    if "soft_labels" in accum_features:
                        accum_features["soft_labels"].append(soft_labels)
                    else:
                        accum_features["soft_labels"] = [soft_labels]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop(
                        "logit_scale"
                    )
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(
                            accumulated[:j] + [model_out[key]] + accumulated[j + 1 :]
                        )

                    # --------------------------------------------------
                    # For soft labels loss calculation
                    # Compute soft label loss for the current batch
                    model_image_embeds = model_out["image_embeds"]
                    normalized_model_image = F.normalize(model_image_embeds, dim=1)
                    model_image_similarities = torch.mm(
                        normalized_model_image, normalized_model_image.T
                    )
                    soft_label_loss = compute_soft_label_loss(
                        model_image_similarities, accum_features["soft_labels"][j]
                    )

                    # Combine losses
                    # total_loss = sum(losses.values()) + soft_label_loss
                    total_loss = soft_label_loss
                    losses["soft_label_loss"] = soft_label_loss
                    losses["loss"] = total_loss

                # --------------------------------------------------

                # --------------------------------------------------
                # Original loss calculation
                # losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                # del inputs
                # del inputs_no_accum
                # total_loss = sum(losses.values())
                # losses["loss"] = total_loss
                # --------------------------------------------------
                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_norm, norm_type=2.0
                    )
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_norm, norm_type=2.0
                    )
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip_norm, norm_type=2.0
                )
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (
            i_accum % args.log_every_n_steps == 0
            or batch_count == num_batches_per_epoch
        ):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = (
                args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            )
            samples_per_second_per_gpu = (
                args.accum_freq * args.batch_size / batch_time_m.val
            )
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, "Please install wandb."
                log_data["step"] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


# Run the test function
train_one_epoch(
    model=model,
    data=data,
    loss=dummy_loss,
    epoch=1,
    optimizer=optimizer,
    scaler=None,
    scheduler=scheduler,
    dist_model=None,  # No distillation in this test
    dino_model=dino_model,
    args=args,
)

print("Trial run completed successfully!")
