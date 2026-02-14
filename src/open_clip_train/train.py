import json
import logging
import math
import os
import time
import sys
import re
import numbers


import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel

from open_clip.loss import CyCLIPLoss

try:
    import wandb
except ImportError:
    wandb = None

try:
    import mlflow
except ImportError:
    mlflow = None

from collections import defaultdict, Counter

sys.path.append(os.path.abspath(".."))  # Add parent directory

from open_clip import get_input_dtype, CLIP, CustomTextCLIP
from open_clip_train.distributed import is_master
from open_clip_train.zero_shot import zero_shot_eval
from open_clip_train.precision import get_autocast
from open_clip.my_metrics import flickr_retrieval_eval_

# from .dino_features_etc import (
#     load_dino_model,
#     extract_dino_features,
#     compute_pairwise_similarities,
#     create_soft_labels,
#     compute_soft_label_loss,
# )

# import indices

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def _to_float_dict(d):
    out = {}
    for k, v in d.items():
        if hasattr(v, "item"):
            out[k] = float(v.item())
        else:
            out[k] = float(v)
    return out

def _sanitize_for_mlflow(name: str) -> str:
    # 1) make common ML names readable
    name = name.replace("@", "_at_").replace("%", "_pct_")
    # 2) hard sanitize anything else MLflow doesn't allow
    # allowed: alnum, underscore, dash, dot, space, colon, slash
    return re.sub(r"[^0-9A-Za-z_\-\. :/]", "_", name)

def _floatify(v):
    try:
        return float(v.item())  # torch / numpy scalar
    except Exception:
        return float(v)
    



def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def get_warmup_scaled_value(step, warmup_steps, max_val):
    if warmup_steps <= 0:
        return max_val
    return min(max_val, (step / warmup_steps) * max_val)


def get_teacher_tau(args) -> float:
    return float(getattr(args, "tau_teacher", 0.15))

def get_text_student_tau(args) -> float:
    return float(getattr(args, "tau_text_student", 0.05))

from types import SimpleNamespace


def make_effective_args(args, step):
    eff = SimpleNamespace(**vars(args))  # shallow copy

    # --- warm up ONLY the new loss weights you actually use ---
    if getattr(args, "enable_warmup_dino_hyperparams", False):
        eff.lambda_soft        = get_warmup_scaled_value(step, args.warmup, getattr(args, "lambda_soft", 0.0))
        eff.lambda_graph_near  = get_warmup_scaled_value(step, args.warmup, getattr(args, "lambda_graph_near", 0.0))
        eff.lambda_graph_far   = get_warmup_scaled_value(step, args.warmup, getattr(args, "lambda_graph_far", 0.0))
        eff.lambda_hard_neg    = get_warmup_scaled_value(step, args.warmup, getattr(args, "lambda_hard_neg", 0.0))

    # (Optional) make these visible to the loss if you ever want schedulers inside it
    eff.current_step  = step
    eff.total_steps   = getattr(args, "total_steps", None)

    # --- safety: if we don't have DINO feats this step, zero out DINO-based λ’s ---
    if eff is not None and not getattr(eff, "have_dino_for_batch", False):
        eff.lambda_soft       = 0.0
        eff.lambda_graph_near = 0.0
        eff.lambda_graph_far  = 0.0
        eff.lambda_hard_neg   = 0.0

    return eff



def train_one_epoch(
    model,
    data,
    loss,
    epoch,
    optimizer,
    scaler,
    scheduler,
    dist_model,
    dino_model,
    dino_processor,
    args,
    tb_writer=None,
):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)
    step_logs = []

    # ---------------------------------------
    # if args.use_dino_general:
    #     print("Will be using Dino features for training.")
    #     precomputed_dino_cpu = torch.load(args.dino_fts_path, map_location="cpu").to(dtype=input_dtype)
    #     precomputed_dino_cpu.requires_grad_(False)

    #     try:
    #         torch.cuda.reset_peak_memory_stats(device)
    #         # move the WHOLE tensor once to *this rank’s* GPU
    #         with torch.no_grad():
    #             precomputed_dino_tensor = precomputed_dino_cpu.to(device, non_blocking=True)
    #         del precomputed_dino_cpu
    #         torch.cuda.synchronize(device)
    #         print(f"[DINO] Loaded to {device}: {tuple(precomputed_dino_tensor.shape)}, "
    #             f"dtype={precomputed_dino_tensor.dtype}. "
    #             f"GPU used={torch.cuda.memory_allocated(device)/1e6:.1f}MB.")
    #         args._dino_on_device = True
    #     except RuntimeError as e:
    #         # OOM → fallback to CPU-slice-per-batch path (fast if pinned)
    #         print(f"[DINO] OOM moving full tensor to GPU → fallback to CPU slices. ({e})")
    #         precomputed_dino_cpu = precomputed_dino_cpu.pin_memory()
    #         precomputed_dino_tensor = precomputed_dino_cpu
    #         args._dino_on_device = False

    # ---------------------------------------

    model.train()
    if args.distill:
        dist_model.eval()

    data["train"].set_epoch(epoch)
    dataloader = data["train"].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # Keep references for epoch-end logging
    last_logit_scale = None
    last_images_bs = None


    # --- epoch-level timers (never reset mid-epoch) ---
    epoch_wall_start = time.time()
    epoch_batch_time_sum = 0.0
    epoch_batch_count = 0

    # --- hold raw losses for the current step (for JSON/TB) ---
    last_total_loss_value = None
    last_raw_losses = None  # dict: {loss_name: float}

    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler and scheduler is not None:
            if callable(scheduler):
                scheduler(step)  # Function-based scheduler
            elif hasattr(scheduler, 'step'):
                scheduler.step()  # Object-based scheduler
            else:
                logging.warning(f"Unknown scheduler type: {type(scheduler)}")

        # Unpack batch first
        if args.use_dino_general:
            images, texts, indices = batch
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)

            # Keep indices on GPU only if the DINO tensor is on GPU
            if getattr(args, "_dino_on_device", False):
                indices = indices.to(device=device, non_blocking=True, dtype=torch.long)
            else:
                indices = indices.to("cpu", dtype=torch.long)
        else:
            images, texts = batch
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            texts  = texts.to(device=device, non_blocking=True)

        # NOW do the DINO index range check (only when enabled)
        if args.use_dino_general:
            pre = getattr(args, "_precomputed_dino", None)
            assert pre is not None, "DINO tensor missing"

            n_feats = int(pre.shape[0])
            idx_cpu = indices.detach().to("cpu", non_blocking=False).long()
            mi, ma = int(idx_cpu.min().item()), int(idx_cpu.max().item())
            if mi < 0 or ma >= n_feats:
                # show a few bad examples to help debugging
                bad = idx_cpu[(idx_cpu < 0) | (idx_cpu >= n_feats)]
                eg = bad[:10].tolist()
                raise ValueError(
                    f"[DINO] Out-of-range indices: min={mi}, max={ma}, feats_rows={n_feats}. "
                    f"Examples of bad indices: {eg}. "
                    "This usually means your dino_index_map does not align with the training CSV order "
                    "OR contains placeholder -1 entries."
                )

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        # Dino features (regularization path)
        if args.use_dino_general:
            precomputed = getattr(args, "_precomputed_dino", None)
            if precomputed is None:
                raise RuntimeError("DINO tensor not initialized")

            # if getattr(args, "_dino_on_device", False):
            #     dino_features = precomputed.index_select(0, indices)  # indices on same device
            # else:
            dino_features = precomputed[indices].to(device, non_blocking=True)



        if args.accum_freq == 1:
            with autocast():
                model_out = model(images, texts)

                # distillation
                logit_scale = model_out["logit_scale"]
                if args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(images, texts)
                    model_out.update({f"dist_{k}": v for k, v in dist_model_out.items()})

                model_keys = model_out.keys()

                # ---- CoCa LOSS ----
                if args.use_coca:
                    if "logits" in model_out and "labels" in model_out:
                        losses = loss(
                            image_features=model_out["image_features"],
                            text_features=model_out["text_features"],
                            logits=model_out["logits"],
                            labels=model_out["labels"],
                            logit_scale=logit_scale,
                            output_dict=True,
                        )
                    else:
                        # fallback for non-coca models accidentally using CoCaLoss
                        raise ValueError(
                            "[CoCaLoss] Expected model_out to contain 'logits' and 'labels'. "
                            "Make sure you're using a CoCa model (e.g., --model ViT-B-32-coca)."
                        )
            
                # ---- CYCLIP LOSS ----
                elif isinstance(loss, CyCLIPLoss):

                    if not all(k in model_out for k in ["image_features", "text_features", "logit_scale"]):
                        raise ValueError(
                            f"[CyCLIPLoss] Model output missing required keys. "
                            f"Found: {list(model_out.keys())}"
                        )

                    losses = loss(
                        image_features=model_out["image_features"],
                        text_features=model_out["text_features"],
                        logit_scale=model_out["logit_scale"],
                        output_dict=True,
                    )
                    
                    total_loss = losses["total_loss"] if "total_loss" in losses else sum(losses.values())
                    
                elif args.use_dino_general:
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]

                    have_dino = dino_features is not None
                    # compute step above as you already do
                    args.have_dino_for_batch = bool(have_dino)
                    loss_args = make_effective_args(args, step)
                    

                    losses = loss(
                        image_features=image_features,
                        text_features=text_features,
                        logit_scale=logit_scale,
                        dino_features=dino_features,
                        args=loss_args,           # ← pass the whole args (copied & tweaked)
                        output_dict=True,
                    )


                    total_loss = losses["total_loss"]
                    last_total_loss_value = float(total_loss.detach().item())
                    last_raw_losses = {k: float(v.detach().item()) for k, v in losses.items() if torch.is_tensor(v)}

                    # --- print or log debug info (only if it exists) ---
                    dbg = losses.get("dbg", {})
                    if dbg and (step % 300 == 0):   # e.g. print every 300 steps
                        print(f"[Step {step}] "
                            f"Δmax(img/txt)={dbg['delta_img_max']:.2f}/{dbg['delta_txt_max']:.2f}, "
                            f"corr(r̂,Δp)(img/txt)={dbg['corr_rhat_dprob_img']:.3f}/{dbg['corr_rhat_dprob_txt']:.3f}, "
                            f"pc_err(img/txt)={dbg['pc_err_img']:.2e}/{dbg['pc_err_txt']:.2e}")
    
                else:
                    losses = loss(**model_out, output_dict=True)
                    total_loss = sum(losses.values())
                    last_total_loss_value = float(total_loss.detach().item())
                    last_raw_losses = {k: float(v.detach().item()) for k, v in losses.items() if torch.is_tensor(v)}


            backward(total_loss, scaler)

        else:
            # gradient accumulation path
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

                accum_images.append(images)
                accum_texts.append(texts)

            if ((i + 1) % args.accum_freq) > 0:
                end = time.time()
                continue

            optimizer.zero_grad()

        # Optimizer step
        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset accum buffers
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # clamp logit scale
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        # meters
        # meters
        batch_time_m.update(time.time() - end)
        # keep epoch-level timing independent of per-window resets
        epoch_batch_time_sum += batch_time_m.val
        epoch_batch_count += 1
        end = time.time()


        # collect per-batch loss values for epoch averages only
        batch_size = len(images)
        last_images_bs = batch_size
        for key, val in losses.items():
            if losses is None:
                losses = {}
            if key not in losses_m:
                losses_m[key] = AverageMeter()
            if isinstance(val, torch.Tensor):
                losses_m[key].update(val.item(), batch_size)
            # else:
            #     raise ValueError(f"Expected a tensor for key '{key}', but got {type(val)}")

        last_logit_scale = logit_scale.item()

        # NOTE: Removed all per-step logging (stdout/TB/MLflow) and per-step meter resets

        # Per-step logging every n steps
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data_every_n_steps = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }            
            log_data_every_n_steps.update({name: val.val for name, val in losses_m.items()})

            log_data_every_n_steps = {"train/" + name: val for name, val in log_data_every_n_steps.items()}

            # --- ensure we have a consistent scalar total loss ---
            total_loss_to_log = last_total_loss_value
            if total_loss_to_log is None:
                for k in ("train/total_loss", "train/loss", "train/original_clip_loss", "train/clip_loss"):
                    if k in log_data_every_n_steps:
                        try:
                            total_loss_to_log = float(log_data_every_n_steps[k]); break
                        except Exception:
                            pass
            if total_loss_to_log is not None:
                log_data_every_n_steps["train/total_loss"] = total_loss_to_log

            # --- also expose each raw loss as its own scalar (loss/<name>) ---
            raw_loss_scalars = {}
            if last_raw_losses:
                for k, v in last_raw_losses.items():
                    raw_loss_scalars[f"loss/{k}"] = float(v)

            # TB logging (existing train/* plus new loss/*)
            if tb_writer is not None:
                for name, val in log_data_every_n_steps.items():
                    tb_writer.add_scalar(name, val, step)
                for name, val in raw_loss_scalars.items():
                    tb_writer.add_scalar(name, val, step)

            # JSON-friendly step record with all losses
            step_rec = {
                "step": int(step),
                "epoch": int(epoch),
                "num_samples": int(num_samples),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "scale": float(logit_scale_scalar),
            }
            if total_loss_to_log is not None:
                step_rec["loss/total_loss"] = float(total_loss_to_log)
            step_rec.update(raw_loss_scalars)          # e.g., loss/contrastive_loss, loss/soft_label_loss, loss/...
            step_logs.append(step_rec)


            # resetting batch / data time meters per log window
            # don't reset on the final window, only mid-epoch
            if batch_count < num_batches_per_epoch:
                batch_time_m.reset()
                data_time_m.reset()


    # ------------------ END EPOCH: single consolidated logging ------------------
    if is_master(args):
        batch_size = last_images_bs if last_images_bs is not None else args.batch_size
        num_samples = num_batches_per_epoch * batch_size * args.accum_freq * args.world_size
        samples_per_epoch = dataloader.num_samples

        logit_scale_scalar = last_logit_scale if last_logit_scale is not None else unwrap_model(model).logit_scale.item()
        loss_log = " ".join([f"{loss_name.capitalize()}: {loss_m.avg:#.5g}" for loss_name, loss_m in losses_m.items()])

        # robust epoch-level throughput
        epoch_wall = max(time.time() - epoch_wall_start, 1e-8)
        total_samples = num_batches_per_epoch * args.accum_freq * args.batch_size * args.world_size
        samples_per_second = total_samples / epoch_wall
        samples_per_second_per_gpu = samples_per_second / max(args.world_size, 1)


        logging.info(
            f"Train Epoch: {epoch} [{num_samples}/{samples_per_epoch} (100%)] "
            f"Data (t): {data_time_m.avg:.3f} "
            f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
            f"LR: {optimizer.param_groups[0]['lr']:5f} "
            f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
        )

        # Build epoch-averaged log payload
        log_data = {
            "train/data_time": data_time_m.avg,
            "train/batch_time": batch_time_m.avg,
            "train/samples_per_second": samples_per_second,
            "train/samples_per_second_per_gpu": samples_per_second_per_gpu,
            "train/scale": logit_scale_scalar,
            "train/lr": optimizer.param_groups[0]["lr"],
        }
        log_data.update({f"train/{name}": val.avg for name, val in losses_m.items()})


        exclude = {"train/data_time", "train/batch_time",
           "train/samples_per_second", "train/samples_per_second_per_gpu",
           "train/scale", "train/lr"}

        # MLflow: once per epoch
        if getattr(args, "use_mlflow", False) and mlflow is not None:
            logging.info("mlflow is used")
            mlflow.log_metrics(
            {k: float(v) for k, v in log_data.items() if k not in exclude},
            step=int(epoch)
            )

    return step_logs



def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None, log_to_mlflow=True, mlflow_step=None):
    """
    Evaluate CLIP on datasets where each image has 5 captions (e.g., Flickr30k),
    using a provided cap2img mapping from caption index -> image index.
    """
    metrics = {}
    if not is_master(args):
        return metrics

    device = torch.device(args.device)
    model.eval()

    # Zero-shot (kept identical to evaluate)
    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # Accumulators (store features on CPU to save VRAM)
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        all_cap2img = []
        last_logit_scale_cpu = None

        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                # Expect (images, texts, cap2img)
                if len(batch) == 3:
                    images, texts, cap2img = batch
                    if hasattr(cap2img, "tolist"):
                        ids = cap2img.tolist()
                    elif isinstance(cap2img, (list, tuple)):
                        if len(cap2img) > 0 and isinstance(cap2img[0], str):
                            # Map paths to stable integer ids (memoized on args)
                            p2i = getattr(args, "_eval_path2id", None)
                            if p2i is None:
                                p2i = {}
                                setattr(args, "_eval_path2id", p2i)
                            ids = []
                            for p in cap2img:
                                if p not in p2i:
                                    p2i[p] = len(p2i)
                                ids.append(p2i[p])
                        else:
                            # Already ints or int-like
                            ids = list(cap2img)
                    else:
                        # Scalar fallback
                        try:
                            ids = [int(cap2img)]
                        except Exception:
                            raise TypeError(f"Unsupported cap2img type in eval: {type(cap2img)}")
                    all_cap2img.extend(ids)
                else:
                    images, texts = batch

                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    model_out = model(images, texts)
                    image_features = model_out["image_features"]          # [B, D]
                    text_features  = model_out["text_features"]           # [B, D]
                    logit_scale    = model_out["logit_scale"]             # scalar or [B]
                    logit_scale    = logit_scale.mean()                   # match evaluate()

                    # Move features to CPU for accumulation
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    last_logit_scale_cpu = logit_scale.detach().float().cpu()

                    # Standard symmetric CE loss (identical to evaluate)
                    logits_per_image = logit_scale * (image_features @ text_features.t())
                    logits_per_text  = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                if gen_loss is not None:
                    cumulative_gen_loss += gen_loss * batch_size
                num_samples += batch_size

                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t"
                    )
                    if gen_loss is not None:
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t"
                        )

            # ----- Build per-image features & compute retrieval metrics -----
            # Concatenate all caption-level features (CPU tensors)
            if cap2img is not None:
                txt_feats = torch.cat(all_text_features, dim=0)            # [N_caps, D]
                img_feats_per_cap = torch.cat(all_image_features, dim=0)   # [N_caps, D] (images repeated per caption)
                cap2img = torch.tensor(all_cap2img, dtype=torch.long)      # [N_caps]

                if txt_feats.shape[0] != cap2img.shape[0] or img_feats_per_cap.shape[0] != cap2img.shape[0]:
                    raise RuntimeError(
                        f"Mismatch: txt_feats={txt_feats.shape[0]}, img_feats_per_cap={img_feats_per_cap.shape[0]}, cap2img={cap2img.shape[0]}"
                    )

                # Create unique image-level feature matrix by taking the first occurrence for each image index
                N_img = int(cap2img.max().item()) + 1
                D = int(img_feats_per_cap.shape[1])
                img_feats = torch.zeros(N_img, D, dtype=img_feats_per_cap.dtype)  # CPU
                seen = torch.zeros(N_img, dtype=torch.bool)
                for i_row, img_idx in enumerate(cap2img.tolist()):
                    if not seen[img_idx]:
                        img_feats[img_idx] = img_feats_per_cap[i_row]
                        seen[img_idx] = True

                # Compute retrieval metrics for 5-caption-per-image datasets
                # (Assumes your utility: clip_retrieval_metrics(img_feats, txt_feats, cap2img))
                val_metrics = clip_retrieval_metrics(
                    img_feats=img_feats, txt_feats=txt_feats, cap2img=cap2img
                )
            else:
                val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )

            # Match logging fields to evaluate()
            loss = cumulative_loss / max(1, num_samples)
            metrics.update(
                {**val_metrics, "clip_val_loss": float(loss.item()), "epoch": epoch, "num_samples": num_samples}
            )
            if num_samples > 0 and cumulative_gen_loss != 0.0:
                gen_loss_avg = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": float(gen_loss_avg.item())})

    if not metrics:
        return metrics

    # Console log (identical format)
    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    # Prepare prefixed log data like evaluate()
    log_data = {"val/" + name: val for name, val in metrics.items()}

    if log_to_mlflow and getattr(args, "use_mlflow", False) and (mlflow is not None) and is_master(args):
        # Optionally exclude non-metrics fields like epoch/num_samples:
        exclude_raw = {"epoch", "num_samples"}  # exclude without 'val/' prefix
        to_log = {
            f"val/{_sanitize_for_mlflow(k)}": _floatify(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and k not in exclude_raw
        }
        step_to_use = int(mlflow_step if mlflow_step is not None else epoch)
        mlflow.log_metrics(to_log, step=step_to_use)

    return metrics

@torch.no_grad()
def clip_retrieval_metrics(img_feats, txt_feats, cap2img):
    """
    img_feats: [N_img, D]      (will be L2-normalized here)
    txt_feats: [N_caps, D]     (will be L2-normalized here)
    cap2img:   [N_caps] Long   maps each caption idx -> its image idx in [0..N_img-1]
    """
    device = img_feats.device
    N_img = img_feats.shape[0]
    N_caps = txt_feats.shape[0]

    # Ensure normalization (cosine similarity via dot product)
    img_feats = F.normalize(img_feats, dim=1)
    txt_feats = F.normalize(txt_feats, dim=1)

    # Similarity matrices
    s_txt2img = txt_feats @ img_feats.t()   # [N_caps, N_img]
    s_img2txt = img_feats @ txt_feats.t()   # [N_img,  N_caps]

    # ----- Text → Image (each caption has exactly one GT image) -----
    ranks_ti = torch.argsort(s_txt2img, dim=1, descending=True)          # [N_caps, N_img]
    gt_img = torch.as_tensor(cap2img, device=device, dtype=torch.long)   # [N_caps]
    # position of the correct image in each row
    pos_rank_ti = (ranks_ti == gt_img.unsqueeze(1)).nonzero()[:, 1].float()  # [N_caps]

    def recall_at(ranks_float, K):  # ranks are 0-based
        return (ranks_float < K).float().mean().item() * 100.0

    ti_r1  = recall_at(pos_rank_ti, 1)
    ti_r5  = recall_at(pos_rank_ti, 5)
    ti_r10 = recall_at(pos_rank_ti, 10)
    ti_mr  = pos_rank_ti.mean().item() + 1.0         # mean rank (1-based)
    ti_med = torch.median(pos_rank_ti).item() + 1.0  # median rank (1-based)

    # Also compute average positive-pair cosine similarity (caption vs its GT image)
    pos_sims = s_txt2img[torch.arange(N_caps, device=device), gt_img]    # [N_caps]
    avg_sim = pos_sims.mean().item()

    # ----- Image → Text (each image has multiple GT captions; credit if any in top-K) -----
    # Build caption indices per image
    caps_per_img = [[] for _ in range(N_img)]
    # cap2img may be on CPU; iterate python-side is fine
    for c_idx, i_idx in enumerate(cap2img):
        caps_per_img[int(i_idx)].append(c_idx)

    ranks_it = torch.argsort(s_img2txt, dim=1, descending=True)          # [N_img, N_caps]

    best_ranks_it = []
    for i in range(N_img):
        gt_caps = set(caps_per_img[i])
        order = ranks_it[i].tolist()
        # lowest rank position among GT captions
        best = next((r for r, c in enumerate(order) if c in gt_caps), len(order))
        best_ranks_it.append(best)
    best_ranks_it = torch.as_tensor(best_ranks_it, device=device).float()

    it_r1  = recall_at(best_ranks_it, 1)
    it_r5  = recall_at(best_ranks_it, 5)
    it_r10 = recall_at(best_ranks_it, 10)
    it_mr  = best_ranks_it.mean().item() + 1.0
    it_med = torch.median(best_ranks_it).item() + 1.0

    # ----- Modality gap: distance between mean embeddings of the two modalities -----
    mu_img = img_feats.mean(dim=0)
    mu_txt = txt_feats.mean(dim=0)
    modality_gap = torch.norm(mu_img - mu_txt, p=2).item()  # L2 distance

    return {
        # text->image recalls & ranks
        "text_to_image_R@1":  ti_r1,
        "text_to_image_R@5":  ti_r5,
        "text_to_image_R@10": ti_r10,
        "text_to_image_mean_rank":   ti_mr,
        "text_to_image_median_rank": ti_med,

        # image->text recalls & ranks
        "image_to_text_R@1":  it_r1,
        "image_to_text_R@5":  it_r5,
        "image_to_text_R@10": it_r10,
        "image_to_text_mean_rank":   it_mr,
        "image_to_text_median_rank": it_med,

        # new:
        "average_similarity": avg_sim,    # avg positive-pair cosine sim
        "modality_gap":       modality_gap,  # L2 distance between modality means
    }

def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)

#-----------------------------------------------------------------------------
#           CLip BLINDS
#-----------------------------------------------------------------------------
