import copy
import glob
import logging
import math
import os
import re
import subprocess
import sys
import random
from datetime import datetime
from functools import partial
import json
from datetime import datetime

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "open_clip"))
)

from my_metrics import extract_and_plot_itm_scores

import numpy as np
import torch
from torch import optim


try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

# add near wandb/tensorboard imports
try:
    import mlflow
except ImportError:
    mlflow = None


sys.path.append(os.path.abspath(".."))
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.append(repo_path)

from open_clip import (
    create_model_and_transforms,
    trace_model,
    get_tokenizer,
    create_loss,
)

from .helpers import (
    LATEST_CHECKPOINT_NAME,
    _mlflow_safe,
    _run_clip_blind_on_split, check_text_gradients, get_input_dtype,
    random_seed, natural_key, _canon_path_local, get_latest_checkpoint,
    _batch_to_device, _safe_to_device, _encode_clip_images_and_paths,
    _build_dino_path_maps, _dedup_by_path, _align_dino_feats_to_paths,
    _pair_stats, _normalize_recall_value, _extract_recalls_0_100,
    _dataset_retrieval_score, _epoch_retrieval_score, _mlflow_sanitize_metric_name,
    get_data, is_master, init_distributed_device, broadcast_object,
    setup_logging, parse_args, cosine_lr, const_lr, const_lr_cooldown,
    train_one_epoch, evaluate, pt_load, check_exists, start_sync_process, remote_sync,
)


def patch_vit_lock(vit):
    import types
    def new_lock(self, unlocked_groups=0, freeze_bn_stats=False):
        print(f"[PATCH] VisionTransformer.lock called with unlocked_groups={unlocked_groups}")
        for p in self.parameters():
            p.requires_grad = False
        if unlocked_groups > 0:
            total = len(self.transformer.resblocks)
            start = total - unlocked_groups
            for i, blk in enumerate(self.transformer.resblocks):
                if i >= start:
                    for p in blk.parameters():
                        p.requires_grad = True
            for p in self.ln_post.parameters():
                p.requires_grad = True
            if isinstance(self.proj, torch.nn.Parameter):
                self.proj.requires_grad = True
            else:
                for p in self.proj.parameters():
                    p.requires_grad = True
            print(f"[PATCH] ✅ Unlocked last {unlocked_groups}/{total} ViT blocks")
    vit.lock = types.MethodType(new_lock, vit)

def main(args):
    args = parse_args(args)


    print(f"use_soft_labels: {args.use_soft_labels}")
    print(f"alpha: {args.alpha}")

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)


    if not hasattr(args, "log_checkpoint"):
        args.log_checkpoint = False
    # keep your existing save_logs logic (below re-computes too, but we need an early value)
    args.save_logs = bool(args.logs) and args.logs.lower() != "none" and is_master(args)
    args.write_checkpoints = bool(args.log_checkpoint) and bool(args.save_logs)

    # --- MLflow toggle & setup (env-driven, zero CLI changes) ---
    args.use_mlflow = bool(int(os.environ.get("MLFLOW_ENABLE", "0"))) and (mlflow is not None)
    if args.use_mlflow and is_master(args):
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", f"file:{os.path.join(args.logs, 'mlruns')}")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "open-clip"))

    # --------------------  Mine
    dino_processor, dino_model = None, None

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For DataLoaders:
    g = torch.Generator()
    g.manual_seed(seed)

    def _wif(worker_id):
        s = seed + worker_id
        random.seed(s); np.random.seed(s); torch.manual_seed(s)


    # -------------------------------------------------------------------------------------


    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace("/", "-")
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        args.name = "-".join(
            [
                date_str,
                f"model_{model_name_safe}",
                f"lr_{args.lr}",
                f"b_{args.batch_size}",
                f"j_{args.workers}",
                f"p_{args.precision}",
                f"ts_{timestamp}"
            ]
        )

    resume_latest = args.resume == "latest"
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f"out-{args.rank}" if args.log_local else "out.log"
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and not resume_latest:
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1
        
    if args.use_mlflow and is_master(args):
        _mlflow_run = mlflow.start_run(run_name=args.name)
        logging.info(f"MLflow run id: {_mlflow_run.info.run_id}")
    else:
        _mlflow_run = None
    try:

        # Setup text logger
        args.log_level = logging.DEBUG if args.debug else logging.INFO
        setup_logging(args.log_path, args.log_level)

        # Setup wandb, tensorboard, checkpoint logging
       
        args.tensorboard = "tensorboard" in args.report_to or "all" in args.report_to
        args.checkpoint_path = os.path.join(log_base_path, "checkpoints")

        if is_master(args):
            args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ""
            # create TB dir if needed
            if args.tensorboard_path:
                os.makedirs(args.tensorboard_path, exist_ok=True)
            # create checkpoints dir ONLY if we will write checkpoints
            if args.write_checkpoints:
                os.makedirs(args.checkpoint_path, exist_ok=True)
        else:
            args.tensorboard_path = ""


        if resume_latest:
            resume_from = None
            checkpoint_path = args.checkpoint_path
            # If using remote_sync, need to check the remote instead of the local checkpoints folder.
            if args.remote_sync is not None:
                checkpoint_path = os.path.join(args.remote_sync, args.name, "checkpoints")
                if args.save_most_recent:
                    print(
                        "Error. Cannot use save-most-recent with remote_sync and resume latest."
                    )
                    return -1
                if args.remote_sync_protocol != "s3":
                    print("Error. Sync protocol not supported when using resume latest.")
                    return -1
            if is_master(args):
                # Checking for existing checkpoint via master rank only. It is possible for
                # different rank processes to see different files if a shared file-system is under
                # stress, however it's very difficult to fully work around such situations.
                if args.save_most_recent:
                    # if --save-most-recent flag is set, look for latest at a fixed filename
                    resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                    if not os.path.exists(resume_from):
                        # If no latest checkpoint has been saved yet, don't try to resume
                        resume_from = None
                else:
                    # otherwise, list checkpoint dir contents and pick the newest checkpoint
                    resume_from = get_latest_checkpoint(
                        checkpoint_path, remote=args.remote_sync is not None
                    )
                if resume_from:
                    logging.info(f"Found latest resume checkpoint at {resume_from}.")
                else:
                    logging.info(f"No latest resume checkpoint found in {checkpoint_path}.")
            if args.distributed:
                # sync found checkpoint path to all ranks
                resume_from = broadcast_object(args, resume_from)
            args.resume = resume_from

        if args.copy_codebase:
            copy_codebase(args)

        # start the sync proces if remote-sync is not None
        remote_sync_process = None
        if is_master(args) and args.remote_sync is not None:
            # first make sure it works
            result = remote_sync(
                os.path.join(args.logs, args.name),
                os.path.join(args.remote_sync, args.name),
                args.remote_sync_protocol,
            )
            if result:
                logging.info("remote sync successful.")
            else:
                logging.info("Error: remote sync failed. Exiting.")
                return -1
            # if all looks good, start a process to do this every args.remote_sync_frequency seconds
            remote_sync_process = start_sync_process(
                args.remote_sync_frequency,
                os.path.join(args.logs, args.name),
                os.path.join(args.remote_sync, args.name),
                args.remote_sync_protocol,
            )
            remote_sync_process.start()

        if args.precision == "fp16":
            logging.warning(
                "It is recommended to use AMP mixed-precision instead of FP16. "
                "FP16 support needs further verification and tuning, especially for train."
            )

        if args.horovod:
            logging.info(
                f"Running in horovod mode with multiple processes / nodes. Device: {args.device}."
                f"Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}."
            )
        elif args.distributed:
            logging.info(
                f"Running in distributed mode with multiple processes. Device: {args.device}."
                f"Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}."
            )
        else:
            logging.info(f"Running with a single process. Device {args.device}.")

        dist_model = None
        args.distill = (
            args.distill_model is not None and args.distill_pretrained is not None
        )
        if args.distill:
            # FIXME: support distillation with grad accum.
            assert args.accum_freq == 1
            # FIXME: support distillation with coca.
            assert "coca" not in args.model.lower()

        if (
            isinstance(args.force_image_size, (tuple, list))
            and len(args.force_image_size) == 1
        ):
            # arg is nargs, single (square) image size list -> int
            args.force_image_size = args.force_image_size[0]
        random_seed(args.seed, 0)
        model_kwargs = {}

        if args.siglip:
            model_kwargs["init_logit_scale"] = np.log(10)  # different from CLIP
            model_kwargs["init_logit_bias"] = -10
        
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            args.model,
            args.pretrained,
            precision=args.precision,
            device=device,
            jit=args.torchscript,
            force_quick_gelu=args.force_quick_gelu,
            force_custom_text=args.force_custom_text,
            force_patch_dropout=args.force_patch_dropout,
            force_image_size=args.force_image_size,
            image_mean=args.image_mean,
            image_std=args.image_std,
            image_interpolation=args.image_interpolation,
            image_resize_mode=args.image_resize_mode,  # only effective for inference
            aug_cfg=args.aug_cfg,
            pretrained_image=args.pretrained_image,
            output_dict=True,
            cache_dir=args.cache_dir,
            **model_kwargs,
        )


        if args.distill:
            # FIXME: currently assumes the model you're distilling from has the same tokenizer & transforms.
            dist_model, _, _ = create_model_and_transforms(
                args.distill_model,
                args.distill_pretrained,
                device=device,
                precision=args.precision,
                output_dict=True,
                cache_dir=args.cache_dir,
            )
        
        # After model creation
        if is_master(args):
            print("\n=== MODEL TEXT COMPONENTS ===")
            for name, param in model.named_parameters():
                if "text" in name or "transformer" in name or "token" in name:
                    print(f"{name}: requires_grad={param.requires_grad}, shape={param.shape}")
            print("============================\n")

        if args.use_bnb_linear is not None:
            print(
                "=> using a layer from bitsandbytes.\n"
                "   this is an experimental feature which requires two extra pip installs\n"
                "   pip install bitsandbytes triton"
                "   please make sure to use triton 2.0.0"
            )
            import bitsandbytes as bnb
            from open_clip.utils import replace_linear

            print(f"=> replacing linear layers with {args.use_bnb_linear}")
            linear_replacement_cls = getattr(
                bnb.nn.triton_based_modules, args.use_bnb_linear
            )
            replace_linear(model, linear_replacement_cls)
            model = model.to(device)

        random_seed(args.seed, args.rank)

        if args.trace:
            model = trace_model(model, batch_size=args.batch_size, device=device)

        if hasattr(model, "visual") and hasattr(model.visual, "transformer"):
            patch_vit_lock(model.visual)
            
        if args.lock_image:
            # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
            model.lock_image_tower(
                unlocked_groups=args.lock_image_unlocked_groups,
                freeze_bn_stats=args.lock_image_freeze_bn_stats,
            )

            trainable = [n for n,p in model.visual.named_parameters() if p.requires_grad]
            print(f"[DEBUG] Unlocked {len(trainable)} visual params")
            for n in trainable[:20]:
                print("  ", n)

        if args.lock_text:
            model.lock_text_tower(
                unlocked_layers=args.lock_text_unlocked_layers,
                freeze_layer_norm=args.lock_text_freeze_layer_norm,
            )

        # --- sanity breakdown after locking ---
        if is_master(args):
            _m = model.module if hasattr(model, "module") else model

            def _cnt(mod):
                total = sum(p.numel() for p in mod.parameters())
                train = sum(p.numel() for p in mod.parameters() if p.requires_grad)
                return total, train

            def _fmt(name, mod):
                if mod is None:
                    return
                tot, trn = _cnt(mod)
                print(f"[params] {name}: total={tot:,} | trainable={trn:,}")

            # whole model
            tot, trn = _cnt(_m)
            print(f"[params] model (all): total={tot:,} | trainable={trn:,}")

            # vision tower
            if hasattr(_m, "visual"):
                v = _m.visual
                _fmt("visual (all)", v)
                for name in ["conv1","layer1","layer2","layer3","layer4","attnpool"]:
                    if hasattr(v, name):
                        _fmt(f"visual.{name}", getattr(v, name))

            # text tower
            if hasattr(_m, "transformer"):
                _fmt("text.transformer", _m.transformer)
            if hasattr(_m, "token_embedding"):
                _fmt("text.token_embedding", _m.token_embedding)
            if hasattr(_m, "positional_embedding"):
                pe = getattr(_m, "positional_embedding")
                if hasattr(pe, "numel"):  # bare Parameter case
                    tot, trn = pe.numel(), (pe.numel() if getattr(pe, "requires_grad", False) else 0)
                    print(f"[params] text.positional_embedding: total={tot:,} | trainable={trn:,}")
            if hasattr(_m, "text_projection") and _m.text_projection is not None:
                tp = _m.text_projection
                if hasattr(tp, "parameters"):
                    _fmt("text.text_projection", tp)
                else:  # bare Parameter
                    tot, trn = tp.numel(), (tp.numel() if getattr(tp, "requires_grad", False) else 0)
                    print(f"[params] text.text_projection: total={tot:,} | trainable={trn:,}")


        if args.grad_checkpointing:
            model.set_grad_checkpointing()

        if is_master(args):
            logging.info("Model:")
            logging.info(f"{str(model)}")
            logging.info("Params:")
            params_file = os.path.join(args.logs, args.name, "params.txt")
            with open(params_file, "w") as f:
                for name in sorted(vars(args)):
                    val = getattr(args, name)
                    logging.info(f"  {name}: {val}")
                    f.write(f"{name}: {val}\n")
                    # Log params + params.txt to MLflow
                if args.use_mlflow:
                    # log_params expects small-ish dict; cast to str to be safe
                    mlflow.log_params({k: str(getattr(args, k)) for k in vars(args)})
                    mlflow.log_artifact(params_file)

            # === Count & log parameters (after freezing/locking/DDP) ===

            _m = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            total_params = sum(p.numel() for p in _m.parameters())
            trainable_params = sum(p.numel() for p in _m.parameters() if p.requires_grad)
            pct = 100.0 * trainable_params / (total_params or 1)
            msg = f"Params — total: {total_params:,} | trainable: {trainable_params:,} ({pct:.2f}%)"
            print(msg)
            logging.info(msg)
            if args.use_mlflow:
                mlflow.log_metric("params_total", int(total_params))
                mlflow.log_metric("params_trainable", int(trainable_params))
                mlflow.log_metric("params_trainable_pct", pct)

        if args.distributed and not args.horovod:
            if args.use_bn_sync:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            ddp_args = {}
            if args.ddp_static_graph:
                # this doesn't exist in older PyTorch, arg only added if enabled
                ddp_args["static_graph"] = True
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[device], **ddp_args
            )

                    
            if args.distill:
                dist_model = torch.nn.parallel.DistributedDataParallel(
                    dist_model, device_ids=[device], **ddp_args
                )

        # create optimizer and scaler
        optimizer = None
        scaler = None

        if args.train_data or args.dataset_type == "synthetic":
            assert not args.trace, "Cannot train with traced model"

            opt = getattr(args, "opt", "adamw").lower()
            if opt.startswith("timm/"):
                from timm.optim import create_optimizer_v2

                timm_opt = opt.split("timm/")[-1]
                opt_kwargs = {}
                assert (args.beta1 is None) == (
                    args.beta2 is None
                ), "When using timm optimizer, BOTH beta1 and beta2 must be specified (or not specified)."
                if args.beta1 is not None:
                    opt_kwargs["betas"] = (args.beta1, args.beta2)
                if args.momentum is not None:
                    opt_kwargs["momentum"] = args.momentum
                optimizer = create_optimizer_v2(
                    model,
                    timm_opt,
                    lr=args.lr,
                    weight_decay=args.wd,
                    eps=args.eps,
                    **opt_kwargs,
                )
            else:
                # If some params are not passed, we use the default values based on model name.
                exclude = (
                    lambda n, p: p.ndim < 2
                    or "bn" in n
                    or "ln" in n
                    or "bias" in n
                    or "logit_scale" in n
                )
                include = lambda n, p: not exclude(n, p)

                # Parameter grouping for optimization
                #------------------------------------------
                named_parameters = list(model.named_parameters())

                def select(params, pred):
                    return [(n, p) for n, p in params if pred(n, p) and p.requires_grad]

                # Track which parameters we've already assigned
                assigned_param_ids = set()
                param_groups = []

                # === 1) Heads: ALL projection layers + attnpool ===
                head_params = select(named_parameters,
                    lambda n, p: ("text_projection" in n) or 
                                ("visual.attnpool" in n) or
                                ("proj" in n and "visual" not in n))
                if head_params:
                    param_ids = [id(p) for n, p in head_params]
                    assigned_param_ids.update(param_ids)
                    param_groups.append({
                        "params": [p for n, p in head_params],
                        "lr": getattr(args, "head_lr", 1e-4),
                        "weight_decay": 0.0,
                    })

                # === 2) logit_scale with its own tiny LR ===
                logit_scale_params = select(named_parameters, lambda n, p: "logit_scale" in n)
                logit_scale_params = [(n, p) for n, p in logit_scale_params if id(p) not in assigned_param_ids]
                if logit_scale_params:
                    param_ids = [id(p) for n, p in logit_scale_params]
                    assigned_param_ids.update(param_ids)
                    logit_scale_lr = getattr(args, "logit_scale_lr", 1e-6)
                    param_groups.append({
                        "params": [p for n, p in logit_scale_params],
                        "lr": logit_scale_lr,
                        "weight_decay": 0.0
                    })

                # === 3) Visual tower ===
                if hasattr(model, "visual"):
                    v = model.visual
                    # ResNet backbone - layer4
                    l4_params = select(named_parameters, lambda n, p: n.startswith("visual.layer4"))
                    l4_params = [(n, p) for n, p in l4_params if id(p) not in assigned_param_ids]
                    if l4_params:
                        param_ids = [id(p) for n, p in l4_params]
                        assigned_param_ids.update(param_ids)
                        param_groups.append({
                            "params": [p for n, p in l4_params],
                            "lr": getattr(args, "resnet_lr4", 2e-5),
                            "weight_decay": args.wd
                        })
                    
                    # ResNet backbone - layer3  
                    l3_params = select(named_parameters, lambda n, p: n.startswith("visual.layer3"))
                    l3_params = [(n, p) for n, p in l3_params if id(p) not in assigned_param_ids]
                    if l3_params:
                        param_ids = [id(p) for n, p in l3_params]
                        assigned_param_ids.update(param_ids)
                        param_groups.append({
                            "params": [p for n, p in l3_params],
                            "lr": getattr(args, "resnet_lr3", 1e-5),
                            "weight_decay": args.wd
                        })

                # === 4) Text tower: comprehensive coverage ===
                text_lr = getattr(args, "text_lr", 5e-5)
                text_params = []

                # Text transformer blocks (all unlocked layers)
                if hasattr(model, "transformer"):
                    blocks = getattr(model.transformer, "resblocks", None) or getattr(model.transformer, "layers", None)
                    if blocks is not None:
                        n_blocks = len(blocks)
                        n_unlock = getattr(args, "lock_text_unlocked_layers", 0)
                        start = max(0, n_blocks - n_unlock)
                        for i in range(start, n_blocks):
                            blk_params = select(named_parameters,
                                lambda n, p, i=i: (f"transformer.resblocks.{i}." in n) or 
                                                (f"transformer.layers.{i}." in n))
                            # Filter out already assigned params
                            blk_params = [(n, p) for n, p in blk_params if id(p) not in assigned_param_ids]
                            text_params.extend(blk_params)

                # Text embeddings and other text parameters (only unassigned ones)
                text_embed_params = select(named_parameters,
                    lambda n, p: ("token_embedding" in n) or 
                                ("positional_embedding" in n) or
                                ("text_projection" in n) or
                                (n.startswith("transformer.") and "resblocks" not in n and "layers" not in n))
                text_embed_params = [(n, p) for n, p in text_embed_params if id(p) not in assigned_param_ids]
                text_params.extend(text_embed_params)

                if text_params:
                    param_ids = [id(p) for n, p in text_params]
                    assigned_param_ids.update(param_ids)
                    param_groups.append({
                        "params": [p for n, p in text_params],
                        "lr": text_lr,
                        "weight_decay": args.wd
                    })

                # === 5) Fallback for any remaining trainables ===
                other_params = [(n, p) for n, p in named_parameters if p.requires_grad and id(p) not in assigned_param_ids]
                if other_params:
                    base_lr = args.lr if args.lr is not None else 5e-5
                    param_groups.append({
                        "params": [p for n, p in other_params], 
                        "lr": base_lr, 
                        "weight_decay": args.wd
                    })

                # === FINALLY: Create the optimizer ===
                optimizer = torch.optim.AdamW(
                    param_groups,
                    lr=args.lr if args.lr is not None else 5e-5,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps
                )

            scaler = None
            if args.precision == "amp":
                try:
                    scaler = torch.amp.GradScaler(device=device)
                except (AttributeError, TypeError) as e:
                    scaler = torch.cuda.amp.GradScaler()

        # optionally resume from a checkpoint
        start_epoch = 0
        if args.resume is not None:
            checkpoint = pt_load(args.resume, map_location="cpu")
            if "epoch" in checkpoint:
                # resuming a train checkpoint w/ epoch and optimizer state
                start_epoch = checkpoint["epoch"]
                sd = checkpoint["state_dict"]
                if not args.distributed and next(iter(sd.items()))[0].startswith("module"):
                    sd = {k[len("module.") :]: v for k, v in sd.items()}
                model.load_state_dict(sd)
                if optimizer is not None:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                if scaler is not None and "scaler" in checkpoint:
                    scaler.load_state_dict(checkpoint["scaler"])
                logging.info(
                    f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})"
                )
            else:
                # loading a bare (model only) checkpoint for fine-tune or evaluation
                model.load_state_dict(checkpoint)
                logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

        # initialize datasets
        tokenizer = get_tokenizer(args.model, cache_dir=args.cache_dir)


        # --- DINO feats: load as float32 on CPU, keep pinned (don't move whole tensor to GPU) ---
        if args.use_dino_general:
            print(f"[INFO] Loading precomputed DINO features from {args.dino_fts_path}")
            dino_fts = torch.load(args.dino_fts_path, map_location="cpu")
            # Force a stable dtype (AMP returns Ellipsis; float32 keeps numerics stable)
            if dino_fts.dtype != torch.float32:
                dino_fts = dino_fts.to(dtype=torch.float32)
            dino_fts.requires_grad_(False)
            dino_fts = dino_fts.pin_memory()  # fast CPU→GPU per-batch transfers

            args._precomputed_dino = dino_fts
            args._dino_on_device = False
            print(f"[DINO] Keeping feats on CPU pinned: shape={tuple(dino_fts.shape)}, dtype={dino_fts.dtype}")
        else:
            args._precomputed_dino = None
            args._dino_on_device = False



        # Step 1: Load DINO index map if needed
        if args.use_dino_general:
            print("[INFO] Loading precomputed DINO index map...")
            dino_index_map_path = args.dino_index_map_path

            #dino_index_map_path = "/kaggle/input/dino-map/flickr30k_dino_index_map.pt"
            precomputed_dino_index_map = torch.load(
                dino_index_map_path
            )

            args._dino_index_map = precomputed_dino_index_map

            # Step 2: Call get_data with dino_index_map
            data = get_data(
                args,
                (preprocess_train, preprocess_val),
                epoch=start_epoch,
                tokenizer=tokenizer,
                dino_index_map=precomputed_dino_index_map,
            )
    
        else:
            args._dino_index_map = None
            precomputed_dino_index_map = None
            data = get_data(
                args,
                (preprocess_train, preprocess_val),
                epoch=start_epoch,
                tokenizer=tokenizer,
                dino_index_map=precomputed_dino_index_map,
            )

        
        assert len(data), "At least one train or eval dataset must be specified."



        # create scheduler if train
        scheduler = None
        if "train" in data and optimizer is not None:
            total_steps = (
                data["train"].dataloader.num_batches // args.accum_freq
            ) * args.epochs
            if args.lr_scheduler == "cosine":
                scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps,lr_min=0.0)
            elif args.lr_scheduler == "const":
                scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
            elif args.lr_scheduler == "const-cooldown":
                assert (
                    args.epochs_cooldown is not None
                ), "Please specify the number of cooldown epochs for this lr schedule."
                cooldown_steps = (
                    data["train"].dataloader.num_batches // args.accum_freq
                ) * args.epochs_cooldown
                scheduler = const_lr_cooldown(
                    optimizer,
                    args.lr,
                    args.warmup,
                    total_steps,
                    cooldown_steps,
                    args.lr_cooldown_power,
                    args.lr_cooldown_end,
                )
            else:
                logging.error(
                    f"Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown."
                )
                exit(1)

        # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
        # args.save_logs = args.logs and args.logs.lower() != "none" and is_master(args)
        # args.write_checkpoints = bool(args.log_checkpoint) and bool(args.save_logs)
        
        
        writer = None
        if args.save_logs and args.tensorboard:
            assert tensorboard is not None, "Please install tensorboard."
            writer = tensorboard.SummaryWriter(args.tensorboard_path)


        # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
        # For compatibility, we save state_dict() of the original model, which shares the
        # weights without the prefix.
        original_model = model
        if args.torchcompile:
            logging.info("Compiling model...")

            if args.grad_checkpointing and args.distributed:
                logging.info(
                    "Disabling DDP dynamo optimizer when grad checkpointing enabled."
                )
                # As of now (~PyTorch 2.4/2.5), compile + grad checkpointing work, but DDP optimizer must be disabled
                torch._dynamo.config.optimize_ddp = False

            model = torch.compile(original_model)


        if "train" not in data:
            # If using int8, convert to inference mode.
            if args.use_bnb_linear is not None:
                from open_clip.utils import convert_int8_model_to_inference_mode
                convert_int8_model_to_inference_mode(model)
            for val_key in (
                "val",
                "imagenet-val",
                "imagenet-v2",
                "flickr30k-val",
                "mscoco-val",
            ):
                if val_key in data:
                    evaluate(model, {"val": data[val_key]}, start_epoch, args, tb_writer=writer, tokenizer=tokenizer, log_to_mlflow=True)
            return

        loss = create_loss(args)

        all_eval_results = []
        epoch_results = []

        #---------------------------------------------------------------------
        #                       Zero shoot - pre training Eval
        #---------------------------------------------------------------------

        for val_key in (
            "val",
            "imagenet-val",
            "imagenet-v2",
            "flickr30k-val",
            "mscoco-val",
        ):
            epoch = 0
            if val_key in data:
                result = evaluate(
                    model,
                    {"val": data[val_key]},
                    epoch,
                    args,
                tb_writer=writer,
                tokenizer=tokenizer,
                log_to_mlflow=True,
                mlflow_step=0
                )
                epoch_results.append({"val_name": val_key, "metrics": result})

        all_eval_results.append(
            {
                "epoch": epoch,
                "results": epoch_results,
            }
        )

        best_epoch = None
        best_score = float("-inf")
        best_tie = float("-inf")
        best_epoch_snapshot = None

        # start_epoch = start_epoch + 1
        for epoch in range(start_epoch, args.epochs):
            if is_master(args):
                logging.info(f"Start epoch {epoch}")

            step_logs = train_one_epoch(
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
                tb_writer=writer,
            )

            epoch_results = []
            completed_epoch = epoch + 1

            for val_key in (
                "val",
                "imagenet-val",
                "imagenet-v2",
                "flickr30k-val",
                "mscoco-val",
            ):
                if val_key in data:
                    result = evaluate(
                        model,
                        {"val": data[val_key]},
                        epoch,
                        args,
                    tb_writer=writer,
                    tokenizer=tokenizer,
                    log_to_mlflow=True,
                    mlflow_step=completed_epoch
                    )
                    epoch_results.append({"val_name": val_key, "metrics": result})

            # --- compute retrieval score for this epoch (robust, recall-only, unit-normalized) ---
            epoch_score, epoch_tie = _epoch_retrieval_score(epoch_results)

            if is_master(args):
                print(f"[DEBUG] epoch={completed_epoch}, epoch_score={epoch_score}, tie={epoch_tie}")
                if math.isnan(epoch_score):
                    print(f"[DEBUG]    ⚠️ epoch_score is NaN — no valid recalls found.")

            epoch_summary = {
                "epoch": completed_epoch,
                "results": epoch_results,
                "train_logs": step_logs,
                "retrieval_score": epoch_score,
                "retrieval_tie": epoch_tie,
            }

            # --- update best ---
            is_better = (epoch_score > best_score) or (epoch_score == best_score and epoch_tie > best_tie)
            if not math.isnan(epoch_score) and is_better:
                best_epoch = completed_epoch
                best_score = epoch_score
                best_tie = epoch_tie
                best_epoch_snapshot = epoch_summary

            # ✅ Add this line:
            if is_master(args):
                print(f"[DEBUG] after epoch {completed_epoch}: best_epoch={best_epoch}, best_score={best_score}")
                
            all_eval_results.append(epoch_summary)

            # Saving checkpoints.
            if args.write_checkpoints:
                checkpoint_dict = {
                    "epoch": completed_epoch,
                    "name": args.name,
                    "state_dict": original_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                if scaler is not None:
                    checkpoint_dict["scaler"] = scaler.state_dict()

                if completed_epoch == args.epochs or (
                    args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
                ):
                    torch.save(
                        checkpoint_dict,
                        os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                    )

                if args.delete_previous_checkpoint:
                    previous_checkpoint = os.path.join(
                        args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt"
                    )
                    if os.path.exists(previous_checkpoint):
                        os.remove(previous_checkpoint)

                if args.save_most_recent:
                    tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
                    latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                    torch.save(checkpoint_dict, tmp_save_path)
                    os.replace(tmp_save_path, latest_save_path)

                if args.use_mlflow and is_master(args):
                    latest_ckpt = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt")
                    if os.path.exists(latest_ckpt):
                        mlflow.log_artifact(latest_ckpt, artifact_path="checkpoints")


        if is_master(args):
            os.makedirs(args.checkpoint_path, exist_ok=True)

            loss_plot_path = os.path.join(args.checkpoint_path, "loss_curves.png")
            results_path   = os.path.join(args.checkpoint_path, "final_itm_results.json")
            best_dump_path = os.path.join(args.checkpoint_path, "best_metrics.json")

            # Loss curve
            import matplotlib.pyplot as plt
            # Flatten all train_logs across epochs
            all_steps = []
            for e in all_eval_results:
                if "train_logs" in e and e["train_logs"]:
                    all_steps.extend(e["train_logs"])
            if all_steps:
                loss_json_path = os.path.join(log_base_path, "loss_steps.json")  # <--- save in logs/<run_name>
                with open(loss_json_path, "w") as jf:
                    json.dump(all_steps, jf, indent=2)

                # Log to MLflow as an artifact
                if args.use_mlflow:
                    mlflow.log_artifact(loss_json_path, artifact_path="figures")

                # (optional) quick plot from JSON data if you want to regenerate the curve here
                try:
                    import matplotlib.pyplot as plt

                    # collect all distinct loss keys present (prefixed "loss/")
                    loss_keys = set()
                    for x in all_steps:
                        for k in x.keys():
                            if k.startswith("loss/"):
                                loss_keys.add(k)
                    loss_keys = sorted(loss_keys)  # stable legend order

                    if loss_keys:
                        plt.figure(figsize=(9,6))
                        # plot each loss curve independently (only when value exists)
                        for lk in loss_keys:
                            xs, ys = [], []
                            for x in all_steps:
                                if lk in x and x[lk] is not None:
                                    xs.append(int(x["step"]))
                                    ys.append(float(x[lk]))
                            if xs and ys:
                                plt.plot(xs, ys, label=lk.replace("loss/",""))

                        plt.xlabel("Step")
                        plt.ylabel("Loss")
                        plt.title("Training Loss Curves")
                        plt.legend()
                        plt.tight_layout()
                        loss_plot_path = os.path.join(args.checkpoint_path, "loss_curves.png")
                        plt.savefig(loss_plot_path)
                        plt.close()
                        if args.use_mlflow:
                            mlflow.log_artifact(loss_plot_path, artifact_path="figures")
                except Exception as _e:
                    logging.warning(f"Loss plotting failed: {_e}")



            results_path = os.path.join(args.checkpoint_path, "final_itm_results.json")
            with open(results_path, "w") as f:
                json.dump(all_eval_results, f, indent=2)

            print(f"Final ITM results saved to: {results_path}")

            # Make plots
            extract_and_plot_itm_scores(
                results_file_path=results_path,
                output_plot_path=os.path.join(args.checkpoint_path, "itm_scores_plot.png"),
                output_similarity_plot_path=os.path.join(
                    args.checkpoint_path, "itm_scores_similarity_plot.png"
                ),
                save_csv_path=os.path.join(args.checkpoint_path, "itm_scores.csv"),
            )

            # === MLflow artifacts for final outputs ===
            if args.use_mlflow:
                mlflow.log_artifact(results_path)
                plot1 = os.path.join(args.checkpoint_path, "itm_scores_plot.png")
                plot2 = os.path.join(args.checkpoint_path, "itm_scores_similarity_plot.png")
                csvp  = os.path.join(args.checkpoint_path, "itm_scores.csv")
                for p in [plot1, plot2, csvp]:
                    if os.path.exists(p):
                        mlflow.log_artifact(p, artifact_path="figures")

            # ---------- Best-epoch export & MLflow ----------
            best_dump_path = os.path.join(args.checkpoint_path, "best_metrics.json")
            best_payload = {
                "best_epoch": best_epoch,
                "best_retrieval_score": best_score,   # mean of R@1/5/10 both dirs, in %
                "best_tie_r1_mean": best_tie,         # tie-breaker: mean of R@1 (both dirs), in %
                "epoch_snapshot": best_epoch_snapshot,  # contains per-dataset metrics
            }
            with open(best_dump_path, "w") as bf:
                json.dump(best_payload, bf, indent=2)

            print(f"[DEBUG] best_epoch={best_epoch}, best_epoch_snapshot={best_epoch_snapshot is not None}")

            if args.use_mlflow and best_epoch_snapshot is not None:
                flat = {
                    "best/epoch": float(best_epoch),
                    "best/retrieval_score": float(best_score),
                    "best/tie_r1_mean": float(best_tie),
                }
                for entry in best_epoch_snapshot.get("results", []):
                    vname = entry.get("val_name", "val")
                    for k, v in entry.get("metrics", {}).items():
                        if isinstance(v, (int, float)):
                            raw_key = f"best/{vname}/{k}"
                            key = _mlflow_sanitize_metric_name(raw_key)   # ← sanitize here
                            flat[key] = float(v)
            
                print(f"[DEBUG] Logging best metrics to MLflow ({len(flat)} entries):")
                for k, v in flat.items():
                    print(f"  {k} = {v}")
                mlflow.log_metrics(flat, step=int(best_epoch))

                mlflow.log_metrics(flat, step=int(best_epoch))
                mlflow.log_artifact(best_dump_path, artifact_path="figures")

            # === Extra test: CLIP-blind pairs with DINO (VAL split) ===
            if is_master(args) and getattr(args, "run_clip_blind", False):
                # thresholds interpret as: CLIP ≥ cmin AND DINO ≤ dmax
                thresholds = getattr(args, "clip_blind_thresholds",
                                    [(0.90, 0.60), (0.85, 0.65), (0.80, 0.65)])

                # --- read paths from CLI (lowercase argparse attrs) ---
                val_dino_feats = getattr(args, "dino_fts_path_val", None)
                val_dino_map   = getattr(args, "dino_index_map_path_val", None)
                trn_dino_feats = getattr(args, "dino_fts_path", None)
                trn_dino_map   = getattr(args, "dino_index_map_path", None)

                # --- VAL ---
                val_key = getattr(args, "clip_blind_val_key", "flickr30k-val")
                if val_key in data and val_dino_feats and val_dino_map:
                    _run_clip_blind_on_split(
                        split_key=val_key,
                        data=data,
                        model=model,
                        device=device,
                        dino_feats_path=val_dino_feats,
                        dino_index_map_path=val_dino_map,
                        checkpoint_path=args.checkpoint_path,
                        thresholds=thresholds,
                        use_mlflow=getattr(args, "use_mlflow", False),
                        split_alias="val"
                    )
                else:
                    logging.warning("[CLIP-blind/val] Missing split or --dino_fts_path_val / --dino_index_map_path_val; skipping.")

                # --- TRAIN SPLIT ---
                train_key = getattr(args, "clip_blind_train_key", "train")
                try:
                    if train_key in data and trn_dino_feats and trn_dino_map:
                        _run_clip_blind_on_split(
                            split_key=train_key,
                            data=data,
                            model=model,
                            device=device,
                            dino_feats_path=trn_dino_feats,
                            dino_index_map_path=trn_dino_map,
                            checkpoint_path=args.checkpoint_path,
                            thresholds=thresholds,
                            use_mlflow=getattr(args, "use_mlflow", False),
                            split_alias="train"
                        )
                    else:
                        logging.warning("[CLIP-blind/train] Missing split or DINO paths; skipping.")
                except Exception as e:
                    logging.warning(f"[CLIP-blind/train] Failed with error: {e}. Continuing without interruption.")

        # run a final sync.
        if remote_sync_process is not None:
            logging.info("Final remote sync.")
            remote_sync_process.terminate()
            result = remote_sync(
                os.path.join(args.logs, args.name),
                os.path.join(args.remote_sync, args.name),
                args.remote_sync_protocol,
            )
            if result:
                logging.info("Final remote sync successful.")
            else:
                logging.info("Final remote sync failed.")

    finally:
        if _mlflow_run is not None and is_master(args):
            mlflow.end_run()


def copy_codebase(args):
    from shutil import copytree, ignore_patterns

    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(
        current_code_path, new_code_path, ignore=ignore_patterns("log", "logs", "wandb")
    )
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    print(sys.argv)
    main(sys.argv[1:])
