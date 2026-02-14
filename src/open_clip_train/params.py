import argparse
import ast
from html import parser


def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        for value in values:
            key, value = value.split("=")
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(
                    value
                )  # fallback to string (avoid need to escape on command line)
        setattr(namespace, self.dest, kw)


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--use_soft_labels",
        action="store_true",
        help="Enable soft labels for contrastive loss",
    )
    parser.add_argument(
        "--enforce_to_text",
        action="store_true",
        help="Enable soft labels of Image also to text",
    )

    # parser.add_argument(
    #     "--use-projection",
    #     action="store_true",
    #     default=False,
    #     help="Use image-to-dino projection (MLP or Linear) before regularization. Default is False.",
    # )

    # parser.add_argument(
    #     "--projection-type",
    #     type=str,
    #     default="mlp",
    #     choices=["mlp", "linear"],
    #     help="Type of projection head to use: 'mlp' (default) or 'linear'.",
    # )

    parser.add_argument(
        "--use_projection",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Use projection head to align CLIP image features with DINO.",
    )

    parser.add_argument(
        "--use_mlflow",
        action="store_true",
        default=True,
        help="Use Mlflow for experiment tracking.",
    )

    parser.add_argument(
        "--projection_type",
        type=str,
        choices=["linear", "mlp"],
        default="mlp",
        help="Type of projection from CLIP to DINO feature space.",
    )

    parser.add_argument(
        "--flickr30k_val",
        type=str,
        default=None,
        help="Path to Flickr30k validation CSV",
    )
    parser.add_argument(
        "--mscoco_val", type=str, default=None, help="Path to MSCOCO validation CSV"
    )

    # -----------------------------
    # Core loss mode
    # -----------------------------

    parser.add_argument(
            "--lambda_original",
            type=float,
            default=1.0,
            help="Weight for original loss term (0 disables).",
        )

    parser.add_argument(
        "--loss_mode",
        type=str,
        default="clip",
        choices=["clip", "siglip"],
        help="Base contrastive loss: 'clip' (InfoNCE) or 'siglip' (BCE).",
    )

    # -----------------------------
    # Soft loss
    # -----------------------------
    parser.add_argument(
        "--lambda_soft",
        type=float,
        default=0.0,
        help="Weight for soft loss term (0 disables).",
    )
    parser.add_argument(
        "--soft_mode",
        type=str,
        default="none",
        choices=["none", "siglip_dino", "kl_teacher"],
        help="Soft loss flavor: DINO-soft SigLIP or KL teacher-student.",
    )
    # SigLIP-soft (percentiles on DINO affinities)
    parser.add_argument(
        "--soft_near_pct",
        type=float,
        default=0.80,
        help="Percentile of DINO affinities treated as near (e.g., 0.80 = 80th).",
    )
    parser.add_argument(
        "--soft_far_pct",
        type=float,
        default=0.20,
        help="Percentile of DINO affinities treated as far (e.g., 0.20 = 20th).",
    )
    parser.add_argument(
        "--soft_w_mid",
        type=float,
        default=0.20,
        help="Weight for mid-range pairs in soft SigLIP (between far and near gates).",
    )
    # KL teacher-student
    parser.add_argument(
        "--teacher_temp",
        type=float,
        default=0.15,
        help="Teacher temperature (DINO image-image) for KL mode.",
    )
    parser.add_argument(
        "--student_temp",
        type=float,
        default=None,
        help="Student temperature (image-image) for KL mode. None -> 1/exp(logit_scale).",
    )

    txt_cov_weight = parser.add_argument(
        "--txt_cov_weight",
        type=float,
        default=0.0,
        help="Weight for text covariance loss.",
    )

    txt_var_weight = parser.add_argument(
        "--txt_var_weight",
        type=float,
        default=0.0,
        help="Weight for text variance loss.",
    )

    parser.add_argument(
        "--topk_teacher",
        type=int,
        default=0,
        help="(Optional) Keep only top-K teacher neighbors per row in KL mode. 0 disables.",
    )
    parser.add_argument(
        "--topp_teacher",
        type=float,
        default=0.0,
        help="(Optional) Nucleus cutoff p for teacher distribution in KL mode. 0 disables.",
    )

    parser.add_argument(
        "--soft_dino_to_text",
        action="store_true",
        help="If set, add text–text KL consistency against DINO teacher distribution.",
    )

    parser.add_argument(
        "--text_lambda",
        type=float,
        default=0.0,
        help="Weight of the text–text KL term added to the soft loss.",
    )

    parser.add_argument(
        "--text_student_temp",
        type=float,
        default=0.05,
        help="Student temperature for text–text KL term.",
    )    

    # -----------------------------
    # WEIGHTS
    # -----------------------------
    parser.add_argument(
        "--lambda_weighted",
        type=float,
        default=0.0,
        help="Weight for weighted loss.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Alpha for weighted loss.",
    )   

    parser.add_argument(
        "--weight_text_symmetry",
        action="store_true",
        default=False,
        help="Set in order to use dino weights for text SYMMETRY as well (both branches). Default is False.",
    )


    # -----------------------------
    # Graph structure (percentile-based selection)
    # -----------------------------
    parser.add_argument(
        "--lambda_graph_near",
        type=float,
        default=0.0,
        help="Weight for graph-near loss (preserve DINO-near neighbors).",
    )
    parser.add_argument(
        "--lambda_graph_far",
        type=float,
        default=0.0,
        help="Weight for graph-far loss (separate DINO-far neighbors).",
    )
    parser.add_argument(
        "--graph_near_pct",
        type=float,
        default=0.80,
        help="Percentile of DINO affinities used as near set (e.g., 0.80 = top 20%).",
    )
    parser.add_argument(
        "--graph_far_pct",
        type=float,
        default=0.20,
        help="Percentile of DINO affinities used as far set (e.g., 0.20 = bottom 20%).",
    )
    parser.add_argument(
        "--far_target_percentile",
        type=float,
        default=0.75,
        help="Percentile of current batch image-image distances to use as the far margin.",
    )

    parser.add_argument(
        "--lambda_geom",
        type=float,
        default=0.0,
        help="Weight for geometry alignment loss (DINO & CLIP).",
    )

    # -----------------------------
    # Hard negatives (percentile gates)
    # -----------------------------
    parser.add_argument(
        "--lambda_hard_neg",
        type=float,
        default=0.0,
        help="Weight for hard negative penalty (text-similar & DINO-far).",
    )
    parser.add_argument(
        "--txt_top_pct",
        type=float,
        default=0.80,
        help="Percentile of text-text cosine above which pairs are 'text-similar'.",
    )
    parser.add_argument(
        "--dino_far_pct",
        type=float,
        default=0.20,
        help="Percentile of DINO affinities below which pairs are 'visually far'.",
    )
    parser.add_argument(
        "--hard_cap_gap",
        type=float,
        default=1.0,
        help="(CLIP mode only) Gap between row-positive and hard-negative logits in smooth hinge.",
    )

    # -----------------------------
    #           END
    # -----------------------------


    parser.add_argument(
        "--use-symmetric-dino-weights",
        action="store_true",
        default=False,
        help="Set in order to use dino weights for both contrastive losses.",
    )

    parser.add_argument(
        "--dino_index_map_path",
        type=str,
        default="/kaggle/input/dino-map/flickr30k_dino_index_map.pt",
        help="Path to the DINO index map file.",
    )


    parser.add_argument(
        "--dino_index_map_path_val",
        type=str,
        default="/leonardo_work/EUHPC_A04_051/nxiros/Thesis/Datasets/Flick30k/Dino/flickr30k_val_dinov2_index_map.pt",
        help="Path to the DINO index map file.",
    )


    parser.add_argument(
        "--dino_fts_path",
        type=str,
        default="/kaggle/input/dino-fts/flickr30k_dino_large_train.pt",
        help="Path to the DINO features file.",
    )

    parser.add_argument(
        "--dino_fts_path_val",
        type=str,
        default="/leonardo_work/EUHPC_A04_051/nxiros/Thesis/Datasets/Flick30k/Dino/flickr30k_val_dinov2_feats.pt",
        help="Path to the DINO features file.",
    )

    parser.add_argument(
        "--use_dino_weight",
        action="store_true",
        default=False,
        help="Use Dino weighting for Clip loss scaling. Default is False.",
    )

    parser.add_argument(
        "--normalize_rows",
        action="store_true",
        help="Normalize the DINO-based weight matrix row-wise.",
    )

    parser.add_argument(
        "--enable_warmup_dino_hyperparams",
        action="store_true",
        help="Linearly warm up DINO-related lambdas and betas over warmup_steps.",
    )

    parser.add_argument(
        "--residual_projection",
        action="store_true",
        help="Use residual connection from input to DINO projection.",
    )

    parser.add_argument(
        "--residual_alpha",
        type=float,
        default=None,
        help="If set, use weighted residual connection: alpha * proj + (1 - alpha) * input.",
    )

    parser.add_argument(
        "--use_layernorm",
        action="store_true",
        help="Apply LayerNorm to projection output.",
    )

    parser.add_argument(
        "--use_symmetric_dino_weights",
        action="store_true",
        help="Use symmetric DINO weights for both branches.",
    )

    parser.add_argument(
        "--normalize_cols",
        action="store_true",
        help="Normalize the DINO-based weight matrix column-wise.",
    )

    parser.add_argument(
        "--beta_weight",
        type=float,
        default=0.0,
        help="Beta for Dino weighting. Default is 5",
    )

    parser.add_argument(
        "--lambda_dino",
        type=float,
        default=0.0,
        help="Regularizor for extra ClipImageEncoder-Dino similarity loss",
    )

    # Soft-target / distillation related args

    parser.add_argument(
        "--use_dino_similarities",
        default=False,
        action="store_true",
        help="Enable self-alignment loss from DINO fts.",
    )


    parser.add_argument(
        "--lambda_self_align",
        type=float,
        default=0.0,
        help="Regularizor for enforcing image(A) close to Dino(A) loss",
    )

    parser.add_argument(
        "--use_dino_self_align",
        default=False,
        action="store_true",
        help="Enable self-alignment loss from DINO fts.",
    )

    # parser.add_argument(
    #     "--lambda_cross",
    #     type=float,
    #     default=0.0,
    #     help="Regularizor for cross model loss",
    # )

    # parser.add_argument(
    #     "--lambda_soft_targets",
    #     type=float,
    #     default=0.0,
    #     help="Weight for the soft target loss from DINO similarities.",
    # )

    parser.add_argument(
        "--lambda_weighted_contrastive_loss",
        type=float,
        default=0.0,
        help="Weight for the weighted contrastive loss.",
    )

    parser.add_argument(
        "--use_dino_soft_targets",
        default=False,
        action="store_true",
        help="Enable soft targets from DINO similarities.",
    )

    parser.add_argument(
        "--lambda_sim_align",
        type=float,
        default=0.0,
        help="Weight for similarity alignment loss between CLIP and DINO features.",
    )

    parser.add_argument(
        "--use_dino_sim_align",
        default=False,
        action="store_true",
        help="Enable similarity alignment loss between CLIP and DINO features.",
    )

    # parser.add_argument(
    #     "--use_dino_reg",
    #     action="store_true",
    #     help="Enable DINO regularization",
    # )

    parser.add_argument(
        "--use_dino_general",
        action="store_true",
        help="Enable DINO fts",
    )

    parser.add_argument(
        "--use_coca",
        action="store_true",
        help="Enable CoCa",
    )
     
    parser.add_argument(
        "--use_CyClip",
        action="store_true",
        help="Enable CyClip",
    )
    
    parser.add_argument("--lambda_cyc_inmodal", type=float, default=0.25)      # λ1
    parser.add_argument("--lambda_cyc_crossmodal", type=float, default=0.25)   # λ2

    # parser.add_argument(
    #     "--soft_dino_to_text",
    #     action="store_true",
    #     help="Enable DINO fts --> text symmetry",
    # )

    # parser.add_argument(
    #     "--alpha",
    #     type=float,
    #     default=0.5,
    #     help="Weight for combining original CLIP loss and soft label loss",
    # )

    parser.add_argument(
        "--soft_temprature",
        type=float,
        default=0.02,
        help="Temprature for soft labels",
    )

    parser.add_argument(
        "--rho",
        type=float,
        default=0.1,
        help="Temprature for soft labels",
    )

    parser.add_argument(
        "--c_clip",
        type=float,
        default=1.0,
        help="Weight for CLIP loss",
    )
    
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to file(s) with training data. When using webdataset, multiple datasources can be combined using the `::` separator.",
    )
    parser.add_argument(
        "--train-data-upsampling-factors",
        type=str,
        default=None,
        help=(
            "When using multiple data sources with webdataset and sampling with replacement, this can be used to upsample specific data sources. "
            "Similar to --train-data, this should be a string with as many numbers as there are data sources, separated by `::` (e.g. 1::2::0.5) "
            "By default, datapoints are sampled uniformly regardless of the dataset sizes."
        ),
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to file(s) with validation data",
    )
    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Required for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--val-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Useful for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["webdataset", "csv", "synthetic", "auto"],
        default="auto",
        help="Which type of dataset to process.",
    )
    parser.add_argument(
        "--dataset-resampled",
        default=False,
        action="store_true",
        help="Whether to use sampling with replacement for webdataset shard selection.",
    )
    parser.add_argument(
        "--csv-separator",
        type=str,
        default="\t",
        help="For csv-like datasets, which separator to use.",
    )
    parser.add_argument(
        "--csv-img-key",
        type=str,
        default="filepath",
        help="For csv-like datasets, the name of the key for the image paths.",
    )
    parser.add_argument(
        "--csv-caption-key",
        type=str,
        default="title",
        help="For csv-like datasets, the name of the key for the captions.",
    )
    parser.add_argument(
        "--imagenet-val",
        type=str,
        default=None,
        help="Path to imagenet val set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet-v2",
        type=str,
        default=None,
        help="Path to imagenet v2 for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Override system default cache path for model & tokenizer file downloads.",
    )

    parser.add_argument(
    "--log-checkpoint",
    action="store_true",
    help="If set, save epoch checkpoints under logs/<name>/checkpoints and (optionally) log them to MLflow."
    )

# ---------------------------------------------------------------------
    # CLIP-blind post-training check (new)
    # ---------------------------------------------------------------------
    group = parser.add_argument_group("clip-blind")
    group.add_argument(
        "--run_clip_blind",
        action="store_true",
        help="Run CLIP-blind check on Flickr30k val after training.",
    )

    group.add_argument(
        "--clip_blind_val_key",
        type=str,
        default="flickr30k-val",
        help="Validation key to use.",
    )

    group.add_argument(
        "--clip_blind_train_key",
        type=str,
        default="train",
        help="Validation key to use.",
    )

    group.add_argument(
        "--clip_blind_dino_feats",
        type=str,
        default="/leonardo_work/EUHPC_A04_051/nxiros/Thesis/Datasets/Flick30k/Dino/flickr30k_val_dinov2_feats.pt",
    )
    group.add_argument(
        "--clip_blind_dino_index_map",
        type=str,
        default="/leonardo_work/EUHPC_A04_051/nxiros/Thesis/Datasets/Flick30k/Dino/flickr30k_val_dinov2_index_map.pt",
    )

    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="log files on local master, otherwise global master only.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--epochs-cooldown",
        type=int,
        default=None,
        help="When scheduler w/ cooldown used, perform cooldown from total_epochs - cooldown_epochs onwards.",
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument(
        "--vit-lr-decay",
        type=float,
        default=0.9,
        help="Layer-wise learning rate decay for Vision Transformers (applied from input to output blocks). "
             "Values <1.0 give smaller LRs to earlier blocks."
    )
    parser.add_argument(
    "--logit-scale-lr",
    type=float,
    default=1e-6,
    help="Very small LR for logit_scale to avoid destabilizing a good pre-trained value."
    )

    parser.add_argument(
        "--head-lr",
        type=float,
        default=1e-4,
        help="Learning rate for projection heads (text_projection, attnpool, logit_scale)."
    )

    parser.add_argument(
        "--text-lr",
        type=float,
        default=5e-5,
        help="Learning rate for the last unlocked text transformer blocks."
    )
    parser.add_argument(
        "--resnet-lr4",
        type=float,
        default=2e-5,
        help="Learning rate for ResNet layer4 if unfrozen."
    )
    parser.add_argument(
        "--resnet-lr3",
        type=float,
        default=1e-5,
        help="Learning rate for ResNet layer3 if unfrozen."
    )


    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument(
        "--momentum", type=float, default=None, help="Momentum (for timm optimizers)."
    )

    parser.add_argument(
        "--warmup", type=int, default=10000, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--opt",
        type=str,
        default="adamw",
        help="Which optimizer to use. Choices are ['adamw', or any timm optimizer 'timm/{opt_name}'].",
    )
    parser.add_argument(
        "--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.",
    )
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="cosine",
        help="LR scheduler. One of: 'cosine', 'const' (constant), 'const-cooldown' (constant w/ cooldown). Default: cosine",
    )
    parser.add_argument(
        "--lr-cooldown-end",
        type=float,
        default=0.0,
        help="End learning rate for cooldown schedule. Default: 0",
    )
    parser.add_argument(
        "--lr-cooldown-power",
        type=float,
        default=1.0,
        help="Power for polynomial cooldown schedule. Default: 1.0 (linear decay)",
    )
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=False,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--zeroshot-frequency", type=int, default=2, help="How often to run zero shot."
    )
    parser.add_argument(
        "--val-frequency",
        type=int,
        default=1,
        help="How often to run evaluation with val data.",
    )

    # parser.add_argument(
    #     "--grad_clip_norm",
    #     type=float,
    #     default=1.0,
    #     help="Gradient clipping norm.",
    # )

    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--precision",
        choices=[
            "amp",
            "amp_bf16",
            "amp_bfloat16",
            "bf16",
            "fp16",
            "pure_bf16",
            "pure_fp16",
            "fp32",
        ],
        default="amp",
        help="Floating point precision.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RN50",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--pretrained",
        default="",
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--pretrained-image",
        default=False,
        action="store_true",
        help="Load imagenet pretrained weights for image tower backbone if available.",
    )
    parser.add_argument(
        "--lock-image",
        default=False,
        action="store_true",
        help="Lock full image tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-image-unlocked-groups",
        type=int,
        default=0,
        help="Leave last n image tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-image-freeze-bn-stats",
        default=False,
        action="store_true",
        help="Freeze BatchNorm running stats in image tower for any locked layers.",
    )
    parser.add_argument(
        "--image-mean",
        type=float,
        nargs="+",
        default=None,
        metavar="MEAN",
        help="Override default image mean value of dataset",
    )
    parser.add_argument(
        "--image-std",
        type=float,
        nargs="+",
        default=None,
        metavar="STD",
        help="Override default image std deviation of of dataset",
    )
    parser.add_argument(
        "--image-interpolation",
        default=None,
        type=str,
        choices=["bicubic", "bilinear", "random"],
        help="Override default image resize interpolation",
    )
    parser.add_argument(
        "--image-resize-mode",
        default=None,
        type=str,
        choices=["shortest", "longest", "squash"],
        help="Override default image resize (& crop) mode during inference",
    )
    parser.add_argument("--aug-cfg", nargs="*", default={}, action=ParseKwargs)
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action="store_true",
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--local-loss",
        default=False,
        action="store_true",
        help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)",
    )
    parser.add_argument(
        "--gather-with-grad",
        default=False,
        action="store_true",
        help="enable full distributed gradient for feature gather",
    )
    parser.add_argument(
        "--force-image-size",
        type=int,
        nargs="+",
        default=None,
        help="Override default image size",
    )
    parser.add_argument(
        "--force-quick-gelu",
        default=False,
        action="store_true",
        help="Force use of QuickGELU activation for non-OpenAI transformer models.",
    )
    parser.add_argument(
        "--force-patch-dropout",
        default=None,
        type=float,
        help="Override the patch dropout during training, for fine tuning with no dropout near the end as in the paper",
    )
    parser.add_argument(
        "--force-custom-text",
        default=False,
        action="store_true",
        help="Force use of CustomTextCLIP model (separate text-tower).",
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action="store_true",
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )
    parser.add_argument(
        "--torchcompile",
        default=False,
        action="store_true",
        help="torch.compile() the model, requires pytorch 2.0 or later.",
    )
    parser.add_argument(
        "--trace",
        default=False,
        action="store_true",
        help="torch.jit.trace the model for inference / eval only",
    )
    parser.add_argument(
        "--accum-freq",
        type=int,
        default=1,
        help="Update the model every --acum-freq steps.",
    )
    parser.add_argument(
        "--device", default="cuda", type=str, help="Accelerator to use."
    )
    # arguments for distributed training
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend",
        default=None,
        type=str,
        help='distributed backend. "nccl" for GPU, "hccl" for Ascend NPU',
    )
    parser.add_argument(
        "--report-to",
        default="",
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']",
    )
    parser.add_argument(
        "--wandb-notes", default="", type=str, help="Notes if logging with wandb"
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="open-clip",
        help="Name of the project if logging with wandb.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged.",
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log directory, and execute from there.",
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--ddp-static-graph",
        default=False,
        action="store_true",
        help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Default random seed.")
    parser.add_argument(
        "--grad-clip-norm", type=float, default=None, help="Gradient clip."
    )
    parser.add_argument(
        "--lock-text",
        default=False,
        action="store_true",
        help="Lock full text tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-text-unlocked-layers",
        type=int,
        default=0,
        help="Leave last n text tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-text-freeze-layer-norm",
        default=False,
        action="store_true",
        help="Freeze LayerNorm running stats in text tower for any locked layers.",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=100,
        help="Log every n steps to tensorboard/console/wandb.",
    )
    parser.add_argument(
        "--coca-caption-loss-weight",
        type=float,
        default=2.0,
        help="Weight assigned to caption loss in CoCa.",
    )
    parser.add_argument(
        "--coca-contrastive-loss-weight",
        type=float,
        default=1.0,
        help="Weight assigned to contrastive loss when training CoCa.",
    )
    parser.add_argument(
        "--remote-sync",
        type=str,
        default=None,
        help="Optinoally sync with a remote path specified by this arg",
    )
    parser.add_argument(
        "--remote-sync-frequency",
        type=int,
        default=300,
        help="How frequently to sync to a remote directly if --remote-sync is not None.",
    )
    parser.add_argument(
        "--remote-sync-protocol",
        choices=["s3", "fsspec"],
        default="s3",
        help="How to do the remote sync backup if --remote-sync is not None.",
    )
    parser.add_argument(
        "--delete-previous-checkpoint",
        default=False,
        action="store_true",
        help="If true, delete previous checkpoint after storing a new one.",
    )
    parser.add_argument(
        "--distill-model",
        default=None,
        help="Which model arch to distill from, if any.",
    )
    parser.add_argument(
        "--distill-pretrained",
        default=None,
        help="Which pre-trained weights to distill from, if any.",
    )
    parser.add_argument(
        "--use-bnb-linear",
        default=None,
        help="Replace the network linear layers from the bitsandbytes library. "
        "Allows int8 training/inference, etc.",
    )
    parser.add_argument(
        "--siglip",
        default=False,
        action="store_true",
        help="Use SigLip (sigmoid) loss.",
    )
    parser.add_argument(
        "--loss-dist-impl",
        default=None,
        type=str,
        help="A string to specify a specific distributed loss implementation.",
    )

    args = parser.parse_args(args)

    if "timm" not in args.opt:
        # set default opt params based on model name (only if timm optimizer not used)
        default_params = get_default_params(args.model)
        for name, val in default_params.items():
            if getattr(args, name) is None:
                setattr(args, name, val)

    return args
