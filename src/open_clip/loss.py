from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

import logging


def gather_features(
    image_features,
    text_features,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
    use_horovod=False,
):
    assert (
        has_distributed
    ), "torch.distributed did not import correctly, please use a PyTorch version with support."
    if use_horovod:
        assert hvd is not None, "Please install horovod"
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(
                    all_image_features.chunk(world_size, dim=0)
                )
                gathered_text_features = list(
                    all_text_features.chunk(world_size, dim=0)
                )
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(
                torch.distributed.nn.all_gather(image_features), dim=0
            )
            all_text_features = torch.cat(
                torch.distributed.nn.all_gather(text_features), dim=0
            )
        else:
            gathered_image_features = [
                torch.zeros_like(image_features) for _ in range(world_size)
            ]
            gathered_text_features = [
                torch.zeros_like(text_features) for _ in range(world_size)
            ]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = (
                    logit_scale * all_image_features @ all_text_features.T
                )
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(
            image_features, text_features, logit_scale
        )

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


from types import SimpleNamespace
def _getarg(args, name, default):
    return getattr(args, name, default) if args is not None else default

def compute_student_tau(logit_scale_tensor):
    # logit_scale_tensor could be either raw ln-scale (≈2–5) or multiplicative (≈10–100)
    val = logit_scale_tensor.detach()
    # If this is a scalar tensor from model_out["logit_scale"], it's usually RAW ln-scale
    # But if upstream already did .exp(), you'll get multiplicative. Handle both.
    # Heuristic: values > 10 are likely multiplicative.
    scale_mult = torch.where(val > 10, val, val.exp())           # multiplicative scale
    scale_mult = torch.clamp(scale_mult, max=100)                # match OpenCLIP cap
    tau_s = (1.0 / scale_mult).clamp(min=0.008, max=0.02)        # safe band for FT
    return tau_s

def _offdiag_mask(B, device):
    eye = torch.eye(B, device=device, dtype=torch.bool)
    return ~eye

def _offdiag_quantile(mat: torch.Tensor, q: float) -> torch.Tensor:
    """Quantile over off-diagonal entries (scalar)."""
    B = mat.shape[0]
    mask = _offdiag_mask(B, mat.device)
    vals = mat[mask].to(torch.float32)
    return torch.quantile(vals, q=q).to(mat.dtype)
    


class ClipLossWithDINOEnhancements(nn.Module):
    def __init__(
        self,
        local_loss: bool = False,
        gather_with_grad: bool = False,
        cache_labels: bool = False,
        rank: int = 0,
        world_size: int = 1,
        use_horovod: bool = False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        self.image_to_dino_proj = None

        self.prev_num_logits = 0
        self.labels = {}  # cache per-device

    # ------------------- Projection helper -------------------
    def init_proj(
        self,
        embed_dim,
        dino_dim,
        device,
        projection_type="mlp",
        residual=False,
        layernorm=False,
    ):
        if self.image_to_dino_proj is None:
            if projection_type == "linear":
                proj = nn.Linear(embed_dim, dino_dim)
            elif projection_type == "mlp":
                hidden_dim = (embed_dim + dino_dim) // 2
                layers = [
                    nn.Linear(embed_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, dino_dim),
                ]
                if layernorm:
                    layers.append(nn.LayerNorm(dino_dim))
                proj = nn.Sequential(*layers)
            else:
                raise ValueError(f"Unknown projection_type: {projection_type}")
            self.image_to_dino_proj = proj.to(device)

    # --------------------------- helpers -------------------------------------
    def get_ground_truth(self, device: torch.device, num_logits: int) -> torch.Tensor:
        dev_key = str(device)
        if self.prev_num_logits != num_logits or dev_key not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels += num_logits * self.rank
            if self.cache_labels:
                self.labels[dev_key] = labels
            self.prev_num_logits = num_logits
        else:
            labels = self.labels[dev_key]
        return labels

    def get_logits(self, image_features: torch.Tensor, text_features: torch.Tensor, logit_scale: torch.Tensor):
        if self.world_size > 1 and gather_features is not None:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )
            if self.local_loss:
                logits_per_image = logit_scale * (image_features @ all_text_features.T)
                logits_per_text  = logit_scale * (text_features  @ all_image_features.T)
            else:
                logits_per_image = logit_scale * (all_image_features @ all_text_features.T)
                logits_per_text  = logits_per_image.T
        else:
            logits_per_image = logit_scale * (image_features @ text_features.T)
            logits_per_text  = logit_scale * (text_features  @ image_features.T)
        return logits_per_image, logits_per_text

    # 1) VICReg-style variance on text (encourage per-dim std ≥ γ)
    # def variance_loss(Z, gamma=1.0, eps=1e-4):
    #     # Z: [B,D], centered
    #     Zc = Z - Z.mean(0, keepdims=True)
    #     std = Zc.pow(2).mean(0).sqrt() + eps
    #     return torch.relu(gamma - std).mean()

    # # 2) Off-diagonal covariance penalty (whitening-ish)
    # def cov_offdiag_loss(Z):
    #     Zc = Z - Z.mean(0, keepdims=True)
    #     C  = (Zc.T @ Zc) / (Zc.size(0) - 1)            # [D,D]
    #     off = C - torch.diag(torch.diag(C))
    #     return (off.pow(2).sum()) / (C.size(0)**2)


    # ------------------------------ forward ----------------------------------
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor,
        dino_features: torch.Tensor | None = None,
        args=None,                      # carries the knobs (later names)
        output_dict: bool = False,
    ):
        device = image_features.device
        B = image_features.shape[0]
        g = getattr

        # ====== NEW PROJECTION KNOBS ======
        use_projection      = g(args, "use_projection", True)
        projection_type     = g(args, "projection_type", "mlp")
        use_layernorm       = g(args, "use_layernorm", False)
        residual_projection = g(args, "residual_projection", False)
        residual_alpha      = g(args, "residual_alpha", None)

        # ----- core CLIP logits -----
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
        labels = self.get_ground_truth(device, B)

        # classic CLIP contrastive CE
        classic_loss = 0.5 * (
            F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)
        )

        # If we need a projection (e.g. soft targets or align losses)
        if dino_features is not None and use_projection:
            self.init_proj(
                embed_dim=image_features.size(-1),
                dino_dim=dino_features.size(-1),
                device=device,
                projection_type=projection_type,
                layernorm=use_layernorm,
            )
            raw_proj = self.image_to_dino_proj(image_features)
            if residual_projection:
                if residual_alpha is None:
                    if raw_proj.shape == image_features.shape:
                        image_proj = F.normalize(image_features + raw_proj, dim=-1)
                    else:
                        image_proj = F.normalize(raw_proj, dim=-1)
                else:
                    if raw_proj.shape == image_features.shape:
                        image_proj = F.normalize(
                            residual_alpha * image_features + (1 - residual_alpha) * raw_proj, dim=-1
                        )
                    else:
                        image_proj = F.normalize(raw_proj, dim=-1)
            else:
                image_proj = F.normalize(raw_proj, dim=-1)
        else:
            image_proj = F.normalize(image_features, dim=-1)

        # =========================
        # Soft loss (KL teacher)
        # =========================
        soft_loss = torch.zeros((), device=device)
        lambda_soft = float(g(args, "lambda_soft", 0.0))
        soft_mode  = g(args, "soft_mode", "none")

        if lambda_soft > 0.0 and soft_mode == "kl_teacher" and dino_features is not None:
            #normalize
            Zs = F.normalize(image_proj, dim=-1)
            Dn = F.normalize(dino_features,  dim=-1)

            # temps
            student_temp = g(args, "student_temp", None)
            # if student_temp is not None:
            #     tau_s = torch.as_tensor(float(student_temp), device=device, dtype=Zs.dtype)
            # else:
            tau_s = compute_student_tau(logit_scale)

            teacher_temp = float(g(args, "teacher_temp", 0.15))
            tau_t = torch.as_tensor(teacher_temp, device=device, dtype=Zs.dtype)

            # sims
            S_student = (Zs @ Zs.T) / tau_s
            S_teacher = (Dn @ Dn.T) / tau_t

            # mask diag in teacher
            eye = torch.eye(B, device=device, dtype=torch.bool)
            S_teacher = S_teacher.masked_fill(eye, float('-inf'))

            with torch.no_grad():
                q = F.softmax(S_teacher, dim=1)

            log_p = F.log_softmax(S_student, dim=1)
            soft_loss_imgimg = F.kl_div(log_p, q, reduction="batchmean")
            soft_loss = soft_loss_imgimg

            # optional text–text consistency
            if bool(g(args, "soft_dino_to_text", False)) and float(g(args, "text_lambda", 0.2)) > 0.0:
                text_lambda       = float(g(args, "text_lambda", 0.2))

                #--------------------OLD---------------------
                text_student_temp = float(g(args, "text_student_temp", 0.05))
                Tn = F.normalize(text_features, dim=-1)
                tau_s_txt = torch.as_tensor(text_student_temp, device=device, dtype=Tn.dtype)
                S_student_TT = (Tn @ Tn.T) / tau_s_txt
                log_p_TT = F.log_softmax(S_student_TT, dim=1)
                soft_loss_TT = F.kl_div(log_p_TT, q, reduction="batchmean")
                soft_loss = soft_loss + text_lambda * soft_loss_TT
                
                # #--------------------NEW---------------------
                # # Student (TEXT-TEXT) uses the SAME adaptive student tau as images
                # # tau_s is already defined above as:
                # #   tau_s = compute_student_tau(logit_scale)
                # Tn = F.normalize(text_features, dim=-1)   # [B,D]
                # S_student_TT = (Tn @ Tn.T) / tau_s        # [B,B] text–text "logits"

                # # Teacher targets: reuse q from the DINO image–image teacher (already diag-masked)
                # # q = softmax(S_teacher) with S_teacher built from DINO & masked diag to -inf

                # log_p_TT = F.log_softmax(S_student_TT, dim=1)
                # soft_loss_TT = F.kl_div(log_p_TT, q, reduction="batchmean")

                # # Accumulate into the same soft_loss bucket
                # soft_loss = soft_loss + text_lambda * soft_loss_TT


        # ---------- Denominator-modulated CLIP CE (DINO-guided) ----------
        lambda_weighted = float(g(args, "lambda_weighted", 0.0))  # new knob
        rho        = float(getattr(args, "rho", 0.1))                 # 5–20% typical
        c_clip     = float(getattr(args, "c_clip", 1.0))         # clip for r-hat
        weighted_loss = torch.zeros((), device=device)
        weight_text_sym = bool(g(args, "weight_text_symmetry", False)) # NEW knob
    
        if lambda_weighted > 0.0 and dino_features is not None and B > 1:
            with torch.no_grad():
                # DINO cosine -> dissimilarity r in [0,2], mask diag
                Dn = F.normalize(dino_features, dim=-1)
                dino_sims = (Dn @ Dn.T).clamp(-1, 1)
                r = (1.0 - dino_sims)
                eye = torch.eye(B, device=device, dtype=torch.bool)
                r = r.masked_fill(eye, 0.0)

            # ---------------- image -> text (row-wise) ----------------
            # p-centering under *unmodified* row probs
            p_img_base = F.softmax(logits_per_image, dim=1)                        # [B,B]
            r_hat_img  = r - (p_img_base * r).sum(dim=1, keepdim=True)             # [B,B]
            r_hat_img  = r_hat_img.clamp(min=-c_clip, max=c_clip)

            # choose beta from rho × (typical row std of logits)
            with torch.no_grad():
                row_std_img = logits_per_image.float().std(dim=1)                  # [B]
                sigma_img   = torch.median(row_std_img).clamp(min=1e-6)            # scalar
            beta_img = (rho * sigma_img / c_clip).item()

            # logit offsets Δ = β * r̂ (diag=0), then CE
            Delta_img          = (beta_img * r_hat_img).masked_fill(eye, 0.0)
            logits_img_tilde   = logits_per_image + Delta_img
            ce_img_den         = F.cross_entropy(logits_img_tilde, labels)

            # ---------------- text -> image (column-wise; optional) ----------------
            if weight_text_sym:
                # For text->image, p-center over images for each text (i.e., rows of logits_per_text)
                p_txt_base = F.softmax(logits_per_text, dim=1)                     # [B,B]
                rT         = r.T                                                   # reuse same DINO metric, transposed
                r_hat_txt  = rT - (p_txt_base * rT).sum(dim=1, keepdim=True)       # [B,B]
                r_hat_txt  = r_hat_txt.clamp(min=-c_clip, max=c_clip)

                with torch.no_grad():
                    row_std_txt = logits_per_text.float().std(dim=1)               # [B]
                    sigma_txt   = torch.median(row_std_txt).clamp(min=1e-6)
                beta_txt = (rho * sigma_txt / c_clip).item()

                Delta_txt        = (beta_txt * r_hat_txt).masked_fill(eye, 0.0)
                logits_txt_tilde = logits_per_text + Delta_txt
            else:
                logits_txt_tilde = logits_per_text

            ce_txt_den = F.cross_entropy(logits_txt_tilde, labels)
            weighted_loss = 0.5 * (ce_img_den + ce_txt_den)
        else:
            # disabled / not applicable (e.g., B==1)
            weighted_loss = torch.zeros((), device=device, dtype=logits_per_image.dtype)

        total_loss = (
            float(g(args, "lambda_original", 1.0)) * classic_loss
            + lambda_soft * soft_loss
            + lambda_weighted * weighted_loss
        )

        # === Diagnostics (cheap scalar summaries) ===
        dbg = {}

        try:
            if lambda_weighted > 0.0 and dino_features is not None and B > 1:
                with torch.no_grad():
                    # Base probabilities (before modulation)
                    p_img_base = F.softmax(logits_per_image, dim=1)
                    p_txt_base = F.softmax(logits_per_text,  dim=1)

                    # Modulated probabilities
                    p_img_tilde = F.softmax(logits_img_tilde, dim=1)
                    p_txt_tilde = F.softmax(logits_txt_tilde, dim=1)

                    # ----- p-centering sanity (should be ~0) -----
                    # For image->text we centered r_hat_img with p_img_base
                    pc_err_img = (p_img_base * r_hat_img).sum(dim=1).abs().mean().item()
                    # For text->image, if sym enabled, we centered r_hat_txt with p_txt_base
                    pc_err_txt = (p_txt_base * r_hat_txt).sum(dim=1).abs().mean().item() if weight_text_sym else 0.0

                    # ----- diagonal sanity (should be 0) -----
                    diag_max_img = r_hat_img.diag().abs().max().item()
                    diag_max_txt = r_hat_txt.diag().abs().max().item() if weight_text_sym else 0.0

                    # ----- Δ stats (logit offsets) -----
                    Delta_img_abs = Delta_img.abs()
                    delta_img_max = Delta_img_abs.max().item()
                    delta_img_mean = Delta_img_abs.mean().item()
                    delta_img_std = Delta_img_abs.std().item()

                    if weight_text_sym:
                        Delta_txt_abs = (Delta_txt.abs())
                        delta_txt_max = Delta_txt_abs.max().item()
                        delta_txt_mean = Delta_txt_abs.mean().item()
                        delta_txt_std = Delta_txt_abs.std().item()
                    else:
                        delta_txt_max = delta_txt_mean = delta_txt_std = 0.0

                    # ----- probability shift L1 per row -----
                    l1_img = (p_img_tilde - p_img_base).abs().sum(dim=1).mean().item()
                    l1_txt = (p_txt_tilde - p_txt_base).abs().sum(dim=1).mean().item()

                    # ----- corr(r_hat, Δp) : are we pushing where r_hat is positive? -----
                    def rowwise_corr(a, b, eps=1e-9):
                        # a,b: [B,B]; compute Pearson corr per row then mean over rows
                        a = a - a.mean(dim=1, keepdim=True)
                        b = b - b.mean(dim=1, keepdim=True)
                        num = (a * b).sum(dim=1)
                        den = (a.pow(2).sum(dim=1).sqrt() * b.pow(2).sum(dim=1).sqrt() + eps)
                        return (num / den).mean().item()

                    corr_img = rowwise_corr(r_hat_img, (p_img_tilde - p_img_base))
                    corr_txt = rowwise_corr(r_hat_txt, (p_txt_tilde - p_txt_base)) if weight_text_sym else 0.0

                    # ----- CE deltas (how much CE changed due to modulation) -----
                    ce_img_base = F.cross_entropy(logits_per_image, labels).item()
                    ce_txt_base = F.cross_entropy(logits_per_text,  labels).item()
                    ce_img_mod  = F.cross_entropy(logits_img_tilde, labels).item()
                    ce_txt_mod  = F.cross_entropy(logits_txt_tilde, labels).item()

                    # ----- up/down weighted counts by r_hat sign (off-diagonal only) -----
                    offdiag = ~torch.eye(B, device=device, dtype=torch.bool)
                    pos_frac_img = (r_hat_img[offdiag] > 0).float().mean().item()
                    neg_frac_img = 1.0 - pos_frac_img
                    if weight_text_sym:
                        pos_frac_txt = (r_hat_txt[offdiag] > 0).float().mean().item()
                        neg_frac_txt = 1.0 - pos_frac_txt
                    else:
                        pos_frac_txt = neg_frac_txt = 0.0

                # Pack into dict
                dbg.update({
                    "pc_err_img": pc_err_img,                 # should be ~0
                    "pc_err_txt": pc_err_txt,                 # ~0 if sym
                    "diag_max_img": diag_max_img,             # should be 0
                    "diag_max_txt": diag_max_txt,             # 0 if sym
                    "delta_img_max": delta_img_max,
                    "delta_img_mean": delta_img_mean,
                    "delta_img_std": delta_img_std,
                    "delta_txt_max": delta_txt_max,
                    "delta_txt_mean": delta_txt_mean,
                    "delta_txt_std": delta_txt_std,
                    "l1_prob_shift_img": l1_img,
                    "l1_prob_shift_txt": l1_txt,
                    "corr_rhat_dprob_img": corr_img,          # expect > 0
                    "corr_rhat_dprob_txt": corr_txt,          # > 0 if sym
                    "ce_img_base": ce_img_base,
                    "ce_txt_base": ce_txt_base,
                    "ce_img_mod": ce_img_mod,
                    "ce_txt_mod": ce_txt_mod,
                    "pos_frac_img": pos_frac_img,
                    "neg_frac_img": neg_frac_img,
                    "pos_frac_txt": pos_frac_txt,
                    "neg_frac_txt": neg_frac_txt,
                    "beta_img": beta_img,
                    "beta_txt": beta_txt if weight_text_sym else 0.0,
                    "rho": rho,
                    "clip_c": c_clip,
                })

                # Optional: print every K steps on rank 0
                k = int(getattr(args, "dbg_print_every", 0) or 0)
                if k > 0 and getattr(self, "rank", 0) == 0 and (int(getattr(args, "global_step", 0)) % k == 0):
                    print(
                        f"[DBG] pc_err(img/txt)={pc_err_img:.3e}/{pc_err_txt:.3e} | "
                        f"Δmax(img/txt)={delta_img_max:.3f}/{delta_txt_max:.3f} | "
                        f"L1Δp(img/txt)={l1_img:.3f}/{l1_txt:.3f} | "
                        f"corr(r̂,Δp)(img/txt)={corr_img:.3f}/{corr_txt:.3f} | "
                        f"CE_base(img/txt)={ce_img_base:.3f}/{ce_txt_base:.3f} -> "
                        f"CE_mod(img/txt)={ce_img_mod:.3f}/{ce_txt_mod:.3f} | "
                        f"β(img/txt)={beta_img:.3f}/{(beta_txt if weight_text_sym else 0.0):.3f}"
                    )
        except Exception as e:
            # Never break training due to debug code
            if getattr(self, "rank", 0) == 0:
                print(f"[DBG] diagnostics failed: {e}")
            dbg = {}

        # If caller asked for output_dict, include debug
        if output_dict:
            # merge into the dict you already return at the end
            extra = {
                "total_loss": total_loss,
                "classic_loss": classic_loss,
                "soft_loss": soft_loss,
                "weighted_loss": weighted_loss,
                "dbg": dbg,
            }
            return extra


class SigLipLoss(nn.Module):
    """Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """

    def __init__(
        self,
        cache_labels: bool = False,
        rank: int = 0,
        world_size: int = 1,
        dist_impl: Optional[str] = None,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.dist_impl = (
            dist_impl or "bidir"
        )  # default to bidir exchange for now, this will likely change
        assert self.dist_impl in ("bidir", "shift", "reduce", "gather")

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(
        self, device, dtype, num_logits, negative_only=False
    ) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(
        self,
        image_features,
        text_features,
        logit_scale,
        logit_bias=None,
        negative_only=False,
    ):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(
        self, image_features, text_features, logit_scale, logit_bias, output_dict=False
    ):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        if self.world_size > 1:
            if self.dist_impl == "bidir":
                right_rank = (self.rank + 1) % self.world_size
                left_rank = (self.rank - 1 + self.world_size) % self.world_size
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )
                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right
                    )
                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            elif self.dist_impl == "shift":
                right_rank = (self.rank + 1) % self.world_size
                left_rank = (self.rank - 1 + self.world_size) % self.world_size
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_right,
                    )
                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left
            elif self.dist_impl == "reduce":
                for i in range(self.world_size):
                    text_from_other = torch.distributed.nn.all_reduce(
                        text_features * (self.rank == i),
                        torch.distributed.ReduceOp.SUM,
                    )
                    loss += float(i != self.rank) * self._loss(
                        image_features,
                        text_from_other,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            elif self.dist_impl == "gather":
                all_text = torch.distributed.nn.all_gather(text_features)
                for i in range(self.world_size):
                    loss += float(i != self.rank) * self._loss(
                        image_features,
                        all_text[i],
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                assert False

        return {"contrastive_loss": loss} if output_dict else loss




class CoCaLoss(ClipLoss):
    def __init__(
        self,
        caption_loss_weight,
        clip_loss_weight,
        pad_id=0,  # pad_token for open_clip custom tokenizer
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod,
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(
        self,
        image_features,
        text_features,
        logits,
        labels,
        logit_scale,
        output_dict=False,
    ):
        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss
        else:
            clip_loss = torch.tensor(0, device=logits.device)

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


class CyCLIPLoss(ClipLoss):
    """
    CyCLIP (Goel et al., NeurIPS 2022):
      L = L_CLIP + λ1 * L_inmodal + λ2 * L_crossmodal
    with cosine-similarity-based pairwise consistency terms.
    """

    def __init__(
        self,
        lambda_inmodal: float = 0.25,      # λ1 (paper default)
        lambda_crossmodal: float = 0.25,   # λ2 (paper default)
        local_loss: bool = False,
        gather_with_grad: bool = False,
        cache_labels: bool = False,
        rank: int = 0,
        world_size: int = 1,
        use_horovod: bool = False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod,
        )
        self.lambda_inmodal = lambda_inmodal
        self.lambda_crossmodal = lambda_crossmodal

    def _get_global_features(self, image_features: torch.Tensor, text_features: torch.Tensor):
        """Match ClipLoss.get_logits semantics:
        - if world_size==1: use local
        - if world_size>1 and local_loss==False: use all-gathered (global)
        - if world_size>1 and local_loss==True: keep local
        """
        if self.world_size > 1 and not self.local_loss:
            all_i, all_t = gather_features(
                image_features,
                text_features,
                local_loss=False,  # global features
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )
            return all_i, all_t
        return image_features, text_features

    @staticmethod
    def _cosine_normalize(x: torch.Tensor) -> torch.Tensor:
        # Do sims in float32 for AMP stability
        return F.normalize(x.float(), dim=-1)

    def forward(self, image_features, text_features, logit_scale, output_dict: bool = False):
        device = image_features.device

        # ----- Standard CLIP contrastive loss (unchanged) -----
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
        labels = self.get_ground_truth(device, logits_per_image.shape[0])
        clip_loss = 0.5 * (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        )

        # ----- CyCLIP consistency terms -----
        I_all, T_all = self._get_global_features(image_features, text_features)
        I = self._cosine_normalize(I_all)
        T = self._cosine_normalize(T_all)

        # Pairwise cosine similarity matrices (float32 for numerical stability)
        S_ii = I @ I.T        # image-image
        S_tt = T @ T.T        # text-text
        S_it = I @ T.T        # image-text
        S_ti = S_it.T         # text-image

        # Cross-modal consistency: || <Ii,Tk> - <Ik,Ti> ||^2
        L_cross = (S_it - S_ti).pow(2).mean()
        # In-modal consistency:    || <Ii,Ik> - <Ti,Tk> ||^2
        L_inmod = (S_ii - S_tt).pow(2).mean()

        total = clip_loss + self.lambda_inmodal * L_inmod + self.lambda_crossmodal * L_cross

        if output_dict:
            return {
                "total_loss": total,
                "clip_loss": clip_loss,
                "inmodal_cyclic": L_inmod,
                "crossmodal_cyclic": L_cross,
                "lambda_inmodal": self.lambda_inmodal,
                "lambda_crossmodal": self.lambda_crossmodal,
            }
        return total

class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return (
            -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1))
            .sum(dim=1)
            .mean(dim=0)
        )

    def forward(
        self,
        image_features,
        text_features,
        logit_scale,
        dist_image_features,
        dist_text_features,
        dist_logit_scale,
        output_dict=False,
    ):
        logits_per_image, logits_per_text = self.get_logits(
            image_features, text_features, logit_scale
        )

        dist_logits_per_image, dist_logits_per_text = self.get_logits(
            dist_image_features, dist_text_features, dist_logit_scale
        )

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image)
            + self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(
    left_rank, right_rank, tensor_to_left, tensor_to_right, group=None
):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv(
        [send_op_right, send_op_left, recv_op_right, recv_op_left]
    )
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (
            NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),
        )


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(
            left_rank, right_rank, tensor_to_left, tensor_to_right, group=group
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + NeighbourExchangeBidir.apply(
            ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs
        )


def neighbour_exchange_bidir_with_grad(
    left_rank, right_rank, tensor_to_left, tensor_to_right, group=None
):
    return NeighbourExchangeBidir.apply(
        left_rank, right_rank, group, tensor_to_left, tensor_to_right
    )


