# helper.py
import os
import re
import glob
import math
import json
import random
import subprocess
from typing import List, Tuple, Dict, Any, Optional
import logging

import numpy as np
import torch

try:
    import mlflow
except ImportError:
    mlflow = None


# ---- local utils / constants -------------------------------------------------

# Add import for get_input_dtype
_ALLOWED = re.compile(r"[^A-Za-z0-9_. :/\\-]+")

def _mlflow_safe(s: str) -> str:
    # swap mathematical symbols to ASCII mnemonics first
    s = (
        s.replace("≥", "ge")
         .replace("≤", "le")
         .replace(">", "gt")
         .replace("<", "lt")
         .replace("=", "eq")
    )
    # replace any remaining disallowed chars with "_"
    return _ALLOWED.sub("_", s)

# In train_one_epoch, add gradient checking
def check_text_gradients(model) -> bool:
    text_has_grad = False
    for name, param in model.named_parameters():
        if "transformer" in name or "text" in name:
            if param.requires_grad and param.grad is not None:
                if param.grad.abs().max().item() > 1e-8:
                    text_has_grad = True
                    break
    return text_has_grad

def get_input_dtype(precision: str):
    p = (precision or "").lower()
    if p in ("amp", "fp16", "half"):
        return torch.float16
    if p in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float32

LATEST_CHECKPOINT_NAME = "epoch_latest.pt"

def random_seed(seed: int = 42, rank: int = 0) -> None:
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def natural_key(string_: str):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\\d+)", string_.lower())]

def _canon_path_local(p: str) -> str:
    p = os.path.realpath(str(p))
    p = os.path.normpath(p)
    root, ext = os.path.splitext(p)
    return root + ext.lower()

def get_latest_checkpoint(path: str, remote: bool):
    # as written, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(
            ["aws", "s3", "ls", path + "/"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [
            os.path.join(path, x.split(" ")[-1])
            for x in result.stdout.decode().split("\\n")[:-1]
        ]
    else:
        checkpoints = glob.glob(path + "**/*.pt", recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None

def _batch_to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    if isinstance(x, (list, tuple)):
        return type(x)(_batch_to_device(t, device) for t in x)
    if isinstance(x, dict):
        return {k: _batch_to_device(v, device) for k, v in x.items()}
    return x

def _safe_to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    if isinstance(x, (list, tuple)):
        return type(x)(_safe_to_device(t, device) for t in x)
    if isinstance(x, dict):
        return {k: _safe_to_device(v, device) for k, v in x.items()}
    return x

@torch.no_grad()
def _encode_clip_images_and_paths(model, dataloader, device):
    """
    Works with CsvDataset eval batches shaped as (images, texts, paths).
    Returns:
      clip_Z:  [M, D] L2-normalized
      paths:  List[str] canonicalized with _canon_path_local
    """
    model_w = model.module if hasattr(model, "module") else model
    model_w.eval()

    embs = []
    paths_all = []

    for batch in dataloader:
        # Expect (images, texts, paths)
        assert isinstance(batch, (tuple, list)) and len(batch) >= 2, \
            "Eval batch must be tuple/list: (images, texts[, paths])."
        images = batch[0]
        paths = batch[2] if len(batch) >= 3 else None
        if paths is None:
            raise RuntimeError("Eval batch missing file paths; CsvDataset should return them in position 3.")

        images = _safe_to_device(images, device)
        feats = model_w.encode_image(images)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        embs.append(feats.cpu())
        # canonicalize paths for robust matching
        paths_all.extend([_canon_path_local(p) for p in paths])

    clip_Z = torch.cat(embs, dim=0) if embs else torch.empty(0)
    return clip_Z, paths_all

def _build_dino_path_maps(dino_index_map_obj: dict):
    """
    Accepts either:
      - dict {path -> idx}, or
      - dict with key 'path_to_index' mapping paths->idx, or
      - dict with key 'basename_to_index' (fallback)
    Returns:
      path2idx (canon paths), base2idx (basenames)
    """
    if not isinstance(dino_index_map_obj, dict):
        raise RuntimeError("DINO index map must be a dict.")

    # direct dict (path->idx), or wrapped
    if "path_to_index" in dino_index_map_obj:
        p2i_raw = dino_index_map_obj["path_to_index"]
    else:
        p2i_raw = {k: v for k, v in dino_index_map_obj.items() if isinstance(v, (int, np.integer))}

    path2idx = {_canon_path_local(k): int(v) for k, v in p2i_raw.items()}

    # optional basename map if provided (or build one)
    if "basename_to_index" in dino_index_map_obj and isinstance(dino_index_map_obj["basename_to_index"], dict):
        base2idx = {os.path.basename(k): int(v) for k, v in dino_index_map_obj["basename_to_index"].items()}
    else:
        base2idx = {os.path.basename(k): int(v) for k, v in path2idx.items()}

    return path2idx, base2idx

def _dedup_by_path(clip_Z: torch.Tensor, paths: List[str]):
    """
    Deduplicate embeddings by canonical path (keep first occurrence).
    Returns:
      clip_Z_u, paths_u, idx_keep
    """
    seen = {}
    idx_keep = []
    for i, p in enumerate(paths):
        if p not in seen:
            seen[p] = i
            idx_keep.append(i)
    if len(idx_keep) == len(paths):
        return clip_Z, paths, list(range(len(paths)))
    idx_keep_t = torch.tensor(idx_keep, dtype=torch.long)
    return clip_Z[idx_keep_t], [paths[i] for i in idx_keep_t.tolist()], idx_keep_t.tolist()

def _align_dino_feats_to_paths(dino_feats: torch.Tensor, path2idx: dict, base2idx: dict, paths: List[str]):
    """
    Align DINO feats to CLIP order using canonical full path first, basename fallback second.
    Returns:
      dino_Z [K, D], keep_idx [K], hit_path, hit_base, miss
    """
    out = []
    keep_idx = []
    hit_path = hit_base = miss = 0
    for i, p in enumerate(paths):
        j = path2idx.get(p, None)
        if j is None:
            j = base2idx.get(os.path.basename(p), None)
            if j is None:
                miss += 1
                continue
            else:
                hit_base += 1
        else:
            hit_path += 1
        out.append(dino_feats[j])
        keep_idx.append(i)
    if not out:
        return None, [], hit_path, hit_base, miss
    Z = torch.stack(out, dim=0).to(torch.float32)
    Z = Z / Z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return Z, keep_idx, hit_path, hit_base, miss

def _pair_stats(clip_Z: torch.Tensor, dino_Z: torch.Tensor,
                thresholds: List[Tuple[float, float]]) -> dict:
    """
    Returns:
      {
        "total_pairs": int,
        "thresholds": [(clip_min, dino_max), ...],
        "results": {
          "clip≥{c}_dino≤{d}": {
              "count": int,                  # absolute # of blind pairs
              "percent": float,              # % of all pairs that are blind
              "clip_high_count": int,        # # pairs with CLIP ≥ c
              "relative_percent": float      # % of CLIP≥c pairs that are blind (the requested metric)
          },
          ...
        },
        "top_pairs": [{i,j,clip_sim,dino_sim,gap}, ...]
      }
    """
    # cosine similarity matrices (assumes L2-normalized rows)
    cs = (clip_Z @ clip_Z.t())
    ds = (dino_Z  @ dino_Z.t())

    # use upper triangle without diagonal
    iu   = torch.triu_indices(cs.size(0), cs.size(1), offset=1)
    cs_u = cs[iu[0], iu[1]]
    ds_u = ds[iu[0], iu[1]]
    gap  = cs_u - ds_u  # positive => CLIP > DINO

    total_pairs = int(cs_u.numel())
    out = {"total_pairs": total_pairs, "results": {}, "thresholds": thresholds}

    for (cmin, dmax) in thresholds:
        clip_high_mask = (cs_u >= cmin)
        blind_mask     = clip_high_mask & (ds_u <= dmax)

        clip_high_count = int(clip_high_mask.sum().item())
        blind_count     = int(blind_mask.sum().item())

        overall_percent   = 100.0 * blind_count / (total_pairs or 1)
        relative_percent  = 100.0 * blind_count / (clip_high_count or 1)

        key = f"clip≥{cmin}_dino≤{dmax}"
        out["results"][key] = {
            "count": blind_count,
            "percent": overall_percent,
            "clip_high_count": clip_high_count,
            "relative_percent": relative_percent
        }

    # rank top gaps for inspection (largest CLIP - DINO)
    topk = min(200, total_pairs)
    if topk > 0:
        top_idx = torch.topk(gap, k=topk).indices.cpu().tolist()
        pairs = list(zip(iu[0].cpu().tolist(), iu[1].cpu().tolist()))
        out["top_pairs"] = [
            {"i": int(pairs[r][0]), "j": int(pairs[r][1]),
             "clip_sim": float(cs_u[r]), "dino_sim": float(ds_u[r]),
             "gap": float(gap[r])}
            for r in top_idx
        ]
    else:
        out["top_pairs"] = []

    return out

def _run_clip_blind_on_split(split_key: str,
                             data: dict,
                             model,
                             device,
                             dino_feats_path: str,
                             dino_index_map_path: str,
                             checkpoint_path: str,
                             thresholds: List[Tuple[float, float]],
                             use_mlflow: bool,
                             split_alias: str):
    """
    split_key: key inside `data` (e.g., "flickr30k-val" or "flickr30k-train")
    split_alias: short label for logging/artifacts ("val" / "train" / custom)
    """
    if split_key not in data:
        logging.info(f"[CLIP-blind] Skipped: '{split_key}' not present in data.")
        return None

    logging.info(f"[CLIP-blind] Running on '{split_key}' …")

    # 1) dataloader
    dl = data[split_key].dataloader
    val_loader = getattr(dl, "loader", dl)

    # 2) CLIP encodings + canonical paths
    clip_Z, paths = _encode_clip_images_and_paths(model, val_loader, device)

    # 3) dedup by path
    clip_Z, paths, keep_clip = _dedup_by_path(clip_Z, paths)

    # 4) Load DINO feats + index maps (CPU, L2-normalize)
    dino_feats = torch.load(dino_feats_path, map_location="cpu")
    if dino_feats.dtype != torch.float32:
        dino_feats = dino_feats.to(torch.float32)
    dino_feats = dino_feats / dino_feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    # --- FIX: unwrap "path_to_index" if present ---
    dino_index_map_obj = torch.load(dino_index_map_path, map_location="cpu")
    if isinstance(dino_index_map_obj, dict) and "path_to_index" in dino_index_map_obj:
        dino_index_map_obj = dino_index_map_obj["path_to_index"]

    path2idx, base2idx = _build_dino_path_maps(dino_index_map_obj)

    # --- FIX: normalize CLIP paths so they match DINO’s "/Images/Images/" pattern ---
    paths = [
        p if p in path2idx else p.replace("/Images/", "/Images/Images/")
        for p in paths
    ]


    # 5) Align DINO to CLIP order (full path first, basename fallback)
    dino_Z, keep_idx, hit_path, hit_base, miss = _align_dino_feats_to_paths(
        dino_feats, path2idx, base2idx, paths
    )

    cov_msg = (f"[CLIP-blind/{split_alias}] Alignment coverage — "
               f"hit_by_path={hit_path}, hit_by_basename={hit_base}, missing={miss}, "
               f"used={len(keep_idx)}/{len(paths)}")
    print(cov_msg); logging.info(cov_msg)

    if dino_Z is None or len(keep_idx) < 4:
        logging.warning(f"[CLIP-blind/{split_alias}] Not enough aligned items; skipping stats.")
        return None

    clip_Z_m = clip_Z[keep_idx]

    stats = _pair_stats(clip_Z_m, dino_Z, thresholds)

    # logging
    logging.info(f"[CLIP-blind/{split_alias}] total_pairs={stats['total_pairs']}")
    for k, v in stats["results"].items():
        logging.info(f"[CLIP-blind/{split_alias}] {k}: "
                     f"count={v['count']} ({v['percent']:.2f}%), "
                     f"clip_high_count={v['clip_high_count']}, "
                     f"relative={v['relative_percent']:.2f}%")
        if use_mlflow and mlflow is not None:
            base = f"clip_blind_{split_alias}_{k}"
            mlflow.log_metric(_mlflow_safe(f"{base}_count"),            v["count"])
            mlflow.log_metric(_mlflow_safe(f"{base}_percent"),          v["percent"])
            mlflow.log_metric(_mlflow_safe(f"{base}_clip_high_count"),  v["clip_high_count"])
            mlflow.log_metric(_mlflow_safe(f"{base}_relative_percent"), v["relative_percent"])

    # save artifacts
    blind_dir = os.path.join(checkpoint_path, "clip_blind", split_alias)
    os.makedirs(blind_dir, exist_ok=True)

    json_path = os.path.join(blind_dir, "clip_blind_stats.json")
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)

    import csv
    csv_path = os.path.join(blind_dir, "clip_blind_top_pairs.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["i","j","clip_sim","dino_sim","gap"])
        w.writeheader()
        for row in stats["top_pairs"]:
            w.writerow(row)

    if use_mlflow and mlflow is not None:
        mlflow.log_artifact(json_path, artifact_path=f"clip_blind/{split_alias}")
        mlflow.log_artifact(csv_path,  artifact_path=f"clip_blind/{split_alias}")

    return stats

# ---- recall extraction / scores ---------------------------------------------

_RE_RECALL = re.compile(r'^(?:val/)?(text[_ ]?to[_ ]?image|image[_ ]?to[_ ]?text)[_/ ]?r@(\d+)$', re.I)

def _normalize_recall_value(v: float) -> Optional[float]:
    """Return recall in [0,100]."""
    if v is None:
        return None
    v = float(v)
    return v * 100.0 if 0.0 <= v <= 1.0 else v  # fractions -> percent

def _extract_recalls_0_100(metrics: dict) -> dict:
    """
    From a flat metrics dict, pull ONLY recalls (R@K) for both directions,
    normalize to [0,100], and return a dict with canonical keys:
      text_to_image_R@1,5,10  and image_to_text_R@1,5,10
    """
    out = {}
    for k, v in metrics.items():
        if not isinstance(v, (int, float)):
            continue
        m = _RE_RECALL.match(k.replace("-", "_"))
        if not m:
            continue
        direction = m.group(1).lower().replace(" ", "_")   # 'text_to_image' or 'image_to_text'
        K = m.group(2)
        key = f"{direction}_R@{K}"
        out[key] = _normalize_recall_value(v)
    return out

_WANTED_KEYS = [
    "text_to_image_R@1", "text_to_image_R@5", "text_to_image_R@10",
    "image_to_text_R@1", "image_to_text_R@5", "image_to_text_R@10",
]

def _dataset_retrieval_score(metrics: dict) -> Tuple[float, float]:
    """
    Compute a single dataset score = mean of the available recalls among _WANTED_KEYS.
    Returns (score_avg, tie_breaker) where tie_breaker is mean R@1 across directions.
    If nothing available, returns (nan, -inf).
    """
    rec = _extract_recalls_0_100(metrics)
    vals = [rec[k] for k in _WANTED_KEYS if k in rec and rec[k] is not None]
    if not vals:
        return float("nan"), float("-inf")
    score = sum(vals) / len(vals)

    r1s = [rec.get("text_to_image_R@1"), rec.get("image_to_text_R@1")]
    r1s = [x for x in r1s if x is not None]
    tie = (sum(r1s) / len(r1s)) if r1s else float("-inf")
    return score, tie

def _epoch_retrieval_score(epoch_results: list) -> Tuple[float, float]:
    """
    epoch_results: list of {'val_name': ..., 'metrics': {...}}.
    Returns (epoch_avg_score, epoch_tie_breaker) where the avg is
    across datasets that have recalls; tie is avg of their R@1 ties.
    """
    scores, ties = [], []
    for r in epoch_results:
        s, t = _dataset_retrieval_score(r.get("metrics", {}))
        if not math.isnan(s):
            scores.append(s)
            ties.append(t)
    if not scores:
        return float("nan"), float("-inf")
    return sum(scores)/len(scores), (sum(ties)/len(ties) if ties else float("-inf"))

def _mlflow_sanitize_metric_name(name: str) -> str:
    # Replace '@' which MLflow disallows
    name = name.replace("@", "_at_")
    # Replace anything not allowed with underscore (keep alnum, space, _ - . : /)
    return re.sub(r"[^A-Za-z0-9 _\\-\\.:/]", "_", name)

# ---- re-exports from open_clip_train ----------------------------------------

from open_clip_train.data import get_data
from open_clip_train.distributed import (
    is_master,
    init_distributed_device,
    broadcast_object,
)
from open_clip_train.logger import setup_logging
from open_clip_train.params import parse_args
from open_clip_train.scheduler import cosine_lr, const_lr, const_lr_cooldown
from open_clip_train.train import train_one_epoch, evaluate  # , compute_clip_blind_stats
from open_clip_train.file_utils import (
    pt_load,
    check_exists,
    start_sync_process,
    remote_sync,
)

__all__ = [
    # constants
    "LATEST_CHECKPOINT_NAME",
    # utils
    "_mlflow_safe", "check_text_gradients", "get_input_dtype",
    "random_seed", "natural_key", "_canon_path_local", "get_latest_checkpoint",
    "_batch_to_device", "_safe_to_device", "_encode_clip_images_and_paths",
    "_run_clip_blind_on_split", "_build_dino_path_maps", "_dedup_by_path", "_align_dino_feats_to_paths",
    "_pair_stats", "_normalize_recall_value", "_extract_recalls_0_100",
    "_dataset_retrieval_score", "_epoch_retrieval_score", "_mlflow_sanitize_metric_name",
    # open_clip_train re-exports
    "get_data", "is_master", "init_distributed_device", "broadcast_object",
    "setup_logging", "parse_args", "cosine_lr", "const_lr", "const_lr_cooldown",
    "train_one_epoch", "evaluate", "pt_load", "check_exists", "start_sync_process", "remote_sync",
]
