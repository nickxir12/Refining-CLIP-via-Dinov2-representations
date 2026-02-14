#!/usr/bin/env python
import os, argparse, csv, json
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import open_clip
import numpy as np


def _read_unique_image_list_from_csv(csv_path: str) -> List[str]:
    pths: List[str] = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ip = row["image"].strip()
            if ip:
                pths.append(ip)
    # unique order-preserving
    seen, uniq = set(), []
    for p in pths:
        if p not in seen:
            uniq.append(p); seen.add(p)
    return uniq


class FlickrImages(Dataset):
    def __init__(self, image_paths: List[str]):
        self.paths = image_paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx: int):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        return p, img


def _normalize_paths(paths: List[str]) -> List[str]:
    return [str(Path(p).resolve()) for p in paths]


def _load_dino_features_any(path: str):
    """
    Supported:
      1) Torch: .pt/.pth with keys:
         - {'paths': [...], 'feats': tensor/ndarray}
         - or {'paths': [...], 'features': ...}
      2) Numpy: .npy features PLUS a sibling .txt with one path per line
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"DINO feature file not found: {p}")

    if p.suffix in [".pt", ".pth"]:
        obj = torch.load(p, map_location="cpu")
        if isinstance(obj, dict) and "paths" in obj and ("feats" in obj or "features" in obj):
            paths = obj["paths"]
            feats = obj.get("feats", obj.get("features"))
            if isinstance(feats, np.ndarray):
                feats = torch.from_numpy(feats)
            elif not torch.is_tensor(feats):
                feats = torch.tensor(feats)
            return _normalize_paths(paths), feats.float()
        raise ValueError("Unsupported .pt/.pth dict format for DINO features. Expected keys 'paths' and 'feats'/'features'.")
    elif p.suffix == ".npy":
        feats_np = np.load(p)
        feats = torch.from_numpy(feats_np).float()
        txt = p.with_suffix(".txt")
        if not txt.exists():
            raise FileNotFoundError(f"Companion paths list not found: {txt}")
        with open(txt, "r", encoding="utf-8") as f:
            paths = [ln.strip() for ln in f if ln.strip()]
        if len(paths) != feats.shape[0]:
            raise ValueError(f"Path count ({len(paths)}) != feats rows ({feats.shape[0]}).")
        return _normalize_paths(paths), feats
    else:
        raise ValueError(f"Unsupported DINO feature file extension: {p.suffix}")


def _build_alignment(index_paths: List[str], dino_paths: List[str]) -> torch.Tensor:
    dino_idx: Dict[str, int] = {p: i for i, p in enumerate(dino_paths)}
    idxs = []
    missing = []
    for p in index_paths:
        if p in dino_idx:
            idxs.append(dino_idx[p])
        else:
            base = Path(p).name
            candidates = [i for i, dp in enumerate(dino_paths) if Path(dp).name == base]
            if len(candidates) == 1:
                idxs.append(candidates[0])
            else:
                missing.append(p)
                idxs.append(-1)
    if missing:
        miss_log = {"missing_count": len(missing), "missing_examples": missing[:10]}
        print("[WARN] Some images from val CSV not found in DINO features:", json.dumps(miss_log, ensure_ascii=False))
    return torch.tensor(idxs, dtype=torch.long)


def _load_index_map(map_path: str) -> Union[Dict[str, int], List[int], torch.Tensor]:
    """
    Accept common forms:
      - dict[path->int]
      - {"index_map": list_or_tensor}
      - plain list/1D tensor saved directly
    """
    obj = torch.load(map_path, map_location="cpu")
    # direct dict[path]->idx
    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()) and all(isinstance(v, (int, np.integer)) for v in obj.values()):
        return {str(Path(k).resolve()): int(v) for k, v in obj.items()}
    # packed dict containing array under a key
    if isinstance(obj, dict) and "index_map" in obj:
        arr = obj["index_map"]
        if isinstance(arr, np.ndarray):
            arr = torch.from_numpy(arr)
        return arr
    # plain list/tensor
    if isinstance(obj, (list, np.ndarray, torch.Tensor)):
        if isinstance(obj, list):
            return obj
        if isinstance(obj, np.ndarray):
            return torch.from_numpy(obj)
        return obj
    raise ValueError("Unsupported index map format. Expect dict[path->idx], a list/1D tensor, or a dict with key 'index_map'.")


def _apply_index_map(csv_paths: List[str], index_map: Union[Dict[str,int], List[int], torch.Tensor]) -> torch.Tensor:
    """
    If dict[path->idx]: build per-csv index list (with -1 for missing).
    If list/1D tensor: assume it's already aligned to csv_paths order (same length).
    """
    if isinstance(index_map, dict):
        norm = {str(Path(k).resolve()): v for k, v in index_map.items()}
        idxs = [norm.get(str(Path(p).resolve()), -1) for p in csv_paths]
        return torch.tensor(idxs, dtype=torch.long)
    else:
        if isinstance(index_map, list):
            idx = torch.tensor(index_map, dtype=torch.long)
        elif isinstance(index_map, np.ndarray):
            idx = torch.from_numpy(index_map).long()
        else:
            idx = index_map.long()
        if idx.ndim != 1 or len(idx) != len(csv_paths):
            raise ValueError(f"Index map length mismatch: got {len(idx)} vs CSV images {len(csv_paths)}.")
        return idx


@torch.no_grad()
def compute_clip_image_embeddings(model, preprocess, paths: List[str], batch_size: int, num_workers: int, device: torch.device):
    ds = FlickrImages(paths)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=lambda batch: ([b[0] for b in batch], torch.stack([preprocess(b[1]) for b in batch], dim=0))
    )
    feats, out_paths = [], []
    for ps, x in dl:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            f = model.encode_image(x)
        f = F.normalize(f.float(), dim=-1)
        feats.append(f.cpu())
        out_paths.extend(ps)
    return out_paths, torch.cat(feats, dim=0)


def main():
    ap = argparse.ArgumentParser("CLIP-blind pairs with cached DINO features")
    ap.add_argument("--val_csv", required=True, help="Flickr30k val CSV (columns: image, caption)")
    ap.add_argument("--dino_feats", required=True, help=".pt/.pth with {'paths','feats'|'features'} OR .npy + .txt")
    ap.add_argument("--dino_index_map", default="", help="Optional: torch file for precomputed index map (dict[path->idx] or list aligned to CSV).")

    ap.add_argument("--clip_model", default="RN50-quickgelu")
    ap.add_argument("--clip_pretrained", default="", help="Path/tag for your fine-tuned checkpoint (open_clip).")
    ap.add_argument("--clip_cache_dir", default=os.environ.get("OPENCLIP_CACHE", ""))

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--clip_hi", type=float, default=0.8)
    ap.add_argument("--dino_lo", type=float, default=0.3)
    ap.add_argument("--output_dir", default="results_blind_pairs")
    ap.add_argument("--save_pairs_csv", action="store_true")

    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "embeddings").mkdir(parents=True, exist_ok=True)

    # 1) Load target image list from CSV
    csv_paths_raw = _read_unique_image_list_from_csv(args.val_csv)
    csv_paths = _normalize_paths(csv_paths_raw)
    print(f"[INFO] Val images (unique): {len(csv_paths)}")

    # 2) Load cached DINO features
    dino_paths, dino_feats = _load_dino_features_any(args.dino_feats)
    print(f"[INFO] DINO cache: {len(dino_paths)} embeddings, dim={dino_feats.shape[-1]}")

    # 3) Determine alignment to CSV images
    if args.dino_index_map:
        print(f"[INFO] Using precomputed DINO index map: {args.dino_index_map}")
        raw_map = _load_index_map(args.dino_index_map)
        idx_map = _apply_index_map(csv_paths, raw_map)
    else:
        print("[INFO] Building alignment by path/basename…")
        idx_map = _build_alignment(csv_paths, dino_paths)

    valid_mask = idx_map >= 0
    if valid_mask.sum().item() < len(csv_paths):
        print(f"[WARN] Using only {valid_mask.sum().item()} / {len(csv_paths)} images present in both CSV and DINO cache.")
    csv_paths_valid = [csv_paths[i] for i, ok in enumerate(valid_mask.tolist()) if ok]
    dino_feats_aligned = dino_feats[idx_map[valid_mask]]
    dino_feats_aligned = F.normalize(dino_feats_aligned.float(), dim=-1)

    # 4) Build CLIP and encode those same valid images
    clip_model, clip_preprocess, _ = open_clip.create_model_and_transforms(
        args.clip_model,
        pretrained=(args.clip_pretrained if args.clip_pretrained else None),
        cache_dir=(args.clip_cache_dir or None),
    )
    clip_model = clip_model.to(device).eval()
    clip_paths, clip_feats = compute_clip_image_embeddings(
        clip_model, clip_preprocess, csv_paths_valid, args.batch_size, args.num_workers, device
    )
    assert clip_paths == csv_paths_valid, "CLIP dataloader path order mismatch."

    # 5) Similarities & blind pairs (image–image, unordered)
    N = len(clip_paths)
    print(f"[INFO] Computing similarities over N={N} images...")
    sims_clip = clip_feats @ clip_feats.T
    sims_dino = dino_feats_aligned @ dino_feats_aligned.T

    triu_mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
    clip_vals = sims_clip[triu_mask]
    dino_vals = sims_dino[triu_mask]
    blind_mask = (clip_vals > args.clip_hi) & (dino_vals < args.dino_lo)

    total_pairs = int(N * (N - 1) // 2)
    blind_count = int(blind_mask.sum().item())
    pct = 100.0 * blind_count / max(1, total_pairs)

    # 6) Save
    torch.save(
        {"paths": clip_paths, "clip": clip_feats, "dino": dino_feats_aligned},
        out_dir / "embeddings" / "val_embeddings.pt"
    )
    with open(out_dir / "summary.txt", "w") as f:
        f.write(f"Images (aligned): {N}\n")
        f.write(f"Total pairs: {total_pairs}\n")
        f.write(f"Condition: CLIP>{args.clip_hi} & DINO<{args.dino_lo}\n")
        f.write(f"Blind pairs: {blind_count}\n")
        f.write(f"Percentage: {pct:.4f}%\n")

    if args.save_pairs_csv and blind_count > 0:
        csv_path = out_dir / "clip_blind_pairs.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["idx_i","idx_j","image_i","image_j","clip_sim","dino_sim"])
            idxs = torch.nonzero(triu_mask, as_tuple=False)
            blind_idxs = idxs[blind_mask]
            cv = clip_vals[blind_mask]
            dv = dino_vals[blind_mask]
            for (i, j), c, d in zip(blind_idxs.tolist(), cv.tolist(), dv.tolist()):
                w.writerow([i, j, clip_paths[i], clip_paths[j], f"{c:.6f}", f"{d:.6f}"])

    print("====================================================")
    print(f"Images (aligned): {N}")
    print(f"Total pairs: {total_pairs}")
    print(f"CLIP>{args.clip_hi} & DINO<{args.dino_lo} -> blind pairs: {blind_count}")
    print(f"Percentage: {pct:.4f}%")
    print(f"Summary: {out_dir / 'summary.txt'}")
    if args.save_pairs_csv and blind_count > 0:
        print(f"Pairs CSV: {out_dir / 'clip_blind_pairs.csv'}")
    print("Done.")


if __name__ == "__main__":
    main()
