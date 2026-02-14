
import argparse, torch, random
import torch.nn.functional as F
import numpy as np

def load_feats(path):
    obj = torch.load(path, map_location="cpu")
    return obj["feats"].float()

@torch.no_grad()
def uniformity(feats: torch.Tensor, n_pairs:int=20000, temperature:float=2.0):
    N = feats.size(0)
    idx_i = torch.randint(0, N, (n_pairs,))
    idx_j = torch.randint(0, N, (n_pairs,))
    z_i = feats[idx_i]; z_j = feats[idx_j]
    val = torch.exp(-temperature * ((z_i - z_j)**2).sum(dim=1)).mean().item()
    return val

@torch.no_grad()
def anisotropy(feats: torch.Tensor, q:int=256):
    X = feats - feats.mean(0, keepdim=True)
    # torch.pca_lowrank returns U, S, V; explained variance from S^2 / sum(S^2)
    q = min(q, min(X.size(0), X.size(1)) - 1)
    U, S, V = torch.pca_lowrank(X, q=q, center=False)
    expl = (S**2) / (S**2).sum()
    pc1 = float(expl[0])
    pc10 = float(expl[:10].sum()) if expl.numel() >= 10 else float(expl.sum())
    pc100 = float(expl[:100].sum()) if expl.numel() >= 100 else float(expl.sum())
    return pc1, pc10, pc100

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats-pt", required=True, help="Path to .pt from dump_features.py")
    ap.add_argument("--pairs", type=int, default=20000)
    ap.add_argument("--temp", type=float, default=2.0)
    args = ap.parse_args()
    feats = load_feats(args.feats_pt)
    feats = F.normalize(feats, dim=-1)
    u = uniformity(feats, n_pairs=args.pairs, temperature=args.temp)
    pc1, pc10, pc100 = anisotropy(feats)
    print(f"[uniformity] {u:.6f} (lower better)")
    print(f"[anisotropy] PC1={pc1*100:.2f}%  PC1-10={pc10*100:.2f}%  PC1-100={pc100*100:.2f}% (flatter better)")

if __name__ == "__main__":
    main()
