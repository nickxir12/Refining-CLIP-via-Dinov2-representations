
import argparse, torch, tqdm
import torch.nn.functional as F

def load_feats(path):
    obj = torch.load(path, map_location="cpu")
    return obj["feats"].float(), obj["labels"].long()

@torch.no_grad()
def knn_top1(train_pt, val_pt, k=20, block=8192, device="cuda"):
    Xtr, Ytr = load_feats(train_pt)
    Xva, Yva = load_feats(val_pt)
    Xtr = F.normalize(Xtr, dim=-1); Xva = F.normalize(Xva, dim=-1)
    Xtr = Xtr.to(device); Ytr = Ytr.to(device); Xva = Xva.to(device); Yva = Yva.to(device)

    correct = 0; total = 0
    for i in tqdm.trange(0, Xva.size(0), block, desc="kNN", ncols=100):
        q = Xva[i:i+block]                     # [b,d]
        S = q @ Xtr.t()                        # [b,ntr]
        topk = torch.topk(S, k=k, dim=1).indices   # [b,k]
        preds = torch.mode(Ytr[topk], dim=1).values
        correct += int((preds == Yva[i:i+block]).sum().item())
        total += preds.numel()
    acc = correct/total
    print(f"[kNN] k={k} top1={acc:.4f}")
    return acc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-pt", required=True)
    ap.add_argument("--val-pt", required=True)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--block", type=int, default=8192)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    knn_top1(args.train_pt, args.val_pt, args.k, args.block, args.device)

if __name__ == "__main__":
    main()
