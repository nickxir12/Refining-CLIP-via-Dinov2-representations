
import argparse, torch, torch.nn as nn, torch.optim as optim, tqdm, math
import torch.nn.functional as F

def load_feats(path):
    obj = torch.load(path, map_location="cpu")
    return obj["feats"].float(), obj["labels"].long(), obj.get("classes", None)

class LinearHead(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes, bias=True)
    def forward(self, x): return self.fc(x)

def run(train_pt, val_pt, epochs=15, lr=1e-2, wd=0.0, bs=2048, device="cuda"):
    Xtr, Ytr, _ = load_feats(train_pt)
    Xva, Yva, classes = load_feats(val_pt)
    in_dim = Xtr.size(1)
    n_classes = int(Ytr.max().item()+1)
    print(f"[data] train={tuple(Xtr.shape)} val={tuple(Xva.shape)} classes={n_classes}")

    # Create loaders over tensors
    tr = torch.utils.data.TensorDataset(Xtr, Ytr); va = torch.utils.data.TensorDataset(Xva, Yva)
    tr_loader = torch.utils.data.DataLoader(tr, batch_size=bs, shuffle=True, drop_last=False)
    va_loader = torch.utils.data.DataLoader(va, batch_size=bs, shuffle=False)

    model = LinearHead(in_dim, n_classes).to(device)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

    best_acc = 0.0; best_epoch = -1
    for ep in range(1, epochs+1):
        model.train()
        loss_sum = 0.0; n = 0
        pbar = tqdm.tqdm(tr_loader, desc=f"LP epoch {ep}/{epochs}", ncols=100)
        for xb, yb in pbar:
            xb = xb.to(device); yb = yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += float(loss.item())*xb.size(0); n += xb.size(0)
            pbar.set_postfix(loss=f"{loss_sum/n:.4f}")
        # eval
        model.eval(); correct=0; total=0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(device); yb = yb.to(device)
                pred = model(xb).argmax(dim=1)
                correct += int((pred==yb).sum().item()); total += yb.numel()
        acc = correct/total
        if acc>best_acc: best_acc, best_epoch = acc, ep
        print(f"[eval] acc@top1={acc:.4f} (best {best_acc:.4f} @epoch {best_epoch})")
    print(f"[done] Linear probe best top1={best_acc:.4f} @epoch {best_epoch}")
    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-pt", required=True)
    ap.add_argument("--val-pt", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--bs", type=int, default=2048)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    run(args.train_pt, args.val_pt, args.epochs, args.lr, args.wd, args.bs, args.device)

if __name__ == "__main__":
    main()
