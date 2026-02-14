
import argparse, os, csv, torch, tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

def build_image_encoder(model_name: str, pretrained: str, device: str, checkpoint: str=None, image_size:int=224):
    try:
        import open_clip
    except ImportError:
        raise RuntimeError("open_clip is required for the default loader. Install it or edit build_image_encoder().")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, force_quick_gelu=True)
    model.visual.eval().to(device)
    if checkpoint:
        sd = torch.load(checkpoint, map_location="cpu")
        if "state_dict" in sd: sd = sd["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    if image_size != 224:
        preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711)),
        ])
    return model, preprocess

class CSVDataset(Dataset):
    def __init__(self, csv_path, root="", path_col="path", label_col="label", preprocess=None, ignore_labels=False):
        self.root = root
        self.preprocess = preprocess
        self.ignore_labels = ignore_labels
        rows = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        if path_col not in rows[0]:
            raise ValueError(f"CSV missing column '{path_col}'. Columns: {list(rows[0].keys())}")
        self.paths = [os.path.join(root, r[path_col]) if root and not os.path.isabs(r[path_col]) else r[path_col] for r in rows]
        if ignore_labels:
            self.labels = [-1] * len(self.paths)
            self.classes = None
        else:
            if label_col not in rows[0]:
                raise ValueError(f"CSV missing column '{label_col}'. Columns: {list(rows[0].keys())}")
            raw_labels = [r[label_col] for r in rows]
            # Map labels (string/int) to contiguous ints
            uniq = sorted(set(raw_labels))
            self.class_to_idx = {c:i for i,c in enumerate(uniq)}
            self.classes = uniq
            self.labels = [self.class_to_idx[l] for l in raw_labels]

    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]
        img = Image.open(p).convert("RGB")
        x = self.preprocess(img) if self.preprocess else transforms.ToTensor()(img)
        y = self.labels[i]
        return x, y

@torch.no_grad()
def extract(csv_path, root, batch_size, workers, device, fp16, model_name, pretrained, checkpoint, image_size, path_col, label_col, ignore_labels, out_path):
    model, preprocess = build_image_encoder(model_name, pretrained, device, checkpoint, image_size)
    ds = CSVDataset(csv_path, root=root, path_col=path_col, label_col=label_col, preprocess=preprocess, ignore_labels=ignore_labels)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=workers, pin_memory=True)
    feats, labels = [], []
    pbar = tqdm.tqdm(dl, desc=f"Extract CSV({os.path.basename(csv_path)})", ncols=100)
    total = 0
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        if fp16:
            with torch.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16):
                f = model.encode_image(x)
        else:
            f = model.encode_image(x)
        f = F.normalize(f.float(), dim=-1)
        feats.append(f.cpu()); labels.append(y.clone())
        total += x.size(0)
        pbar.set_postfix(n=total)
    feats = torch.cat(feats)
    labels = torch.cat(labels)
    payload = {"feats": feats, "labels": labels}
    if getattr(ds, "classes", None) is not None:
        payload["classes"] = ds.classes
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(payload, out_path)
    print(f"[save] {out_path} -> feats={tuple(feats.shape)} labels={labels.unique().numel()} classes")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with at least a 'path' column (and 'label' if labeled)")
    ap.add_argument("--root", default="", help="Root to prepend to relative paths in CSV")
    ap.add_argument("--path-col", default="path")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--ignore-labels", action="store_true", help="Treat dataset as unlabeled (labels=-1)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--model", default="ViT-B-16")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--checkpoint", default=None, help="Path to your CLIP weights (optional)")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--out", required=True, help="Output .pt path")
    args = ap.parse_args()

    extract(args.csv, args.root, args.batch_size, args.workers, args.device, args.fp16,
            args.model, args.pretrained, args.checkpoint, args.image_size,
            args.path_col, args.label_col, args.ignore_labels, args.out)

if __name__ == "__main__":
    main()
