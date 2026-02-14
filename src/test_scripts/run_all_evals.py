
#!/usr/bin/env python3
"""
run_all_evals.py — central orchestrator

EDIT THE USER CONFIG SECTION or pass flags:
  python run_all_evals.py --train_csv ... --val_csv ... --checkpoint ...

It will:
  1) Extract features from CSV(s) with your checkpoint (no training)
  2) Run k-NN Top-1 (no training)
  3) Run geometry metrics (no training)
  4) Train a linear probe on frozen features

Requires these sibling scripts (already provided):
  - extract_features_from_csv.py
  - eval_knn.py
  - eval_geometry.py
  - eval_linear_probe.py
"""
import argparse, os, subprocess, sys, shlex

HERE = os.path.dirname(os.path.abspath(__file__))

def run(cmd: str):
    print(f"\n$ {cmd}")
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        raise SystemExit(f"[ERROR] Command failed ({ret}): {cmd}")

def main():
    default = dict(
        train_csv = "/ABSOLUTE/OR/REL/PATH/TO/train.csv",
        val_csv   = "/ABSOLUTE/OR/REL/PATH/TO/val.csv",
        geom_csv  = "",  # optional; if empty uses val features
        data_root = "",

        path_col  = "path",
        label_col = "label",
        ignore_labels_for_geom = False,

        model_name = "ViT-B-16",
        pretrained = "openai",
        checkpoint = "/ABSOLUTE/OR/REL/PATH/TO/checkpoint.pt",
        image_size = 224,

        batch_size = 256,
        workers    = 8,
        fp16       = True,

        do_extract = True,
        do_knn     = True,
        do_geom    = True,
        do_linear  = True,

        out_dir    = "feats_out",
        train_pt   = "train.pt",
        val_pt     = "val.pt",
        geom_pt    = "geom.pt",

        knn_k      = 20,
        knn_block  = 8192,

        geom_pairs = 20000,
        geom_temp  = 2.0,

        lp_epochs  = 15,
        lp_lr      = 1e-2,
        lp_wd      = 0.0,
        lp_bs      = 2048,

        device     = "cuda",
    )
    ap = argparse.ArgumentParser()
    for k,v in default.items():
        if isinstance(v, bool):
            ap.add_argument(f"--{k}", dest=k, action="store_true")
            ap.add_argument(f"--no-{k}", dest=k, action="store_false")
            ap.set_defaults(**{k: v})
        else:
            ap.add_argument(f"--{k}", type=type(v), default=v)
    cfg = vars(ap.parse_args())

    out_dir = cfg["out_dir"]
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(HERE, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    dump_csv_py = os.path.join(HERE, "extract_features_from_csv.py")
    knn_py      = os.path.join(HERE, "eval_knn.py")
    geom_py     = os.path.join(HERE, "eval_geometry.py")
    linprobe_py = os.path.join(HERE, "eval_linear_probe.py")

    train_pt = os.path.join(out_dir, cfg["train_pt"]) if cfg["train_pt"] else os.path.join(out_dir, "train.pt")
    val_pt   = os.path.join(out_dir, cfg["val_pt"])   if cfg["val_pt"]   else os.path.join(out_dir, "val.pt")
    geom_pt  = os.path.join(out_dir, cfg["geom_pt"])  if cfg["geom_pt"]  else os.path.join(out_dir, "geom.pt")

    # 1) Extract features
    if cfg["do_extract"]:
        # Train (for kNN & linear probe)
        if cfg["do_knn"] or cfg["do_linear"]:
            if os.path.isfile(cfg["train_csv"]):
                cmd = [sys.executable, dump_csv_py,
                    "--csv", shlex.quote(cfg["train_csv"]),
                    "--root", shlex.quote(cfg["data_root"]),
                    "--path-col", shlex.quote(cfg["path_col"]),
                    "--label-col", shlex.quote(cfg["label_col"]),
                    "--out", shlex.quote(train_pt),
                    "--model", shlex.quote(cfg["model_name"]),
                    "--pretrained", shlex.quote(cfg["pretrained"]),
                    "--image-size", str(cfg["image_size"]),
                    "--batch-size", str(cfg["batch_size"]),
                    "--workers", str(cfg["workers"]),
                    "--device", shlex.quote(cfg["device"])]
                if cfg["fp16"]: cmd.append("--fp16")
                if cfg["checkpoint"] and cfg["checkpoint"].lower() != "none":
                    cmd += ["--checkpoint", shlex.quote(cfg["checkpoint"])]
                run(" ".join(cmd))
            else:
                print("[WARN] train_csv not found; skipping train features.")

        # Val (for kNN, linear probe, geometry default)
        need_val = cfg["do_knn"] or cfg["do_linear"] or (cfg["do_geom"] and not cfg["geom_csv"])
        if need_val and os.path.isfile(cfg["val_csv"]):
            cmd = [sys.executable, dump_csv_py,
                "--csv", shlex.quote(cfg["val_csv"]),
                "--root", shlex.quote(cfg["data_root"]),
                "--path-col", shlex.quote(cfg["path_col"]),
                "--label-col", shlex.quote(cfg["label_col"]),
                "--out", shlex.quote(val_pt),
                "--model", shlex.quote(cfg["model_name"]),
                "--pretrained", shlex.quote(cfg["pretrained"]),
                "--image-size", str(cfg["image_size"]),
                "--batch-size", str(cfg["batch_size"]),
                "--workers", str(cfg["workers"]),
                "--device", shlex.quote(cfg["device"])]
            if cfg["fp16"]: cmd.append("--fp16")
            if cfg["checkpoint"] and cfg["checkpoint"].lower() != "none"]:
                cmd += ["--checkpoint", shlex.quote(cfg["checkpoint"])]
            run(" ".join(cmd))
        elif need_val:
            print("[WARN] val_csv not found; skipping val features.")

        # Geometry-only CSV
        if cfg["do_geom"] and cfg["geom_csv"]:
            if os.path.isfile(cfg["geom_csv"]):
                cmd = [sys.executable, dump_csv_py,
                    "--csv", shlex.quote(cfg["geom_csv"]),
                    "--root", shlex.quote(cfg["data_root"]),
                    "--path-col", shlex.quote(cfg["path_col"]),
                    "--out", shlex.quote(geom_pt),
                    "--model", shlex.quote(cfg["model_name"]),
                    "--pretrained", shlex.quote(cfg["pretrained"]),
                    "--image-size", str(cfg["image_size"]),
                    "--batch-size", str(cfg["batch_size"]),
                    "--workers", str(cfg["workers"]),
                    "--device", shlex.quote(cfg["device"]),
                    "--ignore-labels"]
                if cfg["fp16"]: cmd.append("--fp16")
                if cfg["checkpoint"] and cfg["checkpoint"].lower() != "none"]:
                    cmd += ["--checkpoint", shlex.quote(cfg["checkpoint"])]
                run(" ".join(cmd))
            else:
                print("[WARN] geom_csv not found; will use val features for geometry.")

    # 2) k-NN
    if cfg["do_knn"] and os.path.isfile(train_pt) and os.path.isfile(val_pt):
        cmd = [sys.executable, os.path.join(HERE, "eval_knn.py"),
            "--train-pt", shlex.quote(train_pt),
            "--val-pt", shlex.quote(val_pt),
            "--k", str(cfg["knn_k"]),
            "--block", str(cfg["knn_block"]),
            "--device", shlex.quote(cfg["device"])]
        run(" ".join(cmd))
    elif cfg["do_knn"]:
        print("[SKIP] k-NN: missing features.")

    # 3) Geometry
    if cfg["do_geom"]:
        geom_in = None
        if cfg["geom_csv"] and os.path.isfile(geom_pt):
            geom_in = geom_pt
        elif os.path.isfile(val_pt):
            geom_in = val_pt
        if geom_in:
            cmd = [sys.executable, os.path.join(HERE, "eval_geometry.py"),
                "--feats-pt", shlex.quote(geom_in),
                "--pairs", str(cfg["geom_pairs"]),
                "--temp", str(cfg["geom_temp"])]
            run(" ".join(cmd))
        else:
            print("[SKIP] Geometry: no features available.")

    # 4) Linear probe
    if cfg["do_linear"] and os.path.isfile(train_pt) and os.path.isfile(val_pt):
        cmd = [sys.executable, os.path.join(HERE, "eval_linear_probe.py"),
            "--train-pt", shlex.quote(train_pt),
            "--val-pt", shlex.quote(val_pt),
            "--epochs", str(cfg["lp_epochs"]),
            "--lr", str(cfg["lp_lr"]),
            "--wd", str(cfg["lp_wd"]),
            "--bs", str(cfg["lp_bs"]),
            "--device", shlex.quote(cfg["device"])]
        run(" ".join(cmd))
    elif cfg["do_linear"]:
        print("[SKIP] Linear probe: missing features.")

    print("\n[Done] Outputs in:", out_dir)

if __name__ == "__main__":
    main()
