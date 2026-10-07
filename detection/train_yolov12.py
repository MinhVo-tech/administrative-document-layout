"""Train a YOLOv12 configuration of Table 10 (Ultralytics 8.3.170).

    python -m detection.train_yolov12 --model YL02               # final model, seed 123
    python -m detection.train_yolov12 --model YL02 --seed 7      # one of the ten runs of Table 8

Every hyper-parameter not set in configs/yolov12.yaml is the Ultralytics default.
"""
import argparse
import random

import numpy as np

from common.config import load_paths, load_yaml, repo_path
from common.constants import CLASS_NAMES


def set_seed(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_data_yaml(paths, out_file):
    """Ultralytics data.yaml with absolute paths (labels are found by replacing /images/ with /labels/)."""
    import yaml

    images = paths["dataset"]["images"]
    data = {"path": str(images.parent), "train": str(images / "train"), "val": str(images / "valid"),
            "test": str(images / "test"), "nc": len(CLASS_NAMES), "names": CLASS_NAMES}
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return out_file


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="YL02", help="YL01, YL02, YL03 or YL04 (configs/yolov12.yaml)")
    ap.add_argument("--seed", type=int, default=None, help="default: value in configs/yolov12.yaml")
    ap.add_argument("--device", default=None, help="e.g. 0 or cpu (default: GPU if available)")
    ap.add_argument("--config", default="configs/yolov12.yaml")
    ap.add_argument("--paths", default="configs/paths.yaml")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    cfg = load_yaml(repo_path(args.config))
    common, m = cfg["common"], cfg["models"][args.model]
    seed = common["seed"] if args.seed is None else args.seed
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    paths = load_paths(args.paths)
    data_yaml = write_data_yaml(paths, paths["outputs"] / "data.yaml")
    run_name = f"{args.model}_seed{seed}"

    print(f"===== {run_name}: imgsz={m['imgsz']} batch={m['batch']} epochs={common['epochs']} =====")
    set_seed(seed)
    model = YOLO(common["weights"])
    model.train(data=str(data_yaml), epochs=common["epochs"], imgsz=m["imgsz"], batch=m["batch"], device=device,
                seed=seed, project=str(paths["outputs"] / "train_yolo"), name=run_name)
    print(model.val(data=str(data_yaml), split="test"))
    print(f"Best weights: {paths['outputs'] / 'train_yolo' / run_name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
