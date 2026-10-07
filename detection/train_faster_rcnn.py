"""Train a Faster R-CNN configuration of Table 9 (Detectron2).

    python -m detection.train_faster_rcnn --model F02             # final model, seed 42
    python -m detection.train_faster_rcnn --model F02 --seed 7    # one of the ten runs of Table 8
"""
import argparse

from detection.detectron2_common import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="F02", help="F01, F02 or F03 (configs/faster_rcnn.yaml)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--config", default="configs/faster_rcnn.yaml")
    ap.add_argument("--paths", default="configs/paths.yaml")
    args = ap.parse_args()
    train("faster_rcnn", args.config, args.model, args.seed, args.paths)
