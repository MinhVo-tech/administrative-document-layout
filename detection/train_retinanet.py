"""Train a RetinaNet configuration of Table 11 (Detectron2).

    python -m detection.train_retinanet --model Re03              # final model, seed 42
    python -m detection.train_retinanet --model Re03 --seed 7     # one of the ten runs of Table 8
"""
import argparse

from detection.detectron2_common import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Re03", help="Re01, Re02 or Re03 (configs/retinanet.yaml)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--config", default="configs/retinanet.yaml")
    ap.add_argument("--paths", default="configs/paths.yaml")
    args = ap.parse_args()
    train("retinanet", args.config, args.model, args.seed, args.paths)
