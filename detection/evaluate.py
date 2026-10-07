"""Evaluate trained detectors on the test split (Tables 9, 10, 11 and the per-field results).

YOLO models are scored against the YOLO labels, Detectron2 models against the VOC labels
(same images, boxes identical up to VOC's integer rounding).

    python -m detection.evaluate                          # the three released models (configs/paths.yaml)
    python -m detection.evaluate --models models/*.pt models/*.pth
"""
import argparse
import glob
import os

import pandas as pd

from common.config import load_paths, load_yaml, repo_path, split_dirs
from common.constants import CLASS_NAMES, IOU_THRESH, IOU_THRESHOLDS, MAP_CONF, REPORT_CONF
from detection.predict import framework_of, predict
from evaluation.io import filter_by_conf, read_voc_gt, read_yolo_gt, save_predictions
from evaluation.map import evaluate_map, evaluate_map_per_class
from evaluation.metrics import evaluate_metrics, evaluate_metrics_per_class

FRAMEWORK_NAME = {"yolo": "YOLOv12", "faster_rcnn": "Faster R-CNN", "retinanet": "RetinaNet"}
DEFAULT_IMG_SIZE = {"yolo": 640, "faster_rcnn": 800, "retinanet": 800}


def inference_sizes():
    """Model name -> inference image size, from the training configs (YL02 -> 800, F01 -> 640, ...)."""
    sizes = {}
    for cfg_file, key in [("configs/yolov12.yaml", "imgsz"), ("configs/faster_rcnn.yaml", "img_size"),
                          ("configs/retinanet.yaml", "img_size")]:
        for name, m in load_yaml(repo_path(cfg_file))["models"].items():
            sizes[name] = m[key]
    return sizes


def model_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def evaluate_model(model_path, paths, split="test", img_size=None, pred_dir=None):
    """One model -> rows (overall + one per class) with precision, recall, f1, iou, map50, map5095."""
    images_dir, labels_dir, voc_dir = split_dirs(paths, split)
    fw = framework_of(model_path)
    name = model_name(model_path)
    img_size = img_size or inference_sizes().get(name.split("_seed")[0], DEFAULT_IMG_SIZE[fw])
    print(f"\n>>> [{FRAMEWORK_NAME[fw]}] {name} (img_size={img_size})")

    pred_all = predict(model_path, images_dir, conf_thres=MAP_CONF, img_size=img_size)
    if pred_dir:
        save_predictions(pred_all, os.path.join(pred_dir, f"{name}_{split}_conf{MAP_CONF}.txt"))
    pred_rep = filter_by_conf(pred_all, REPORT_CONF)
    gt = read_yolo_gt(labels_dir, images_dir) if fw == "yolo" else read_voc_gt(voc_dir, CLASS_NAMES, images_dir)

    p, r, f1, iou = evaluate_metrics(pred_rep, gt, CLASS_NAMES, IOU_THRESH)
    map5095, map50 = evaluate_map(pred_all, gt, CLASS_NAMES, IOU_THRESHOLDS)
    per_cls = evaluate_metrics_per_class(pred_rep, gt, CLASS_NAMES, IOU_THRESH)
    ap50, ap5095 = evaluate_map_per_class(pred_all, gt, CLASS_NAMES, IOU_THRESHOLDS)

    base = {"framework": FRAMEWORK_NAME[fw], "model": name, "img_size": img_size}
    rows = [{**base, "field": "__overall__", "precision": p, "recall": r, "f1": f1, "iou": iou,
             "map50": map50, "map5095": map5095}]
    for c in CLASS_NAMES:
        s = per_cls[c]
        rows.append({**base, "field": c, "precision": s["precision"], "recall": s["recall"], "f1": s["f1"],
                     "iou": s["iou"], "map50": ap50[c], "map5095": ap5095[c]})
    print_rows(rows)
    return rows


def print_rows(rows):
    print(f"{'Field':<20}{'Precision':>10}{'Recall':>10}{'F1':>10}{'IoU':>10}{'mAP50':>10}{'mAP50:95':>10}")
    for r in rows:
        print(f"{r['field']:<20}{r['precision']:>10.4f}{r['recall']:>10.4f}{r['f1']:>10.4f}{r['iou']:>10.4f}"
              f"{r['map50']:>10.4f}{r['map5095']:>10.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", help="checkpoint files or glob patterns (default: models in paths.yaml)")
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--save-predictions", action="store_true", help="also write the raw predictions (conf 0.001)")
    args = ap.parse_args()

    paths = load_paths(args.paths)
    if args.models:
        models = sorted({m for pat in args.models for m in glob.glob(pat)})
    else:
        models = [str(p) for p in paths["models"].values()]
    missing = [m for m in models if not os.path.exists(m)]
    if missing:
        raise FileNotFoundError(f"Model file(s) not found: {missing}. See README.md, section 'Models'.")

    out_dir = paths["outputs"] / "detection"
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for m in models:
        rows += evaluate_model(m, paths, args.split, pred_dir=out_dir / "predictions" if args.save_predictions else None)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"metrics_{args.split}_per_field.csv", index=False)
    overall = df[df.field == "__overall__"].drop(columns="field")
    cols = ["map5095", "map50", "precision", "recall", "f1", "iou"]
    overall[cols] = (overall[cols] * 100).round(2)
    overall.to_csv(out_dir / f"metrics_{args.split}.csv", index=False)
    print("\n" + overall.rename(columns={"map5095": "mAP50:95", "map50": "mAP50", "precision": "P", "recall": "R",
                                         "f1": "F1", "iou": "IoU"}).to_string(index=False))
    print(f"\nSaved {out_dir / f'metrics_{args.split}.csv'} and the per-field table.")


if __name__ == "__main__":
    main()
