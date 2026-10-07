"""Table 14: inference-time ensembles (Soft-NMS and WBF) of YL02, F02 and Re03.

    python -m ensemble.fuse
    python -m ensemble.fuse --use-cache      # reuse the member predictions of a previous run

For every row of configs/ensemble.yaml:
  * Precision / Recall / F1: member boxes with score >= 0.5 (rounded like the original .txt files),
    fused with the row parameters; Soft-NMS keeps fused boxes with score >= report_thresh.
  * mAP50 / mAP50:95: member boxes with score >= 0.001 (complete PR curve), same fusion parameters.
All members and the ground truth (VOC labels) use the same test images.
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd

from common.config import load_paths, load_yaml, repo_path, split_dirs
from common.constants import CLASS_NAMES, IOU_THRESH, IOU_THRESHOLDS, MAP_CONF, REPORT_CONF
from detection.predict import predict
from evaluation.io import filter_by_conf, list_images, read_voc_gt
from evaluation.map import evaluate_map
from evaluation.metrics import evaluate_metrics


def quantize(pred_dict):
    """Round like the prediction .txt files of the original pipeline (box .2f, score .4f)."""
    return {img: [{"box": [float(f"{v:.2f}") for v in p["box"]], "score": float(f"{p['score']:.4f}"),
                   "label": p["label"]} for p in preds]
            for img, preds in pred_dict.items()}


# ---------------- Soft-NMS ----------------
def _iou_one_to_many(a, B):
    xA, yA = np.maximum(a[0], B[:, 0]), np.maximum(a[1], B[:, 1])
    xB, yB = np.minimum(a[2], B[:, 2]), np.minimum(a[3], B[:, 3])
    inter = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter
    return np.where(union == 0, 0.0, inter / np.where(union == 0, 1.0, union))


def soft_nms(boxes, sigma, iou_thresh, score_thresh):
    """Gaussian Soft-NMS within each class. boxes: (N, 6) = x1, y1, x2, y2, score, label."""
    boxes = boxes.copy()
    scores = boxes[:, 4].copy()
    keep = []
    for i in range(boxes.shape[0]):
        m = i + np.argmax(scores[i:])
        if i != m:
            boxes[[i, m]] = boxes[[m, i]]
            scores[[i, m]] = scores[[m, i]]
        if i + 1 < boxes.shape[0]:
            iou = _iou_one_to_many(boxes[i, :4], boxes[i + 1:, :4])
            decay = (boxes[i + 1:, 5] == boxes[i, 5]) & (iou > iou_thresh)
            idx = np.where(decay)[0] + i + 1
            scores[idx] *= np.exp(-(iou[decay] ** 2) / sigma)
        if scores[i] >= score_thresh:
            keep.append(np.concatenate([boxes[i][:4], [scores[i]], [boxes[i][5]]]))
    return np.stack(keep, axis=0) if keep else np.empty((0, 6))


def softnms_fusion(pred_dicts, sigma, iou_thresh, score_thresh):
    fused = {}
    for img in set().union(*(d.keys() for d in pred_dicts)):
        rows = [p["box"] + [p["score"], p["label"]] for d in pred_dicts for p in d.get(img, [])]
        if rows:
            out = soft_nms(np.array(rows, dtype=float), sigma, iou_thresh, score_thresh)
            fused[img] = [{"box": list(r[:4]), "score": float(r[4]), "label": int(r[5])} for r in out]
    return fused


# ---------------- Weighted Boxes Fusion ----------------
def read_img_sizes(images_dir):
    import cv2

    sizes = {}
    for f in list_images(images_dir):
        im = cv2.imread(os.path.join(images_dir, f))
        if im is not None:
            sizes[f] = (im.shape[1], im.shape[0])
    return sizes


def wbf_fusion(pred_dicts, img_sizes, iou_thr, skip_box_thr, conf_type="avg", weights=None):
    from ensemble_boxes import weighted_boxes_fusion

    fused = {}
    for img, (w, h) in img_sizes.items():
        boxes, scores, labels = [], [], []
        for d in pred_dicts:
            ps = d.get(img, [])
            boxes.append([[p["box"][0] / w, p["box"][1] / h, p["box"][2] / w, p["box"][3] / h] for p in ps])
            scores.append([p["score"] for p in ps])
            labels.append([p["label"] for p in ps])
        b, s, l = weighted_boxes_fusion(boxes, scores, labels, weights=weights, iou_thr=iou_thr,
                                        skip_box_thr=skip_box_thr, conf_type=conf_type)
        fused[img] = [{"box": [x[0] * w, x[1] * h, x[2] * w, x[3] * h], "score": float(sc), "label": int(lb)}
                      for x, sc, lb in zip(b, s, l)]
    return fused


# ---------------- one row of Table 14 ----------------
def run_row(row, preds_low, gt, img_sizes):
    members = row["members"]

    inp = [quantize(filter_by_conf(preds_low[m], REPORT_CONF)) for m in members]
    if row["kind"] == "softnms":
        fused_rep = quantize(softnms_fusion(inp, row["sigma"], row["iou"], row["report_thresh"]))
    else:
        fused_rep = wbf_fusion(inp, img_sizes, row["iou_thr"], row["skip_box_thr"], weights=row["weights"])
    precision, recall, f1, _ = evaluate_metrics(fused_rep, gt, CLASS_NAMES, IOU_THRESH)

    inp_all = [preds_low[m] for m in members]
    if row["kind"] == "softnms":
        fused_all = softnms_fusion(inp_all, row["sigma"], row["iou"], MAP_CONF)
    else:
        fused_all = wbf_fusion(inp_all, img_sizes, row["iou_thr"], MAP_CONF, weights=row["weights"])
    map5095, map50 = evaluate_map(fused_all, gt, CLASS_NAMES, IOU_THRESHOLDS)

    return {"Model": row["model"], "Method": row["method"], "mAP50:95": 100 * map5095, "mAP50": 100 * map50,
            "Precision": 100 * precision, "Recall": 100 * recall, "F1-score": 100 * f1}


def member_predictions(cfg, paths, images_dir, cache_file, use_cache):
    if use_cache and os.path.exists(cache_file):
        print(f"[INFO] Loading cached predictions: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    members = sorted({m for row in cfg["rows"] for m in row["members"]})
    preds = {}
    for m in members:
        print(f"[INFO] Predicting with {m} ...")
        preds[m] = predict(str(paths["models"][m]), images_dir, conf_thres=MAP_CONF,
                           img_size=cfg["inference_img_size"][m])
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(preds, f)
    return preds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/ensemble.yaml")
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--use-cache", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(repo_path(args.config))
    paths = load_paths(args.paths)
    images_dir, _, voc_dir = split_dirs(paths, "test")
    out_dir = paths["outputs"] / "table14"

    preds_low = member_predictions(cfg, paths, images_dir, str(out_dir / "member_preds_conf0.001.pkl"),
                                   args.use_cache)
    gt = read_voc_gt(voc_dir, CLASS_NAMES, images_dir)
    img_sizes = read_img_sizes(images_dir)
    common = set(gt) & set(img_sizes)
    for m in preds_low:
        common &= set(preds_low[m])
    print(f"[INFO] images: GT={len(gt)}, sizes={len(img_sizes)}, common to all models={len(common)}")
    assert len(common) == len(gt), "image file names differ between the models and the ground truth"

    rows = []
    for row in cfg["rows"]:
        print(f"[RUN] {row['model']} {row['method']} ...")
        rows.append(run_row(row, preds_low, gt, img_sizes))
    table = pd.DataFrame(rows).round(2)
    table.to_csv(out_dir / "table14_ensemble.csv", index=False)
    print("\nInference-time ensembles\n" + table.to_string(index=False))
    print(f"\nSaved {out_dir / 'table14_ensemble.csv'}")


if __name__ == "__main__":
    main()
