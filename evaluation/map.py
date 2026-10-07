"""AP50 / AP50:95 per class and mAP (Tables 8-11, 14).

Use predictions extracted at MAP_CONF (0.001) so the precision-recall curve is complete.
AP = all-point interpolated area under the PR curve; mAP = mean of the per-class APs.
"""
import numpy as np

from common.constants import IOU_THRESHOLDS
from evaluation.metrics import compute_iou


def compute_ap(recalls, precisions):
    """All-point interpolated AP."""
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    idx = np.where(recalls[1:] != recalls[:-1])[0]
    return np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1])


def evaluate_map_per_class(pred_dict, gt_dict, class_names, iou_thresholds=IOU_THRESHOLDS):
    """({class: AP50}, {class: AP50:95}). Classes without ground truth get NaN (ignored in the mean)."""
    ap50, ap5095 = {}, {}
    for cls_id, cls_name in enumerate(class_names):
        gt_per_img = {img: [g for g in gts if g["label"] == cls_id] for img, gts in gt_dict.items()}
        num_gt = sum(len(v) for v in gt_per_img.values())
        if num_gt == 0:
            ap50[cls_name] = ap5095[cls_name] = float("nan")
            continue

        preds = [(img, p["box"], p["score"])
                 for img in gt_dict for p in pred_dict.get(img, []) if p["label"] == cls_id]
        preds.sort(key=lambda x: -x[2])

        aps = []
        for thr in iou_thresholds:
            used = {img: [False] * len(v) for img, v in gt_per_img.items()}
            tp = np.zeros(len(preds))
            fp = np.zeros(len(preds))
            for k, (img, box, _) in enumerate(preds):
                best_iou, best_i = 0, -1
                for i, gt in enumerate(gt_per_img[img]):
                    if used[img][i]:
                        continue
                    iou = compute_iou(box, gt["box"])
                    if iou > best_iou:
                        best_iou, best_i = iou, i
                if best_iou >= thr and best_i >= 0:
                    tp[k] = 1
                    used[img][best_i] = True
                else:
                    fp[k] = 1
            tpc, fpc = np.cumsum(tp), np.cumsum(fp)
            aps.append(compute_ap(tpc / num_gt, tpc / (tpc + fpc + 1e-9)) if len(preds) else 0.0)
        ap50[cls_name] = float(aps[0])  # iou_thresholds[0] = 0.50
        ap5095[cls_name] = float(np.mean(aps))
    return ap50, ap5095


def evaluate_map(pred_dict, gt_dict, class_names, iou_thresholds=IOU_THRESHOLDS):
    """(mAP50:95, mAP50) = mean of the per-class APs."""
    ap50, ap5095 = evaluate_map_per_class(pred_dict, gt_dict, class_names, iou_thresholds)
    return float(np.nanmean(list(ap5095.values()))), float(np.nanmean(list(ap50.values())))
