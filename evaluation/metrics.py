"""Precision / Recall / F1 / mean IoU at a fixed IoU threshold (Tables 8-11, 14).

Use predictions filtered at REPORT_CONF (0.5). A prediction is a true positive when it matches an
unmatched ground-truth box of the same class with IoU >= iou_thresh (greedy, in prediction order).
"""
import numpy as np


def compute_iou(box1, box2):
    xA, yA = max(box1[0], box2[0]), max(box1[1], box2[1])
    xB, yB = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)


def _prf(tp, fp, fn, ious):
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f1, (np.mean(ious) if ious else 0.0)


def evaluate_metrics(pred_dict, gt_dict, class_names, iou_thresh=0.5):
    """Micro-averaged over all boxes. Returns (precision, recall, f1, mean IoU of the true positives)."""
    tp = fp = fn = 0
    ious = []
    for img_file, gt_boxes in gt_dict.items():
        matched = [False] * len(gt_boxes)
        for pred in pred_dict.get(img_file, []):
            found = False
            for i, gt in enumerate(gt_boxes):
                if matched[i] or pred["label"] != gt["label"]:
                    continue
                iou = compute_iou(pred["box"], gt["box"])
                if iou >= iou_thresh:
                    tp += 1
                    ious.append(iou)
                    matched[i] = found = True
                    break
            if not found:
                fp += 1
        fn += matched.count(False)
    return _prf(tp, fp, fn, ious)


def evaluate_metrics_per_class(pred_dict, gt_dict, class_names, iou_thresh=0.5):
    """{class_name: {TP, FP, FN, precision, recall, f1, iou}} (predictions sorted by score per image)."""
    out = {}
    for cls_id, cls_name in enumerate(class_names):
        tp = fp = fn = 0
        ious = []
        for img_file in gt_dict:
            gt_boxes = [g for g in gt_dict[img_file] if g["label"] == cls_id]
            preds = sorted([p for p in pred_dict.get(img_file, []) if p["label"] == cls_id],
                           key=lambda p: -p["score"])
            matched = [False] * len(gt_boxes)
            for pred in preds:
                found = False
                for i, gt in enumerate(gt_boxes):
                    if matched[i]:
                        continue
                    iou = compute_iou(pred["box"], gt["box"])
                    if iou >= iou_thresh:
                        tp += 1
                        ious.append(iou)
                        matched[i] = found = True
                        break
                if not found:
                    fp += 1
            fn += matched.count(False)
        p, r, f1, iou = _prf(tp, fp, fn, ious)
        out[cls_name] = {"TP": tp, "FP": fp, "FN": fn, "precision": p, "recall": r, "f1": f1, "iou": iou}
    return out
