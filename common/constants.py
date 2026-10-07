"""Constants shared by every module (class list, thresholds, file extensions)."""
import numpy as np

# Order = class id in the YOLO labels and in the trained models.
CLASS_NAMES = [
    "code",
    "documentTitle",
    "documentType",
    "issuanceDate",
    "issuingAgency",
    "signatory",
    "signatoryPosition",
]

SPLITS = ["train", "valid", "test"]
IMG_EXTS = (".jpg", ".jpeg", ".png")

# Detection evaluation (same values as in the paper)
IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)  # 0.50, 0.55, ..., 0.95 for mAP50:95
MAP_CONF = 0.001     # low score threshold so the PR curve used for AP/mAP is complete
REPORT_CONF = 0.5    # score threshold used for Precision / Recall / F1 / IoU
IOU_THRESH = 0.5     # matching IoU for Precision / Recall / F1
