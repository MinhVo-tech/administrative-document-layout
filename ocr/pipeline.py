"""PDF -> page images (300 dpi) -> YOLOv12 field detection -> VietOCR -> seven text fields.

Page selection (`ocr_pdf`): pages are rendered and classified one by one, in page order.
  * a page with a `code` box is a Front page, a page with `signatory` / `signatoryPosition` is a Last page,
    any other page is dropped; a page with both is Front and Last;
  * the first Front and the first Last page are kept and the scan stops as soon as both are found;
  * on each kept page, boxes of the same class are read top-to-bottom, left-to-right and joined;
  * pages are read in page order and a field already filled by a lower page is not read again.

Requires poppler (pdf2image), ultralytics, vietocr.
"""
import io
import math
import re
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from common.constants import CLASS_NAMES

# The original run sent all pages of a document to YOLO in one batch: when page sizes differ, Ultralytics
# pads every page to a square, while a page alone is padded to a rectangle and the boxes move slightly.
# A tiny extra image in the batch reproduces the square padding (only the real page's result is used).
MATCH_BATCH_LETTERBOX = True
_DUMMY = np.zeros((32, 32, 3), np.uint8)


def load_models(yolo_path, device):
    from ultralytics import YOLO
    from vietocr.tool.config import Cfg
    from vietocr.tool.predictor import Predictor

    yolo = YOLO(str(yolo_path))
    yolo.to(device)
    cfg = Cfg.load_config_from_name("vgg_transformer")  # pretrained VietOCR weights, downloaded on first use
    cfg["device"] = device
    return yolo, Predictor(cfg)


def _sync(device):
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def group_rows(boxes, row_threshold=15):
    """Group boxes into text rows (top-to-bottom), each row sorted left-to-right."""
    rows = []
    for box in sorted(boxes, key=lambda b: b["y_center"]):
        for row in rows:
            if abs(row[0]["y_center"] - box["y_center"]) < row_threshold:
                row.append(box)
                break
        else:
            rows.append([box])
    return [sorted(row, key=lambda b: b["x_center"]) for row in rows]


def ocr_page(page, boxes, detector, skip=frozenset()) -> Dict[str, str]:
    """OCR the detected boxes of one page -> {field: text}. Fields in `skip` are not read."""
    import cv2
    from PIL import Image

    # Same JPEG round trip and 6-decimal box rounding as the pipeline that produced the paper results.
    buf = io.BytesIO()
    page.save(buf, format="JPEG")
    img = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    pw, ph = page.size

    parsed = []
    for cls_idx, xyxy in boxes:
        if CLASS_NAMES[cls_idx] in skip:
            continue
        x1, y1, x2, y2 = xyxy[:4]
        xc = float(f"{((x1 + x2) / 2) / pw:.6f}") * w
        yc = float(f"{((y1 + y2) / 2) / ph:.6f}") * h
        bw = float(f"{(x2 - x1) / pw:.6f}") * w
        bh = float(f"{(y2 - y1) / ph:.6f}") * h
        parsed.append({"class_name": CLASS_NAMES[cls_idx],
                       "xyxy": [int(xc - bw / 2), int(yc - bh / 2), int(xc + bw / 2), int(yc + bh / 2)],
                       "x_center": xc, "y_center": yc})

    rois, names = [], []
    for row in group_rows(parsed):
        for b in row:
            x1, y1, x2, y2 = b["xyxy"]
            roi = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if roi.size:
                rois.append(Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)))
                names.append(b["class_name"])

    per_class: Dict[str, List[str]] = {}
    for name, text in zip(names, detector.predict_batch(rois) if rois else []):
        if text.strip():
            per_class.setdefault(name, []).append(text.strip())
    return {name: " ".join(chunks) for name, chunks in per_class.items()}


def pages_have_mixed_sizes(pdf_path, n_pages) -> bool:
    """True when the pages of the PDF have different pixel sizes at 300 dpi (read with pdfinfo)."""
    from pdf2image import pdfinfo_from_path

    info = pdfinfo_from_path(str(pdf_path), first_page=1, last_page=n_pages)
    dims = set()
    for key, value in info.items():
        if re.match(r"Page\s+\d+ size", key):
            m = re.match(r"\s*([\d.]+) x ([\d.]+)", value)
            if m:
                dims.add((math.ceil(float(m.group(1)) * 300 / 72), math.ceil(float(m.group(2)) * 300 / 72)))
    if not dims:
        print(f"WARNING: could not read the page sizes of {Path(pdf_path).name}; assuming equal sizes")
    return len(dims) > 1


def _boxes(res):
    return [(int(k), res.boxes.xyxy[i].cpu().numpy()) for i, k in enumerate(res.boxes.cls)]


def ocr_pdf(pdf_path, yolo, detector, device="cuda"):
    """Returns (fields, timings, n_pages); fields = [{field: raw text}, ...]."""
    from pdf2image import convert_from_path, pdfinfo_from_path

    t0 = time.perf_counter()
    n_pages = int(pdfinfo_from_path(str(pdf_path))["Pages"])
    square = MATCH_BATCH_LETTERBOX and pages_have_mixed_sizes(pdf_path, n_pages)
    t_render = t_yolo = 0.0
    front = last = None  # (page number, PIL page, boxes)
    processed = 0
    for pno in range(1, n_pages + 1):
        a = time.perf_counter()
        page = convert_from_path(str(pdf_path), dpi=300, first_page=pno, last_page=pno)[0]
        b = time.perf_counter()
        _sync(device)
        imgs = [np.array(page.convert("RGB"))] + ([_DUMMY] if square else [])
        boxes = _boxes(yolo(imgs, verbose=False)[0])
        _sync(device)
        t_render += b - a
        t_yolo += time.perf_counter() - b
        processed += 1

        names = {CLASS_NAMES[k] for k, _ in boxes}
        has_code = "code" in names
        has_sig = "signatory" in names or "signatoryPosition" in names
        if has_code and has_sig:
            if front is None and last is None:
                front = last = (pno, page, boxes)  # one page holds both: keep only this page
            elif front is None:
                front = (pno, page, boxes)
            elif last is None:
                last = (pno, page, boxes)
        elif has_code and front is None:
            front = (pno, page, boxes)
        elif has_sig and last is None:
            last = (pno, page, boxes)
        if front is not None and last is not None:
            break
    t_cls = time.perf_counter()

    out, filled = [], set()
    for pno, page, boxes in sorted({p[0]: p for p in (front, last) if p is not None}.values(), key=lambda p: p[0]):
        for name, text in ocr_page(page, boxes, detector, skip=filled).items():
            filled.add(name)  # the lower page wins
            out.append({name: text})
    _sync(device)
    t_end = time.perf_counter()

    timings = {"render_s": t_render, "yolo_s": t_yolo, "crop_ocr_s": t_end - t_cls,
               "merge_s": (t_cls - t0) - t_render - t_yolo, "total_s": t_end - t0, "pages_processed": processed}
    return out, timings, n_pages


def find_pdfs(pdf_root):
    """{document id (file stem): path} for <pdf_root>/<Type>/<doc>.pdf."""
    return {p.stem: p for sub in sorted(Path(pdf_root).iterdir()) if sub.is_dir() for p in sorted(sub.glob("*.pdf"))}
