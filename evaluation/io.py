"""Ground-truth / prediction readers and writers.

Box format everywhere: {"box": [x1, y1, x2, y2] (pixels), "label": class_id, "score": float (predictions only)}.
Dictionaries are keyed by the image FILE NAME (e.g. "Chi-thi_012-0-2_front_0_png_jpg.rf.<hash>.jpg").
"""
import os
import xml.etree.ElementTree as ET

from common.constants import IMG_EXTS


def list_images(images_dir):
    return [f for f in os.listdir(images_dir) if f.lower().endswith(IMG_EXTS)]


def yolo_label_to_xyxy(label, img_w, img_h):
    cls, x, y, w, h = label
    return [(x - w / 2) * img_w, (y - h / 2) * img_h, (x + w / 2) * img_w, (y + h / 2) * img_h]


def read_yolo_gt(labels_dir, images_dir):
    """YOLO .txt labels -> pixel boxes (image size read from the image). Used to evaluate YOLO models."""
    import cv2

    gts = {}
    for img_file in list_images(images_dir):
        label_file = os.path.join(labels_dir, os.path.splitext(img_file)[0] + ".txt")
        if not os.path.exists(label_file):
            continue
        img_h, img_w = cv2.imread(os.path.join(images_dir, img_file)).shape[:2]
        boxes = []
        with open(label_file, "r") as f:
            for line in f:
                vals = line.strip().split()
                if len(vals) != 5:
                    continue
                cls, xc, yc, w, h = map(float, vals)
                boxes.append({"box": yolo_label_to_xyxy([cls, xc, yc, w, h], img_w, img_h), "label": int(cls)})
        gts[img_file] = boxes
    return gts


def read_voc_gt(xml_dir, class_names, images_dir):
    """Pascal VOC .xml labels -> pixel boxes. Used to evaluate Detectron2 models and the ensembles.
    Keyed by the real image file name (matched by stem), so it lines up with the prediction keys."""
    stem_to_img = {os.path.splitext(f)[0]: f for f in list_images(images_dir)}
    gts = {}
    for file in os.listdir(xml_dir):
        if not file.endswith(".xml"):
            continue
        stem = os.path.splitext(file)[0]
        root = ET.parse(os.path.join(xml_dir, file)).getroot()
        if stem in stem_to_img:
            img_file = stem_to_img[stem]
        else:
            tag = root.find("filename")
            img_file = tag.text if tag is not None else stem + ".jpg"
            print(f"[WARN] No image with stem '{stem}' in {images_dir}; using '{img_file}'")
        boxes = []
        for obj in root.findall("object"):
            name = obj.find("name")
            name = name.text if name is not None else ""
            bbox = obj.find("bndbox")
            if name not in class_names or bbox is None:
                continue
            box = [float(bbox.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax")]
            boxes.append({"box": box, "label": class_names.index(name)})
        gts[img_file] = boxes
    return gts


def filter_by_conf(pred_dict, conf_thres):
    return {k: [p for p in v if p["score"] >= conf_thres] for k, v in pred_dict.items()}


def save_predictions(pred_dict, out_path):
    """One line per box: image,x1,y1,x2,y2,score,label (full precision, so read_predictions is lossless)."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        for img_file, preds in pred_dict.items():
            for p in preds:
                f.write(",".join(map(str, [img_file, *p["box"], p["score"], p["label"]])) + "\n")


def read_predictions(pred_file, image_files=None):
    """Inverse of save_predictions. Images in `image_files` with no detection get an empty list."""
    pred_dict = {f: [] for f in (image_files or [])}
    with open(pred_file, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            pred_dict.setdefault(parts[0], []).append(
                {"box": list(map(float, parts[1:5])), "score": float(parts[5]), "label": int(parts[6])})
    return pred_dict
