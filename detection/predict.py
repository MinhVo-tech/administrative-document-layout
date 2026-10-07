"""Run a trained detector on a folder of page images.

Returns {image_file: [{"box": [x1, y1, x2, y2], "score": float, "label": int}, ...]} for YOLOv12
(Ultralytics) and for Faster R-CNN / RetinaNet (Detectron2), so every model is evaluated by the
same code in evaluation/.
"""
import gc
import os

from common.constants import CLASS_NAMES, MAP_CONF
from evaluation.io import list_images

DETECTRON2_BASE = {
    "faster_rcnn": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
    "retinanet": "COCO-Detection/retinanet_R_50_FPN_3x.yaml",
}


def framework_of(model_path):
    """'yolo' for Ultralytics .pt files; Detectron2 .pth files: 'retinanet' if the name starts with 'Re'."""
    name = os.path.basename(model_path)
    if name.endswith(".pt"):
        return "yolo"
    return "retinanet" if name.lower().startswith("re") else "faster_rcnn"


def predict_yolo(model_path, images_dir, conf_thres=MAP_CONF, img_size=None, batch_size=15, device=None):
    """img_size=None uses the image size stored in the checkpoint."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    kwargs = {} if img_size is None else {"imgsz": img_size}
    if device is not None:
        kwargs["device"] = device
    image_files = list_images(images_dir)
    preds = {}
    for start in range(0, len(image_files), batch_size):
        batch = image_files[start:start + batch_size]
        results = model([os.path.join(images_dir, f) for f in batch], conf=conf_thres, verbose=False, **kwargs)
        for img_file, res in zip(batch, results):
            preds[img_file] = [
                {"box": b.xyxy[0].tolist(), "score": b.conf[0].item(), "label": int(b.cls[0].item())}
                for b in res.boxes
            ]
        del results
    return preds


def build_detectron2_predictor(model_path, framework, conf_thres=MAP_CONF, img_size=800, device="cuda",
                               num_classes=len(CLASS_NAMES)):
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(DETECTRON2_BASE[framework]))
    cfg.MODEL.WEIGHTS = str(model_path)
    cfg.INPUT.MIN_SIZE_TEST = img_size
    cfg.INPUT.MAX_SIZE_TEST = img_size
    cfg.MODEL.DEVICE = device
    # Detectron2's default score threshold (0.05) would silently drop low-score boxes and truncate the
    # PR curve used for AP/mAP; the default 100 detections per image is raised for the same reason.
    if framework == "retinanet":
        cfg.MODEL.RETINANET.NUM_CLASSES = num_classes
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = conf_thres
    else:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_thres
    cfg.TEST.DETECTIONS_PER_IMAGE = 300
    return DefaultPredictor(cfg)


def predict_detectron2(model_path, images_dir, framework, conf_thres=MAP_CONF, img_size=800, device="cuda",
                       batch_size=10):
    """Images are processed one by one (a single DefaultPredictor is not thread-safe)."""
    import cv2
    import torch

    predictor = build_detectron2_predictor(model_path, framework, conf_thres, img_size, device)
    image_files = list_images(images_dir)
    preds = {}
    for start in range(0, len(image_files), batch_size):
        for img_file in image_files[start:start + batch_size]:
            img = cv2.imread(os.path.join(images_dir, img_file))
            if img is None:
                preds[img_file] = []
                continue
            inst = predictor(img)["instances"].to("cpu")
            preds[img_file] = [
                {"box": box.tolist(), "score": float(score), "label": int(label)}
                for box, score, label in zip(inst.pred_boxes.tensor.numpy(), inst.scores.numpy(),
                                             inst.pred_classes.numpy())
                if score >= conf_thres
            ]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return preds


def predict(model_path, images_dir, conf_thres=MAP_CONF, img_size=None, device=None):
    """Dispatch on the checkpoint type. Detectron2 models default to 800 px and CUDA."""
    fw = framework_of(model_path)
    if fw == "yolo":
        return predict_yolo(model_path, images_dir, conf_thres, img_size, device=device)
    return predict_detectron2(model_path, images_dir, fw, conf_thres, img_size or 800, device or "cuda")
