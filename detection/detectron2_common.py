"""Training code shared by Faster R-CNN (Table 9) and RetinaNet (Table 11), Detectron2 R50-FPN 3x."""
import math
import os
import random
import xml.etree.ElementTree as ET

import numpy as np

from common.config import load_paths, load_yaml, repo_path
from common.constants import CLASS_NAMES, IMG_EXTS


def set_seed(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_document_dicts(xml_dir, images_dir):
    """Detectron2 records from VOC .xml files; images are looked up by stem in images_dir."""
    import cv2
    from detectron2.structures import BoxMode

    stem_to_img = {os.path.splitext(f)[0]: f for f in os.listdir(images_dir) if f.lower().endswith(IMG_EXTS)}
    records = []
    for idx, xml_file in enumerate(sorted(f for f in os.listdir(xml_dir) if f.lower().endswith(".xml"))):
        root = ET.parse(os.path.join(xml_dir, xml_file)).getroot()
        stem = os.path.splitext(xml_file)[0]
        if stem not in stem_to_img:
            print(f"Warning: image for {xml_file} not found in {images_dir}")
            continue
        img_path = os.path.join(images_dir, stem_to_img[stem])
        size = root.find("size")
        try:
            width, height = int(size.find("width").text), int(size.find("height").text)
        except (AttributeError, TypeError, ValueError):
            height, width = cv2.imread(img_path).shape[:2]

        annotations = []
        for obj in root.findall("object"):
            name = obj.find("name")
            bbox = obj.find("bndbox")
            if name is None or bbox is None or name.text not in CLASS_NAMES:
                continue
            xmin, ymin, xmax, ymax = (float(bbox.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax"))
            if xmax <= xmin or ymax <= ymin:
                print(f"Warning: invalid box in {xml_file}: {[xmin, ymin, xmax, ymax]}")
                continue
            annotations.append({"bbox": [xmin, ymin, xmax, ymax], "bbox_mode": BoxMode.XYXY_ABS,
                                "category_id": CLASS_NAMES.index(name.text), "iscrowd": 0})
        if annotations:
            records.append({"file_name": img_path, "image_id": idx, "height": height, "width": width,
                            "annotations": annotations})
    print(f"Loaded {len(records)} images from {xml_dir}")
    return records


def register_splits(paths):
    from detectron2.data import DatasetCatalog, MetadataCatalog

    for split in ["train", "valid", "test"]:
        name = f"document_{split}"
        if name not in DatasetCatalog.list():
            DatasetCatalog.register(name, lambda s=split: get_document_dicts(
                str(paths["dataset"]["voc"] / s), str(paths["dataset"]["images"] / s)))
            MetadataCatalog.get(name).set(thing_classes=CLASS_NAMES)


def setup_cfg(framework, common, m, output_dir, num_train_images):
    from detectron2 import model_zoo
    from detectron2.config import get_cfg

    max_iter = math.ceil(num_train_images / common["batch_size"]) * m["epochs"]
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(common["base_config"]))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(common["base_config"])  # COCO-pretrained
    cfg.DATASETS.TRAIN = ("document_train",)
    cfg.DATASETS.TEST = ("document_valid",)
    cfg.DATALOADER.NUM_WORKERS = common["num_workers"]
    cfg.DATALOADER.PIN_MEMORY = True
    cfg.SOLVER.IMS_PER_BATCH = common["batch_size"]
    cfg.SOLVER.BASE_LR = common["base_lr"]
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = tuple(int(f * max_iter) for f in common["lr_steps"])
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.AMP.ENABLED = common["amp"]
    if framework == "retinanet":
        cfg.MODEL.RETINANET.NUM_CLASSES = len(CLASS_NAMES)
    else:
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = common["roi_batch_size_per_image"]
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES)
    cfg.INPUT.MIN_SIZE_TRAIN = (m["img_size"],)
    cfg.INPUT.MAX_SIZE_TRAIN = m["img_size"]
    cfg.INPUT.MIN_SIZE_TEST = m["img_size"]
    cfg.INPUT.MAX_SIZE_TEST = m["img_size"]
    os.makedirs(output_dir, exist_ok=True)
    cfg.OUTPUT_DIR = str(output_dir)
    print(f"max_iter = {max_iter}")
    return cfg


def train(framework, config_file, model_key, seed=None, paths_file="configs/paths.yaml"):
    """Train one configuration (e.g. framework='faster_rcnn', model_key='F02') and evaluate it on test (COCO AP)."""
    from detectron2.data import build_detection_test_loader
    from detectron2.engine import DefaultTrainer
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.utils.logger import setup_logger

    setup_logger()
    cfg_all = load_yaml(repo_path(config_file))
    common, m = cfg_all["common"], cfg_all["models"][model_key]
    seed = common["seed"] if seed is None else seed
    paths = load_paths(paths_file)
    register_splits(paths)

    num_train = len([f for f in os.listdir(paths["dataset"]["voc"] / "train") if f.endswith(".xml")])
    out_dir = paths["outputs"] / f"train_{framework}" / f"{model_key}_seed{seed}"

    # Seeds python / numpy / torch as in the paper runs; Detectron2's own cfg.SEED keeps its default.
    set_seed(seed)
    cfg = setup_cfg(framework, common, m, out_dir, num_train)

    class Trainer(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            return COCOEvaluator(dataset_name, cfg, False, output_folder or os.path.join(cfg.OUTPUT_DIR, "inference"))

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    evaluator = COCOEvaluator("document_test", cfg, False, output_dir=os.path.join(cfg.OUTPUT_DIR, "test_eval"))
    metrics = inference_on_dataset(trainer.model, build_detection_test_loader(cfg, "document_test"), evaluator)
    print(f"Test results: {metrics}")
    print(f"Final weights: {os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')}")
