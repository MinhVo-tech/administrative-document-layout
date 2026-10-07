# Administrative Document Layout: Field Detection and Extraction for Vietnamese Administrative Documents

Code, annotations and models for the paper **TODO: paper title** (TODO: venue, year).

The pipeline locates seven information fields on Vietnamese administrative documents and reads them:

```
PDF ──► page images ──► field detection (YOLOv12) ──► text recognition (VietOCR) ──► 7 fields
```

Fields: `code` (document number), `documentTitle`, `documentType`, `issuanceDate`, `issuingAgency`,
`signatory`, `signatoryPosition`. Document types: Decision, Resolution, Directive, Dispatch, Telegram, Notice, Plan.

| | |
|---|---|
| Dataset | 1,513 documents, 2,720 annotated page images, 14,458 boxes — see [dataset/README.md](dataset/README.md) |
| Detectors | YOLOv12-S, Faster R-CNN and RetinaNet (R50-FPN) |
| Paper | TODO: link / DOI |
| Data and models (Zenodo) | TODO: DOI |

## Repository structure

```
administrative-document-layout/
├── configs/            paths and training / ensemble settings
├── common/             class list, thresholds, config loading
├── dataset/            annotations, splits, dataset tools and documentation (CC BY-NC 4.0)
│   ├── labels/             YOLO labels  <split>/<image>.txt
│   ├── annotations/        Pascal VOC labels (voc/<split>/<image>.xml) and OCR ground truth (ocr_gt.json)
│   ├── splits/             <split>.txt: image file and source document
│   ├── tools/              consistency checks, PDF types, release packaging
│   ├── images/             page images            (download)
│   └── pdf-test/           154 test PDFs          (download)
├── detection/          training (YOLOv12, Faster R-CNN, RetinaNet), inference, evaluation (Tables 9-11)
├── evaluation/         Precision / Recall / F1 / IoU, mAP, statistical tests (Table 8)
├── ensemble/           Soft-NMS and WBF ensembles (Table 14)
├── ocr/                PDF → YOLOv12 → VietOCR pipeline, CER/WER (Table 16)
├── results/            per-run metrics of the ten-seed experiment (Table 8)
└── models/             released checkpoints   (download)
```

## Installation

```bash
git clone https://github.com/TODO/administrative-document-layout.git
cd administrative-document-layout
pip install -r requirements.txt
pip install --no-deps vietocr==0.3.13                                    # pins Pillow 10.2 otherwise
pip install 'git+https://github.com/facebookresearch/detectron2.git'     # Faster R-CNN / RetinaNet
sudo apt-get install poppler-utils                                       # PDF rendering (pdf2image)
```

All commands below are run from the repository root (`python -m <module>`).

## Data and models

The annotations are in this repository. Page images, test PDFs and model weights are on Zenodo
(TODO: DOI). After downloading:

```bash
unzip images.zip   -d dataset/       # -> dataset/images/{train,valid,test}/*.jpg
unzip pdf-test.zip -d dataset/       # -> dataset/pdf-test/<Type>/*.pdf
mv YL02.pt F02.pth Re03.pth models/
sha256sum -c SHA256SUMS.txt
```

Paths can be changed in [configs/paths.yaml](configs/paths.yaml).

## Training

```bash
python -m detection.train_yolov12 --model YL02             # YL01-YL04, configs/yolov12.yaml
python -m detection.train_faster_rcnn --model F02          # F01-F03,   configs/faster_rcnn.yaml
python -m detection.train_retinanet --model Re03           # Re01-Re03, configs/retinanet.yaml
python -m detection.train_yolov12 --model YL02 --seed 7    # one run of the ten-seed experiment (Table 8)
```

Seeds of Table 8: 7, 42, 66, 123, 777, 1024, 2027, 3407, 4096, 9999. Trained weights are written to
`outputs/train_*`; evaluate them with `python -m detection.evaluate --models <checkpoint>`.

Inference on GPU is not bit-exact (non-deterministic CUDA kernels in cuDNN and Detectron2), so
re-running the evaluation, especially on a different GPU or library version, may change the metrics
by up to ~0.2 percentage points.

## License

* **Code and model weights:** [GNU AGPL-3.0](LICENSE). The detectors are trained with
  [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) and
  [Detectron2](https://github.com/facebookresearch/detectron2) (Apache-2.0); OCR uses
  [VietOCR](https://github.com/pbcquoc/vietocr) (Apache-2.0).
* **Dataset** (annotations, page images, PDFs): [CC BY-NC 4.0](dataset/LICENSE), **for non-commercial
  research only**. The documents contain names, signatures and seals of officials; see
  [dataset/README.md](dataset/README.md#intended-use) before using them.

## Citation

```bibtex
@article{TODO_key,
  title   = {TODO: Paper title},
  author  = {TODO: Author list},
  journal = {TODO: Journal / Conference},
  year    = {TODO},
  doi     = {TODO}
}
```
