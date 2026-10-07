# Models

Place the released checkpoints here (download from Zenodo, see the main README):

| File | Detector | Configuration | Used for |
|---|---|---|---|
| `YL02.pt` | YOLOv12-S (Ultralytics 8.3.170) | 800 px, batch 8, 150 epochs | Tables 8, 10, 14, 16 |
| `F02.pth` | Faster R-CNN R50-FPN (Detectron2) | 800 px, 150 epochs | Tables 8, 9, 14 |
| `Re03.pth` | RetinaNet R50-FPN (Detectron2) | 800 px, 200 epochs | Tables 8, 11, 14 |

Verify the downloads with `sha256sum -c SHA256SUMS.txt` (checksums published with the Zenodo record).

The other configurations of Tables 9-11 (YL01, YL03, YL04, F01, F03, Re01, Re02) and the 30 seed runs of
Table 8 are not released; they can be retrained with the scripts in `detection/` and the settings in
`configs/`. The per-run metrics of Table 8 are in `results/all_runs_metrics.csv`.

License: AGPL-3.0 (see `LICENSE` in the repository root).
