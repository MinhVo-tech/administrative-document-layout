# Results

`all_runs_metrics.csv` — test-set metrics of the ten-seed experiment (Table 8): one row per run,
3 detectors (`YL02` = YOLOv12, `F02` = Faster R-CNN, `Re03` = RetinaNet) × 10 seeds
(7, 42, 66, 123, 777, 1024, 2027, 3407, 4096, 9999).

Columns: `precision`, `recall`, `f1`, `iou` (score ≥ 0.5, IoU ≥ 0.5) and `map50`, `map5095` (score ≥ 0.001),
as fractions. Produced by `evaluation/statistical_tests.py --models-dir ...`; Table 8 is recomputed from this
file with `python -m evaluation.statistical_tests`.
