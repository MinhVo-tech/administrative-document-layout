"""Table 16: per-field CER / WER of the full pipeline on the 154 test PDFs, before and after post-processing.

    python -m ocr.field_metrics                                  # run YOLO + VietOCR on the PDFs (GPU)
    python -m ocr.field_metrics --predictions outputs/ocr/pred_raw.json   # only recompute the metrics

CER / WER = (substitutions + deletions + insertions) / number of ground-truth characters / words,
pooled over all documents of a field. 'After' strips the 'Số' prefix of `code` (prediction and ground
truth) and normalises `issuanceDate` to 'ngày D tháng M năm YYYY'.
"""
import argparse
import json
import os
from typing import Callable

from common.config import load_paths, repo_path, utf8_stdout
from common.constants import CLASS_NAMES
from ocr.postprocess import flatten_labels, identity, normalize_date, remove_so_prefix


def levenshtein_ops(gt, pred):
    """(substitutions, deletions, insertions, len(gt)) for two strings or two word lists."""
    gt, pred = list(gt), list(pred)
    n, m = len(gt), len(pred)
    d = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]   # (cost, S, D, I)
    for i in range(1, n + 1):
        d[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        d[0][j] = (j, 0, 0, j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if gt[i - 1] == pred[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                c, s, de, ins = d[i - 1][j - 1]
                sub = (c + 1, s + 1, de, ins)
                c, s, de, ins = d[i - 1][j]
                dele = (c + 1, s, de + 1, ins)
                c, s, de, ins = d[i][j - 1]
                inse = (c + 1, s, de, ins + 1)
                d[i][j] = min(sub, dele, inse, key=lambda x: x[0])
    return d[n][m][1], d[n][m][2], d[n][m][3], n


def field_cer_wer(gt, pred, code_fn: Callable, date_fn: Callable):
    """{'cer': {field: rate}, 'wer': {field: rate}}."""
    cer_stats, wer_stats = {}, {}
    for doc, gt_fields in gt.items():
        pred_fields = pred.get(doc, {})
        for key, gt_value in gt_fields.items():
            pred_value = pred_fields.get(key, "")
            if key == "issuanceDate":
                pred_value = date_fn(pred_value)
            if key == "code":
                pred_value, gt_value = code_fn(pred_value), code_fn(gt_value)
            g, p = str(gt_value), str(pred_value)
            cer_stats.setdefault(key, []).append(levenshtein_ops(g, p))
            wer_stats.setdefault(key, []).append(levenshtein_ops(g.split(), p.split()))

    def rate(stats):
        out = {}
        for key, rows in stats.items():
            total = sum(r[3] for r in rows)
            out[key] = sum(r[0] + r[1] + r[2] for r in rows) / total if total else 0.0
        return out

    return {"cer": rate(cer_stats), "wer": rate(wer_stats)}


def run_pipeline(paths, gt, device):
    from ocr.pipeline import find_pdfs, load_models, ocr_pdf

    yolo, detector = load_models(paths["models"]["YL02"], device)
    pdfs = find_pdfs(paths["dataset"]["pdf_test"])
    missing = [d for d in gt if d not in pdfs]
    if missing:
        raise FileNotFoundError(f"{len(missing)} test PDF(s) not found, e.g. {missing[:3]}. See dataset/README.md.")
    preds = {}
    for n, doc in enumerate(sorted(gt), 1):
        preds[doc], _, _ = ocr_pdf(pdfs[doc], yolo, detector, device)
        print(f"[{n}/{len(gt)}] {doc}")
    return preds


def main():
    utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", default=None, help="reuse raw predictions {doc: [{field: text}]}")
    ap.add_argument("--device", default=None)
    ap.add_argument("--paths", default="configs/paths.yaml")
    args = ap.parse_args()

    paths = load_paths(args.paths)
    out_dir = paths["outputs"] / "ocr"
    os.makedirs(out_dir, exist_ok=True)
    with open(paths["dataset"]["ocr_gt"], encoding="utf-8") as f:
        gt = flatten_labels(json.load(f))
    print(len(gt), "ground-truth documents")

    if args.predictions:
        with open(repo_path(args.predictions), encoding="utf-8") as f:
            raw = json.load(f)
    else:
        import torch

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        raw = run_pipeline(paths, gt, device)
        with open(out_dir / "pred_raw.json", "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)
    pred = flatten_labels(raw)
    print("Documents without prediction:", [d for d in gt if d not in pred])

    before = field_cer_wer(gt, pred, code_fn=identity, date_fn=identity)
    after = field_cer_wer(gt, pred, code_fn=remove_so_prefix, date_fn=normalize_date)
    fields = [f for f in CLASS_NAMES if f in before["cer"]]
    header = f"{'Field':<20}{'CER(Before)':>12}{'CER(After)':>12}{'WER(Before)':>12}{'WER(After)':>12}"
    lines = ["Table 16: Per-field OCR performance before and after post-processing (%)", header, "-" * len(header)]
    with open(out_dir / "table16.csv", "w", encoding="utf-8") as f:
        f.write("Field,CER_Before,CER_After,WER_Before,WER_After\n")
        for k in fields:
            vals = [before["cer"][k], after["cer"][k], before["wer"][k], after["wer"][k]]
            lines.append(f"{k:<20}" + "".join(f"{100 * v:>12.2f}" for v in vals))
            f.write(k + "," + ",".join(f"{100 * v:.2f}" for v in vals) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {out_dir / 'table16.csv'}")


if __name__ == "__main__":
    main()
