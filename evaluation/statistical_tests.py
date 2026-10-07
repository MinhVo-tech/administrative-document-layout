"""Table 8: mean +/- std over ten seeds and Wilcoxon signed-rank tests with Holm correction.

    # from the released per-run metrics (no GPU, no model needed)
    python -m evaluation.statistical_tests

    # re-create results/all_runs_metrics.csv from the 30 seed checkpoints (not released), then the tables
    python -m evaluation.statistical_tests --models-dir path/to/seed_models
        expects <dir>/YL02_seed<s>.pt, <dir>/F02_seed<s>.pth, <dir>/Re03_seed<s>.pth for every seed

Seeds: 7, 42, 66, 123, 777, 1024, 2027, 3407, 4096, 9999. Std is the sample standard deviation (ddof=1).
Each metric is tested for the three pairs (YL02-F02, YL02-Re03, F02-Re03), two-sided, paired by seed;
the three p-values of a metric are Holm-corrected.
"""
import argparse
import os

import numpy as np
import pandas as pd

from common.config import load_paths, repo_path, utf8_stdout

SEEDS = [7, 42, 66, 123, 777, 1024, 2027, 3407, 4096, 9999]
MODELS = {"YL02": "YOLOv12", "F02": "Faster R-CNN", "Re03": "RetinaNet"}
METRICS = ["map5095", "map50", "precision", "recall", "f1", "iou"]
METRIC_NAMES = {"map5095": "mAP50:95", "map50": "mAP50", "precision": "Precision", "recall": "Recall",
                "f1": "F1-score", "iou": "IoU"}
PAIRS = [("YL02", "F02"), ("YL02", "Re03"), ("F02", "Re03")]


def holm_bonferroni(pvals, alpha=0.05):
    """Holm step-down adjusted p-values and significance flags."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj_sorted = np.array(pvals, dtype=float)[order] * (m - np.arange(m))
    adj_sorted = np.minimum(np.maximum.accumulate(adj_sorted), 1.0)
    adj = np.empty(m)
    adj[order] = adj_sorted
    return adj.tolist(), (adj < alpha).tolist()


def mean_std_table(runs):
    """Table 8 in percent: '85.40 ± 0.24' per metric (rows) and model (columns)."""
    table = {}
    for key, name in MODELS.items():
        d = runs[runs.model == key]
        table[name] = [f"{100 * d[m].mean():.2f} ± {100 * d[m].std(ddof=1):.2f}" for m in METRICS]
    return pd.DataFrame(table, index=[METRIC_NAMES[m] for m in METRICS])


def wilcoxon_table(runs):
    from scipy.stats import wilcoxon

    rows = []
    for metric in METRICS:
        tmp = []
        for a, b in PAIRS:
            va = runs[runs.model == a].set_index("seed")[metric]
            vb = runs[runs.model == b].set_index("seed")[metric]
            seeds = sorted(set(va.index) & set(vb.index))
            stat, p = wilcoxon(va.loc[seeds].values, vb.loc[seeds].values, alternative="two-sided",
                               zero_method="wilcox", correction=False, mode="auto")
            tmp.append({"metric": METRIC_NAMES[metric], "pair": f"{a} vs {b}", "n": len(seeds),
                        "statistic": float(stat), "p_raw": float(p)})
        p_holm, sig = holm_bonferroni([t["p_raw"] for t in tmp])
        for t, ph, s in zip(tmp, p_holm, sig):
            rows.append({**t, "p_holm": ph, "significant": s})
    return pd.DataFrame(rows)


def run_seed_models(models_dir, paths):
    from detection.evaluate import evaluate_model

    rows = []
    for seed in SEEDS:
        for key, ext in [("YL02", "pt"), ("F02", "pth"), ("Re03", "pth")]:
            path = os.path.join(models_dir, f"{key}_seed{seed}.{ext}")
            overall = evaluate_model(path, paths, "test")[0]
            rows.append({"model": key, "seed": seed, **{m: overall[m] for m in
                                                       ["precision", "recall", "f1", "iou", "map50", "map5095"]}})
    return pd.DataFrame(rows)


def main():
    utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="results/all_runs_metrics.csv", help="per-run metrics (model, seed, ...)")
    ap.add_argument("--models-dir", default=None, help="evaluate the 30 seed checkpoints first")
    ap.add_argument("--paths", default="configs/paths.yaml")
    args = ap.parse_args()

    paths = load_paths(args.paths)
    out_dir = paths["outputs"] / "table8"
    os.makedirs(out_dir, exist_ok=True)
    if args.models_dir:
        runs = run_seed_models(args.models_dir, paths)
        runs.to_csv(out_dir / "all_runs_metrics.csv", index=False)
        print(f"Saved {out_dir / 'all_runs_metrics.csv'}")
    else:
        runs = pd.read_csv(repo_path(args.runs))

    counts = runs.groupby("model").seed.nunique().to_dict()
    print(f"Runs per model: {counts}")

    table = mean_std_table(runs)
    table.to_csv(out_dir / "table8_mean_std.csv")
    print("\nTable 8: mean ± std across ten seeds (%)\n" + table.to_string())

    tests = wilcoxon_table(runs)
    tests.to_csv(out_dir / "table8_wilcoxon_holm.csv", index=False)
    print("\nWilcoxon signed-rank (two-sided) + Holm correction\n" + tests.to_string(index=False))
    print(f"\nSaved tables in {out_dir}")


if __name__ == "__main__":
    main()
