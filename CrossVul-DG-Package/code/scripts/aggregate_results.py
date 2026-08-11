"""Aggregate per-fold per-seed results into paper-ready tables + significance tests.

Walks ``--runs_dir`` for ``results.json`` files produced by ``run_lopo.py`` and,
for each (backbone, method), reports across the LOPO folds:

  * PR-AUC          (primary, threshold-free)
  * F1@tuned-thr    (threshold tuned per fold on val)
  * recall@P50      (recall at precision >= 0.5)
  * worst-project F1 (min over folds -- the robustness number)

Each fold value is first averaged over seeds. Paired Wilcoxon signed-rank tests
(over the 10 folds) compare every method against ERM, and CC-DRO against
GroupDRO. A LaTeX table is emitted for direct inclusion in the paper.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scipy.stats import wilcoxon
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

PRIMARY = "pr_auc"
METRICS = ["pr_auc", "f1", "recall_at_p50"]
METHOD_ORDER = ["erm", "da-coral", "da-dann", "irm", "dro", "cc-dro", "pd-dro"]
PRETTY = {
    "erm": "ERM-FT", "da-coral": "DA-CORAL$^\\dagger$", "da-dann": "DA-DANN$^\\dagger$",
    "irm": "IRM", "dro": "GroupDRO", "cc-dro": "\\textbf{CC-DRO (ours)}",
    "pd-dro": "PD+DRO",
}


def fold_means(folds: dict, metric: str) -> dict:
    """held_out -> mean-over-seeds value of ``metric``."""
    out = {}
    for held, seeds in folds.items():
        vals = [m[metric] for m in seeds.values()
                if metric in m and m[metric] == m[metric]]   # drop NaN
        if vals:
            out[held] = float(np.mean(vals))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("paper_tables"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # backbone -> method -> held_out -> seed -> test metrics
    bucket = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for p in args.runs_dir.rglob("results.json"):
        with open(p) as f:
            r = json.load(f)
        a = r["args"]
        backbone = a.get("backbone", "microsoft/codebert-base").split("/")[-1]
        bucket[backbone][a["method"]][a["held_out"]][a["seed"]] = r["test"]

    summary = {}
    for backbone, methods in bucket.items():
        print(f"\n===== backbone: {backbone} =====")
        header = f"{'method':<18}" + "".join(f"{m:>14}" for m in METRICS) + f"{'worstF1':>10}"
        print(header); print("-" * len(header))

        rows = {}
        for method in [m for m in METHOD_ORDER if m in methods] + \
                      [m for m in methods if m not in METHOD_ORDER]:
            folds = methods[method]
            line = f"{method:<18}"
            row = {}
            for metric in METRICS:
                fm = fold_means(folds, metric)
                if fm:
                    vals = np.array(list(fm.values()))
                    mu, sd = vals.mean(), vals.std()
                    line += f"{mu*100:7.1f}±{sd*100:.1f}"
                    row[metric] = {"mean": float(mu), "std": float(sd),
                                   "per_fold": fm}
                else:
                    line += f"{'-':>14}"
            f1_fm = fold_means(folds, "f1")
            worst = min(f1_fm.values()) if f1_fm else float("nan")
            row["worst_f1"] = float(worst)
            line += f"{worst*100:9.1f}"
            print(line)
            rows[method] = row
        summary[backbone] = rows

        # ---- paired Wilcoxon over folds (primary metric) ----
        if _HAVE_SCIPY:
            print(f"\nPaired Wilcoxon (folds, {PRIMARY}):")
            comparisons = [(m, "erm") for m in methods if m != "erm"]
            if "cc-dro" in methods and "dro" in methods:
                comparisons.append(("cc-dro", "dro"))
            for a_m, b_m in comparisons:
                fa, fb = fold_means(methods.get(a_m, {}), PRIMARY), \
                         fold_means(methods.get(b_m, {}), PRIMARY)
                common = sorted(set(fa) & set(fb))
                if len(common) >= 6:
                    da = np.array([fa[k] for k in common])
                    db = np.array([fb[k] for k in common])
                    if np.allclose(da, db):
                        print(f"  {a_m} vs {b_m}: identical, skipped")
                        continue
                    stat, pval = wilcoxon(da, db)
                    sig = "*" if pval < 0.05 else " "
                    print(f"  {a_m:>8} vs {b_m:<8} n={len(common)} "
                          f"Δ={100*(da.mean()-db.mean()):+.1f}  p={pval:.4f} {sig}")
        else:
            print("\n[scipy not installed -> Wilcoxon skipped; pip install scipy]")

    # ---- LaTeX main table (PRIMARY backbone = codebert if present) ----
    main_bb = "codebert-base" if "codebert-base" in summary else next(iter(summary))
    _write_latex(summary[main_bb], args.out / "main_table.tex", main_bb)

    with open(args.out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.out/'summary.json'} and {args.out/'main_table.tex'}")


def _write_latex(rows: dict, path: Path, backbone: str) -> None:
    lines = [
        "% Auto-generated by aggregate_results.py -- do not edit by hand.",
        "\\begin{table}[t]", "\\centering",
        f"\\caption{{Leave-one-project-out results on MegaVul ({backbone}). "
        "Mean over 10 folds (each averaged over seeds). PR-AUC is the primary "
        "metric; worst-project F1 measures robustness. $^\\dagger$ uses "
        "unlabelled target data.}",
        "\\label{tab:main}",
        "\\begin{tabular}{lcccc}", "\\toprule",
        "Method & PR-AUC & F1$_\\text{thr}$ & R@P50 & worst-F1 \\\\", "\\midrule",
    ]
    for method in [m for m in METHOD_ORDER if m in rows]:
        r = rows[method]
        def cell(metric):
            if metric in r and isinstance(r[metric], dict):
                return f"{r[metric]['mean']*100:.1f}\\,$\\pm$\\,{r[metric]['std']*100:.1f}"
            return "--"
        worst = r.get("worst_f1", float("nan"))
        lines.append(f"{PRETTY.get(method, method)} & {cell('pr_auc')} & "
                     f"{cell('f1')} & {cell('recall_at_p50')} & "
                     f"{worst*100:.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
