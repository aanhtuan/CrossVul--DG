# CrossVul-DG — submission package

Class-Conditional Domain Generalization for Cross-Project Software
Vulnerability Detection. Three deliverables:

```
CrossVul-DG-Package/
├── CrossVul-DG_Proposal.docx   # research proposal (method + detailed experiments + 12-week plan)
├── paper/
│   ├── main.tex                # IEEEtran conference paper (CC-DRO), double-blind
│   ├── main.pdf                # compiled 5-page preview
│   └── README.md
└── code/                       # runnable repository (see code/README.md)
```

## What changed vs. the previous version
- **New method (the contribution): CC-DRO** — class-conditional DRO over
  (project × class) cells + a cross-project positive-class risk-invariance
  penalty, targeting the coupled domain-shift + class-imbalance problem.
- **Imbalanced setting handled honestly and identically for every method:**
  class-weighted loss (`pos_weight`), validation-tuned decision threshold,
  **PR-AUC as the primary metric**, plus recall@P50 and worst-project F1.
- **DA leakage fixed:** DA baselines use the *earlier* portion of the held-out
  project as the unlabelled target; all methods are scored on the *later*
  portion (identical test set).
- **Multi-backbone:** CodeBERT (main) + UniXcoder, GraphCodeBERT (generality).
- **Statistics:** paired Wilcoxon signed-rank tests; auto-generated LaTeX tables.
- **Tiered experiment matrix** sized for a single 24 GB GPU within ~12 weeks.

## The one thing left to do: run the experiments
The manuscript/proposal numbers are placeholders. On your GPU machine:

```bash
cd code
pip install -r requirements.txt
python scripts/run_sanity_check.py            # ~1 min, verifies the pipeline incl. CC-DRO
python scripts/prepare_data.py --raw_dir data/megavul_raw --out_dir data/processed \
    --top_k 10 --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --dedup
bash scripts/run_full_sweep.sh                # Tier 1 (main table)
bash scripts/run_tier2_backbones.sh           # Tier 2 (backbone generality)
python scripts/aggregate_results.py --runs_dir runs/ --out paper_tables/
```

Then paste `paper_tables/main_table.tex` into the paper and fill the remaining
tables/figure with measured values.

## Note on recent references
Two near-cutoff citations (CEVD, *Future Internet* 2026; VulGate, arXiv
2508.16625, 2025) were carried over from the prior draft — please verify the
venue, year, and exact figures against the primary sources before submission.
