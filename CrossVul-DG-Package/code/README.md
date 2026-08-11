# CrossVul-DG

Code for the paper:
**"CrossVul-DG: Class-Conditional Domain Generalization for Cross-Project
Software Vulnerability Detection"**

CrossVul-DG studies invariance/robustness-based **Domain Generalization (DG)**
for cross-project vulnerability detection on MegaVul, under the realistic
constraint of **no target-project data at training time**. Its central
contribution is **CC-DRO**, a *class-conditional* distributionally-robust
objective designed for the coupled domain-shift + class-imbalance problem that
SVD actually faces.

> **⚠️ The numbers in the manuscript are placeholders until the sweep is run.**
> Run `scripts/run_full_sweep.sh` (Tier 1) and `scripts/run_tier2_backbones.sh`
> (Tier 2) on real MegaVul, then regenerate the tables with
> `scripts/aggregate_results.py`. `scripts/run_sanity_check.py` only verifies the
> pipeline on tiny synthetic data.

---

## 1. What is CC-DRO?

Cross-project SVD couples two shifts: a **domain** shift across projects and a
severe **class imbalance** (MegaVul is ≈1 vulnerable : 19 safe). Off-the-shelf
DG objectives (IRM, GroupDRO) take the *project* as the unit of invariance and
are **class-blind**: under 1:19 imbalance, worst-case *project* risk can be
driven down by excelling on the dominant safe class, leaving the rare-but-
critical vulnerability signal under-served and inconsistent across projects.

**CC-DRO** changes the group structure and adds an invariance term:

1. **Class-conditional groups.** Groups are `(project × class)` cells → `2K`
   cells for `K` source projects. GroupDRO is run over these cells so the worst
   case *explicitly includes* the `(project p, vulnerable)` cells.
2. **Cross-project positive-class risk invariance.** A penalty equal to the
   variance, across projects, of the positive-class (vulnerable) risk. Minimising
   it forces the model's vulnerability *recall* to be consistent from one project
   to the next.

Objective per optimisation step:

```
L = Σ_g q_g · R_g  +  λ_inv · Var_p( R_{p,pos} )
```

where `g` ranges over the `(project × class)` cells present in the step, `q` is
the exponentiated-gradient DRO weight, and `R_{p,pos}` is project `p`'s mean loss
on its positive samples. See `src/losses.py` and `src/trainers.py`.

> **Relation to prior work.** CEVD (Liu, *Future Internet* 2026) is a DG method
> that clusters embeddings and ensembles per-cluster experts. CrossVul-DG instead
> applies invariance/robustness objectives over the *natural project*
> environments, and CC-DRO additionally conditions on class — neither CEVD nor
> vanilla IRM/GroupDRO does this. VulGate (Safdar et al., 2025) is a
> generalization-focused *dataset*, orthogonal to the learning objective here.

---

## 2. Imbalanced-setting protocol (applied identically to every method)

Because MegaVul is kept at its natural ≈1:19 ratio, all methods share:

- **Weighted BCE** with `pos_weight = N_neg/N_pos` on the training fold
  (`src/data.compute_pos_weight`, capped for stability).
- **PR-AUC for model selection** on val (threshold-free; val-F1@0.5 is unreliable
  under imbalance).
- **Decision threshold tuned on val** (max-F1), then applied unchanged at test —
  no test-set peeking (`src/evaluate.tune_threshold`).
- **Primary metric: PR-AUC.** Also reported: F1 at the tuned threshold,
  recall@precision≥0.5, and **worst-project F1** (robustness).
- **DA target ≠ test set.** DA baselines use the chronologically-*earlier*
  portion of the held-out project as the unlabelled target; *all* methods are
  scored on the *later* portion (`src/data.split_held_out`). This removes the
  transductive leakage of training DA on the very functions being scored.

---

## 3. Repository layout

```
crossvul-dg/
├── configs/default.yaml          # all hyperparameters (incl. cc_dro)
├── data/                         # how to obtain MegaVul; synthetic stub
├── src/
│   ├── data.py                   # loader, temporal split, LOPO, pos_weight, DA split
│   ├── models.py                 # EncoderClassifier (CodeBERT/UniXcoder/GraphCodeBERT)
│   ├── losses.py                 # weighted BCE, IRM, GroupDRO, CC-DRO, CORAL
│   ├── trainers.py               # ERM / IRM / DRO / CC-DRO / DA trainers
│   ├── pseudo_domains.py         # k-means clustering of tail projects (pd-dro)
│   ├── evaluate.py               # PR-AUC, recall@P, threshold tuning, per-project
│   └── utils.py
├── scripts/
│   ├── prepare_data.py           # MegaVul → cached parquet splits
│   ├── run_lopo.py               # one LOPO fold (entry point)
│   ├── run_full_sweep.sh         # Tier 1: CodeBERT × 7 methods × 10 folds × 3 seeds
│   ├── run_tier2_backbones.sh    # Tier 2: UniXcoder/GraphCodeBERT × 3 methods × 2 seeds
│   ├── run_sanity_check.py       # end-to-end on synthetic data (incl. CC-DRO)
│   └── aggregate_results.py      # tables + paired Wilcoxon + LaTeX
└── notebooks/analysis.ipynb
```

---

## 4. Installation & sanity check

Tested on Python 3.10, CUDA 12.1.

```bash
conda create -n crossvul-dg python=3.10 -y && conda activate crossvul-dg
pip install -r requirements.txt
python scripts/run_sanity_check.py     # ~1 min, CPU; verifies the whole pipeline incl. CC-DRO
```

---

## 5. Reproducing the paper

**Step 1 — MegaVul.** See `data/README.md`. Place the raw release under
`data/megavul_raw/`.

**Step 2 — Preprocess** (≈3 min):

```bash
python scripts/prepare_data.py --raw_dir data/megavul_raw --out_dir data/processed \
    --top_k 10 --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --dedup
```

**Step 3 — One fold** (sanity on real data):

```bash
python scripts/run_lopo.py --config configs/default.yaml \
    --method cc-dro --held_out linux --seed 0 \
    --data_path data/processed/top10.parquet \
    --output_dir runs/codebert/cc-dro/linux/seed0
```

**Step 4 — Full sweep, then tables:**

```bash
bash scripts/run_full_sweep.sh         # Tier 1 (main table)
bash scripts/run_tier2_backbones.sh    # Tier 2 (backbone generality)
python scripts/aggregate_results.py --runs_dir runs/ --out paper_tables/
```

---

## 6. Methods (`--method`)

| Flag       | Description                                                  |
|------------|--------------------------------------------------------------|
| `erm`      | Standard ERM fine-tuning (ignores project labels)            |
| `irm`      | Invariant Risk Minimization (IRMv1)                          |
| `dro`      | GroupDRO over project groups                                 |
| `cc-dro`   | **Proposed**: class-conditional DRO + positive-risk invariance |
| `pd-dro`   | Pseudo-domain clustering of tail projects + GroupDRO         |
| `da-coral` | Deep CORAL (DA baseline; uses unlabelled target)             |
| `da-dann`  | DANN (DA baseline; uses unlabelled target)                   |

DA baselines are *not* strict DG (they touch the unlabelled, earlier portion of
the held-out project). They serve as a reference: *can DG match DA with no target
access?*

---

## 7. Compute budget (per fold, single 24 GB GPU, CodeBERT-base)

| Setting                            | Approx. wall-clock |
|------------------------------------|--------------------|
| Single fold, 1 seed                | ~45–70 min         |
| Tier 1 (7×10×3 = 210 runs)         | ~7–10 days         |
| Tier 2 (2×3×10×2 = 120 runs)       | ~4–6 days          |
| Synthetic sanity                   | <2 min (CPU)       |

To fit a deadline you can: reduce Tier 1 seeds to 2, evaluate the most-contended
DA baseline on a subset of folds, or freeze the encoder for IRM. See the proposal
for the prioritised reduction plan.

---

## 8. License

Code: MIT. MegaVul data is subject to its upstream license.
