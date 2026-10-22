# Real CodeBERT training results

Date: 2026-10-20, Asia/Ho_Chi_Minh. **Scope: one bounded real-data LOPO fold, seed 0.**
This report contains measured results, not full-population or multi-project claims.

## VERIFIED — common experimental population

Held-out `torvalds/linux` was preregistered as the largest cleaned project, using
row counts rather than scores. Every run uses the same 288 source rows from nine
projects (15 positives), 128 source-validation rows (12 positives), and 256 late
held-out test rows (13 positives). An unused DG target pool is capped at 128 rows.
Hashes and original full partition counts are in `data/manifests/controlled_cohort.json`.

The compute-budget protocol is `bounded-real-lopo-v1`: training has proportional
per-project class quotas with both classes retained; validation/target/test caps
are deterministic SHA-256 rankings without label-based selection. No example
was dropped according to model correctness, difficulty, or scores. These caps
are separate from the full dataset's quality exclusions. Full unbounded train /
validation / test counts are 75,241 / 7,247 / 35,577 and were NOT trained here.

All runs fine-tune the complete pretrained CodeBERT encoder plus classifier,
using revision `3b0952feddeffad0063f274080e3c23d75e7eb39`, seed 0, length 128,
batch size 2, FP16 autocast with float32 BCE/master parameters, non-reentrant
gradient checkpointing, AdamW LR 2e-5, weight decay .01, clipping 1, two epochs,
and no warmup. CC-DRO uses class-balanced cells, unweighted cell BCE, eta .01,
and invariance warmup of one epoch. ERM uses pooled shuffling and source-only
positive weight 18.2. The selected CC-DRO changes only lambda_inv from 1 to .1.

## TESTED — real diagnostic and completed scores

The separate real tiny-fit check used 16 real rows (8 per class), 50 updates,
and achieved training-sample AP/F1 1.0. This only establishes that the pretrained
model can fit the diagnostic; it is not generalization evidence.

| Run | Validation AP | Test AP | Test ROC-AUC | Test F1 | Test precision | Test recall |
|---|---:|---:|---:|---:|---:|---:|
| ERM | 0.176674 | 0.228204 | 0.631371 | 0.128205 | 0.076923 | 0.384615 |
| CC-DRO lambda=1 | 0.224125 | 0.118264 | 0.649573 | 0.000000 | 0.000000 | 0.000000 |
| CC-DRO lambda=0.1 (validation-selected) | 0.309945 | 0.129253 | 0.652105 | 0.111111 | 0.200000 | 0.076923 |

Positive prevalence is 9.375% on validation and 5.078125% on test for every run.
AP means Average Precision. Each F1/precision/recall uses its own threshold chosen
on validation, never on test. Confusion matrices use `[[TN, FP], [FN, TP]]`:

- ERM: `[[183, 60], [8, 5]]`; validation threshold `0.005598249`.
- CC-DRO lambda=1: `[[240, 3], [13, 0]]`; validation threshold `0.532303572`.
- CC-DRO lambda=0.1 (validation-selected): `[[239, 4], [12, 1]]`; validation threshold `0.525718868`.

**VERIFIED:** CC-DRO did not outperform ERM in test AP or F1 on this cohort.
Reducing lambda raised CC-DRO validation AP and changed its test point, but that
is not evidence of a generalizable improvement. The selected run has higher test
precision and lower recall than ERM at their respective validation thresholds.
No test-based threshold correction or hyperparameter search was performed.

The selection manifest was written before the selected run's test result.
All other new ablation files contain validation only. The baseline's validation
point was extracted from its preregistered completed result with an explicit
origin flag; its test metrics were excluded from the selector.

## VERIFIED — optimization and resume evidence

ERM completed 287 optimizer updates: one of 288 planned updates was safely
skipped by GradScaler after overflow, reducing scale from 1024 to 512. No
nonfinite parameters or probabilities were accepted; metrics validation passed.
The epoch containing the overflow has an undefined aggregate gradient norm
serialized as null, while the skipped-step counter remains explicit.

CC-DRO completed 32 coupled optimizer updates, zero AMP skips, with all 18 cells
represented. Both methods drew 288 source examples per epoch, but their effective
batch and optimizer-step counts differ: CC-DRO aggregates nine environments per
update and repeats rare positives. This is an explicit comparison limitation.
All source rows are available to both methods; their sampling objectives differ.

The CC-DRO watchdog stopped after the valid second-epoch latest/best checkpoint
had been atomically saved because system swap grew more than 1 GiB. Its lowest
observed available RAM was still about 1.91 GiB, but the conservative guard acted
before risking further pressure. Exact checkpoint resume completed evaluation;
no additional optimizer steps were taken. Native resumed history is identical.
The selected ablation checkpoint was also evaluated without additional training.

## VERIFIED — resources

Hardware: Fedora Linux, i7-12700H, about 15 GiB usable RAM, RTX 3050 Laptop 4 GiB.
CUDA environment: Python 3.12.14, PyTorch 2.10.0+cu128, CUDA runtime 12.8,
NVIDIA driver 610.57.04. CPU verification environment is retained separately.
`requirements-gpu.lock` records the tested dependency versions.

| Process | Guard wall seconds | Peak RSS GiB | Peak process GPU GiB | Outcome |
|---|---:|---:|---:|---|
| Tiny real fit | 87.19 | 1.783 | 2.529 | Completed |
| ERM | 60.80 | 2.556 | 2.527 | Completed |
| CC-DRO training | 43.17 | 3.013 | 2.531 | Guarded stop; valid checkpoint recovered |
| CC-DRO resumed evaluation | 20.26 | 3.065 | 1.783 | Completed |
| Selected CC-DRO evaluation | 16.18 | 3.355 | 1.783 | Completed |

Guard wall time includes process startup and teardown. CC-DRO training plus
resume totals about 63.43 seconds of guarded execution. The selected evaluation
row excludes the earlier lambda=.1 training time; its original training resource
record is `ablation_lambda01_guard/resources.json`. Do not use the resumed
runner's shorter internal runtime/peak as the total training measurement.

Actual full-model training peak allocated CUDA memory is about 2.38 GB, with
about 2.56 GB reserved; NVML process peaks include CUDA context/library memory.
Watchdogs checked available RAM, global swap growth, runtime and child RSS/GPU
use every two seconds. No CUDA OOM occurred. One DuckDB export budget failure
was fixed, and one proactive training-process stop was recovered.

## NOT RUN / HYPOTHESIS

- Full-population training, multiple held-out projects and multiple seeds: NOT RUN.
  Cross-project mean, variance and worst-project robustness cannot be estimated
  from one fold/seed. A descriptive one-fold mean/worst equals its point value;
  a zero one-observation population standard deviation is not robustness evidence.
- Length 256/512: NOT RUN after conservative idle-GPU preflight rejection.
  This is a budget estimate, not a measured CUDA OOM or proof of absolute hardware incapacity.
- Weak transfer, small positive support, sparse optimizer updates, threshold
  calibration shift and code truncation are hypotheses discussed in
  `OPTIMIZATION_REPORT.md`; no unmeasured causal improvement is asserted.

## Reproduction

Use a fresh output namespace if code/data/environment identities differ.
The original completed runs skip only when exact experiment identity matches.

```sh
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MALLOC_ARENA_MAX=2
CV_COMMON_ARGS=(--config configs/local_gpu.yaml --held_out torvalds/linux --seed 0 --data_path data/processed/top10.parquet --train_per_project 32 --val_limit 128 --target_limit 128 --test_limit 256)
.venv-gpu/bin/python scripts/run_lopo.py "${CV_COMMON_ARGS[@]}" --method erm --output_dir runs/reproduce/bounded/erm
.venv-gpu/bin/python scripts/run_lopo.py "${CV_COMMON_ARGS[@]}" --method cc-dro --output_dir runs/reproduce/bounded/cc-dro
.venv-gpu/bin/python scripts/run_lopo.py "${CV_COMMON_ARGS[@]}" --method cc-dro --output_dir runs/reproduce/ablations/lambda01 --selection_only --override cc_dro.lambda_inv=0.1
# Freeze the validation selection before repeating the chosen command without --selection_only.
```

For watchdog protection wrap any child command as:

```sh
.venv/bin/python scripts/run_guarded.py --out runs/reproduce/resource_log --timeout_seconds 900 -- .venv-gpu/bin/python scripts/run_lopo.py ...
```

Use the concrete commands above in place of `...`. Original per-job commands,
exit codes and resource snapshots are under `runs/real_data_integration/`;
validation ablation commands are in `runs/real_ablations/bounded-v1/commands.json`.
For full-population remote execution use `notebooks/colab_real_training.ipynb`
and the local source bundle; omit all compute caps. No automatic upload occurs.

Real analysis notebook cells were executed headlessly on the completed baseline cohort; `figures/cohort_0_ap.png` was generated. The Colab/Kaggle notebook was syntax-checked only.
