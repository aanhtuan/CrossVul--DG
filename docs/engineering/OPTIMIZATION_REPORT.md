# Controlled optimization report

Date: 2026-10-20, Asia/Ho_Chi_Minh.
**Scope: one preregistered bounded real-data fold; no generalization guarantee.**

## VERIFIED — baseline before optimization

ERM and CC-DRO were first run with the same CodeBERT backbone, input revision,
source/validation/test hashes, held-out project and seed. The complete measured
point results are in `TRAINING_RESULTS.md`. Baseline CC-DRO had higher validation
AP but worse test AP/F1 than ERM. No change was chosen from test performance.

## TESTED — one-variable validation-only ablations

All runs use the same 288 / 128 / 256 real train / validation / test IDs, source
seed 0, two epochs, batch 2, invariance warmup 1, eta .01 and the same executable
source/software signature. The baseline is lambda=1, LR=2e-5, length=128.
Only the indicated lambda or LR changes in each row. Six unique configurations
were measured: the existing baseline plus five new training runs.

| Variant | lambda_inv | LR | Length | Validation AP | Validation F1 |
|---|---:|---:|---:|---:|---:|
| base | 1.0 | 2e-05 | 128 | 0.224125 | 0.320000 |
| lambda0 | 0 | 2e-05 | 128 | 0.280149 | 0.320000 |
| lambda001 | 0.01 | 2e-05 | 128 | 0.226684 | 0.320000 |
| lambda01 | 0.1 | 2e-05 | 128 | 0.309945 | 0.320000 |
| lr1e-5 | 1.0 | 1e-05 | 128 | 0.270286 | 0.333333 |
| lr3e-5 | 1.0 | 3e-05 | 128 | 0.215151 | 0.333333 |

New ablation runs use `--selection_only`: no test forward pass, no test threshold
or test metric is computed. The baseline validation point is explicitly marked
as extracted from a completed preregistered baseline, with its already measured
test fields removed from selection. The selector reads only validation AP.

## VERIFIED — frozen choice and one final test evaluation

Best among these six configurations: **lambda_inv=0.1, LR=2e-5, max_length=128**,
validation AP **0.3099452878**. Other common settings remain unchanged.
`data/manifests/validation_selection.json` was written before the selected
checkpoint's test result. Its timestamp and matching experiment signature were
checked. Reusing the selected completed checkpoint added zero optimizer steps.

The one subsequent test evaluation gave AP **0.1292532262**, ROC-AUC
**0.6521050965**, F1 **0.1111111111**, precision **0.2**, recall **0.0769230769**,
confusion matrix `[[239,4],[12,1]]`, prevalence **13/256**. Threshold
**0.5257188678** was selected on validation. ERM remains better in test AP and F1
on this cohort. No second threshold or parameter search followed this result.

This is the best *measured validation configuration in this cohort*, not a
universal optimum. The small validation population has only 12 positives;
selection uncertainty and overfitting to validation remain material limitations.

## VERIFIED observations versus HYPOTHESIS

1. **Class support and sampling — VERIFIED:** only 15 positive source examples
   appear in the 288-row compute cohort, and individual projects have 1–3 positives.
   ERM sees every source example once/epoch; CC-DRO repeatedly draws those rare
   positives. Its recorded epoch-2 unique samples per environment are 12–16 out
   of 32 draws. **HYPOTHESIS:** repeated positive support and sampling variance
   limit transferable vulnerability features. Do not remove hard negatives.
2. **Optimizer exposure — VERIFIED:** ERM took 287 valid optimizer steps versus
   32 CC-DRO coupled steps for the same 576 source draws over two epochs.
   CC-DRO uses a larger effective multi-environment batch. **HYPOTHESIS:** it may
   require a separately validated schedule/update budget. These runs do not
   isolate sampling, objective and optimizer-step effects from each other.
3. **Invariance scale — VERIFIED:** baseline CC-DRO's active-epoch classification
   loss averaged about .647878, weighted penalty .004706, mean penalty/risk ratio
   about .007254. Risk-coefficient norms were about .235706 for classification
   and .044110 for invariance. There is no scalar-loss evidence that lambda=1
   dominated classification here. Coefficient norms are not parameter-gradient
   norms; total pre-clipping gradient norm is logged separately.
4. **Domain/calibration shift — VERIFIED:** validation prevalence 9.375% versus
   test 5.078125%; selected model makes 5 positive test predictions, only one true.
   Baseline CC-DRO makes 3, none true. **HYPOTHESIS:** project shift, small support
   and calibration transfer contribute. The exact causal contribution is unmeasured.
   Test thresholds were not retuned to compensate.
5. **Truncation — VERIFIED:** 67.7355% of all clean functions exceed 128 tokens.
   **HYPOTHESIS:** losing later code may weaken transfer. Longer length is not
   asserted beneficial without a matched validation experiment.
6. **Numerics — VERIFIED:** one ERM AMP overflow update was skipped and logged;
   baseline/selected CC-DRO had zero skipped steps. Full best-model and optimizer
   finiteness checks are recorded in `checkpoint_validation.json`. No CUDA OOM.
   Native FP16 replay gradients match a simultaneous graph within the tested
   precision tolerance, including checkpointed tiny RoBERTa. This preserves the
   objective formula, not bitwise equivalence to full FP32 arithmetic.

## NOT RUN — sequence lengths and larger evaluation

Idle-GPU preflight measured 3,787,063,296 free CUDA bytes. Conservative requirements:
128 = 3,121,741,824 bytes (passed); 256 = 4,732,354,560 (rejected);
512 = 11,174,805,504 (rejected). These are declared estimates, not measured OOMs
or proof that every alternative implementation is incapable of fitting.
Longer-length training was not forced. See `data/manifests/sequence_capacity.json`.

Full-population training, larger training budgets, multiple projects/seeds and
robust mean/variance/worst-project conclusions are NOT RUN. With about 75k
source rows in the full Linux fold, full fine-tuning costs orders of magnitude
more updates than the local pilot. The current results establish working real
training and diagnostics within an explicit compute budget, not final paper scores.

## Reproduction and remote continuation

```sh
# Print the full-data one-variable plan; execute jobs sequentially after preflight.
.venv-gpu/bin/python scripts/plan_real_ablations.py --held_out torvalds/linux
# Select among compatible validation-only artifacts, without reading test metrics.
.venv-gpu/bin/python scripts/select_validation_run.py --runs_dir runs/real_ablations/bounded-v1 --method cc-dro --out data/manifests/validation_selection.json
```

For exact local bounded commands use `commands.json` in that runs directory or
the common-argument block in `TRAINING_RESULTS.md`; preserve target_limit=128
as well as source/validation/test caps. The general printed full-data plan uses
zero caps. Evaluate only the frozen winner by repeating its matching command
without `--selection_only`; the completed checkpoint resumes with no extra steps.
Use fresh output directories for changed identities.

A local source bundle and `notebooks/colab_real_training.ipynb` provide a manual
Colab/Kaggle path for larger-memory/longer-length/full-population work. The notebook
checks the actual available GPU, preserves platform CUDA PyTorch, preprocesses
streamingly, runs full baselines and prints validation-only ablations. It has not
been executed in Colab/Kaggle and was not uploaded automatically. Since local fixes
are uncommitted, do not substitute a clone of the older GitHub remote for the bundle.

Future priority: larger source support and replicated full-population baselines;
then matched length 256/512, update/sampling-budget ablations and additional
projects/seeds. Keep held-out test evaluation frozen during selection.
