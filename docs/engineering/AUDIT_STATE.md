# CrossVul-DG current handoff

Updated: 2026-10-20, Asia/Ho_Chi_Minh. Branch: `fix/training-pipeline-audit`.
Baseline Git commit: `9ae55d80752245cfc60df577aa065f3d8c1fd742`.
All local edits remain uncommitted. No push, merge, PR or cloud upload.
Preserve papers/archives, original raw data, and existing experiment evidence.

## Completed phases / VERIFIED

1. Located original `megavul/megavul.json` (2,668,796,982 bytes). Folder contains
   only data, no source checkout or graphs. Canonical hardlink at
   `data/raw/megavul/megavul.json`, original preserved unchanged.
   SHA256: eabe8d3325be358d7a93327e3ae4f001725517a0067c4d80198f89904177419f.
2. Streamed all 353873 records: 17975 positive / 335898 negative. All required
   types/dates/IDs checked. Missing dates/CVE/commit counts zero.
3. Cleaned to 330345 rows (15733 positive / 314612 negative), 1054 projects.
   Top10 167470; tail 162875. Exclusions: 458 unchanged-before/after, 17016
   quarantine union, 6054 further duplicates. Detailed distribution in manifests.
4. FIXED real export failure at DuckDB 256 MB by bounding Parquet row groups to
   2048. Regression passed, full rerun 115.05 s, peak RSS 554144 KiB. Idempotence
   checked: same signature/hashes and unchanged Parquet mtime.
5. All 10 full metadata LOPO folds PASSED overlap/project/time checks. Historical
   LOPO is not global forward-time evaluation. Bounded near screen: 2000 rows,
   195 comparisons, 16 same-project candidate pairs; none crosses partitions in
   registered Linux fold or has both members in the controlled benchmark.
6. Exact token histogram over all clean rows: median206, p90=885, p95=1249,
   p99=2394, max42440; truncation128/256/512 =67.7355%/42.3851%/21.6743%.
7. Separate `.venv-gpu`: torch2.10.0+cu128, CUDA12.8, driver610.57.04. CPU env
   retained. CPU tests90 pass/3CUDA skip; native GPU93 pass/0skip. Native replay
   gradients include FP16 and checkpointed tiny RoBERTa.
8. Full pretrained CodeBERT tiny real fit: 16 real rows, 50 updates, fit AP/F1=1.
   This is diagnostic only. GPU allocated peak2381476352 bytes.
9. Matched real pilot: held torvalds/linux, seed0, train288/val128/test256,
   target128 unused by DG. Exact hash/selection policy recorded before metrics.
   Full source/val/test fold sizes75241/7247/35577 were NOT trained here.
10. Real ERM and CC-DRO completed; five additional validation-only ablations
    completed. Best lambda=.1, LR2e-5, length128 chosen solely by validation AP.
    Frozen selection preceded its single test evaluation; no extra optimizer steps.

## Measured bounded scores

| Run | Val AP | Test AP | Test F1 |
|---|---:|---:|---:|
| ERM | .1766742972 | .2282040608 | .1282051282 |
| CC-DRO lambda1 | .2241249462 | .1182639177 | 0 |
| Selected CC-DRO lambda.1 | .3099452878 | .1292532262 | .1111111111 |

CC-DRO did not outperform ERM in test AP/F1 on this cohort. No general improvement
or full-population claim. Confusion matrices, thresholds, prevalence and exact
configs are in TRAINING_RESULTS.md and REAL_VALIDATION.json.

## Confirmed failures / fixes / recovery

- Prior audit's label/leakage/ERM/dual/checkpoint/metric defects remain fixed and tested.
- Real DuckDB writer-budget failure FIXED and re-tested on entire input.
- One ERM AMP overflow step was skipped safely (287 valid vs288 planned steps).
  Model/optimizer finite checks passed. CC-DRO baseline/selected had zero AMP skips.
- CC-DRO watchdog stopped AFTER valid epoch2 checkpoint when global swap grew
  >1GiB. Matching checkpoint resumed evaluation with no further optimization;
  history/identity preserved. No CUDA OOM, no system freeze.
- Checkpoint best models and optimizer states of all three reported runs are
  explicitly finite. Separate latest and best checkpoint semantics retained.

## Evidence / reproduction pointers

- REAL_DATA_REPORT.md: original paths, full counts, exclusions, leakage, tokens.
- TRAINING_RESULTS.md: measured metrics, configs, sampling/step caveats, resources.
- OPTIMIZATION_REPORT.md: six validation configurations, frozen choice, limitations.
- `data/manifests/`: acquisition, dataset_audit, fold_audit, token_lengths,
  controlled_cohort, validation_selection, sequence_capacity, checkpoint_validation.
- `runs/real_data_integration/`: complete logs, watchdog JSON and commands.
- `runs/real_baselines/bounded-v1/`: matched ERM/CC-DRO checkpoints/results.
- `runs/real_ablations/bounded-v1/`: validation-only runs; only chosen lambda01
  subsequently evaluated on test. Baseline point has explicit extraction origin.
- Executable source signature for measured jobs:
  fc06e02b2ef74100a5c6fd5bb24e36fa800749a53bb0fd21b1c2a70f8cf4c5de.
- README supplies exact commands, environment locks and manual remote workflow.

## Remaining limitations / NOT RUN

- User-supplied release metadata and publisher checksum are unavailable. Local
  SHA and full schema are verified; independent publisher authentication is not.
- Full-population training, multi-project/seed evaluation and robustness variance
  remain NOT RUN. Current result is a resource-bounded pilot, not final research.
- Local length256/512 training NOT RUN: conservative idle-GPU preflight estimates
  exceed available VRAM. These are estimates, not measured OOMs or proof no other
  implementation can fit. Manual Colab/Kaggle source workflow is prepared, not run.
- Near-duplicate screen is bounded lexical evidence, not exhaustive clone detection.
- Small support, repeated positives, fewer CC optimizer steps, truncation and
  project/calibration shift remain research hypotheses. Test was never tuned.

## Next research work (do not auto-start another sweep)

1. Use larger source budgets and full-population ERM/CC baselines on suitable compute.
2. Keep existing test and selection artifacts frozen; evaluate new settings by source validation.
3. Run matched length256/512 and update/sampling-budget studies with resource checks.
4. Replicate over projects/seeds before reporting mean/variance/worst-project robustness.
5. Manual source bundle + colab_real_training.ipynb avoids cloning stale uncommitted code.

## Final delivery checks

- Best-model parameters and optimizer state of all three reported runs explicitly finite.
- Final selected evaluation exited 0; no new optimization and unchanged history/signature.
- Real analysis cells executed headlessly against measured baselines; two affected notebook regressions passed.
- Colab notebook cells compile; remote execution NOT RUN. Manual source bundle prepared.
- Executable source hash still matches all measured runs. Git whitespace checks passed.
- Final original preservation check: same inode, SHA256 and original mtime.
