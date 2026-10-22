# CrossVul-DG

Class-conditional domain generalization for cross-project vulnerability detection.
Active package: `src/crossvul_dg/`. Historical papers and archives are preserved
under `CrossVul-DG-Package/`; their illustrative scores are not measured results.

Current evidence (2026-10-20): **353,873 real MegaVul records validated; 330,345
clean rows; all 10 metadata LOPO folds verified; 93 native CPU/CUDA tests passed.**
Full CodeBERT training ran on one explicitly bounded real-data fold. These are
pilot results, not full-population or multi-project research conclusions.

- [Real data and quality report](docs/engineering/REAL_DATA_REPORT.md)
- [Real training metrics and resources](docs/engineering/TRAINING_RESULTS.md)
- [Validation-only optimization](docs/engineering/OPTIMIZATION_REPORT.md)
- [Current handoff state](docs/engineering/AUDIT_STATE.md)
- [Historical initial engineering audit](docs/engineering/AUDIT_REPORT.md)

## Structure

```text
configs/                   Effective YAML configurations
src/crossvul_dg/            Data, models, losses, trainers, metrics, provenance
scripts/                   Prepare/audit/train/guard/select/aggregate tools
tests/                    Unit, integration and regression fixtures
data/raw/megavul/          Canonical raw megavul.json (hardlink, ignored)
data/processed/            top10.parquet, tail.parquet, splits.json (ignored)
data/manifests/            Trackable quality/provenance/split/selection metadata
runs/                      Ignored checkpoints, results and resource logs
experiments/outputs/       Ignored diagnostic artifacts and source bundle
notebooks/                 Cohort-aware analysis and manual Colab/Kaggle workflow
docs/engineering/          Reports, validation evidence and persistent handoff
```

## Dataset identity and preparation

User-supplied original: `megavul/megavul.json`, 2,668,796,982 bytes.
Canonical hardlink: `data/raw/megavul/megavul.json`. The original is preserved.
SHA-256: `eabe8d3325be358d7a93327e3ae4f001725517a0067c4d80198f89904177419f`.
The complete flattened schema was validated against the
[official MegaVul specification](https://github.com/Icyrockton/MegaVul/blob/main/SPECIFICATION.md).
Publisher checksum/release identifier remain user-unspecified.

```sh
.venv/bin/python scripts/prepare_data.py --raw_dir data/raw/megavul --out_dir data/processed --dedup --batch_size 512 --memory_limit 256MB
.venv/bin/python scripts/audit_data.py --data_path data/processed/top10.parquet
HF_HOME="$PWD/.cache/huggingface" .venv/bin/python scripts/audit_token_lengths.py
```

JSON arrays/JSONL are streamed. Positives require `func_before`; negatives use
`func`. Invalid/ambiguous rows, conflicting/cross-project exact-code groups and
post-fix overlaps have explicit exclusion counts. CVE/commit aliases survive
deduplication. Output publication is staged, verified completed outputs are
reused, and `--as_of` reproduces the recorded timestamp bound. Interrupted raw
processing restarts the stream; it never silently publishes partial results.

Protocol v2 intentionally changes the old population. LOPO source train/validation
and held-out target/test are chronological within their respective populations,
with metadata/code overlaps purged. Historical cross-project evaluation does not
guarantee every training date precedes every test date. Near-duplicate checks
are bounded lexical diagnostics, not an exhaustive semantic-clone guarantee.

## Environments and tests

The existing CPU `.venv` is retained. `.venv-gpu` uses CUDA PyTorch 2.10.0+cu128
and was verified on the RTX 3050. Do not replace the working GPU installation with
the CPU lock. Exact version lists are `requirements-cpu.lock` and
`requirements-gpu.lock`; `requirements.txt` is the bounded portable runtime set.
Install the package with `uv pip install --python ENV/bin/python --no-deps -e .`
after installing a matching dependency environment.

To recreate GPU dependencies in a fresh environment:

```sh
uv venv --python .venv/bin/python .venv-gpu-new
UV_CONCURRENT_DOWNLOADS=1 uv pip install --python .venv-gpu-new/bin/python -r requirements-gpu.lock --extra-index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-gpu-new/bin/python --no-deps -e .
```

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-gpu/bin/python -m pytest -q
uv pip check --python .venv-gpu/bin/python
```

CPU: 90 passed, 3 CUDA skips. Native GPU: 93 passed. Tests cover strict labels,
streaming/parquet/splits, leakage, replay gradients, FP16 checkpointed RoBERTa,
all supported trainer interfaces, resume, identity rejection, aggregation and
CC-DRO chart rendering. See reports for the executed scope.

## Resource-aware real training

`configs/local_gpu.yaml`: length 128, batch 2, FP16, non-reentrant gradient
checkpointing, two epochs, zero loader workers. Optimizer states/master weights
remain float32. Coupled CC-DRO uses two-pass RNG replay and one environment graph
at a time; its objective remains `sum(q_g R_g) + lambda_inv Var(R_project,pos)`.

The real tiny-fit diagnostic:

```sh
MALLOC_ARENA_MAX=2 HF_HOME="$PWD/.cache/huggingface" .venv-gpu/bin/python scripts/run_real_diagnostics.py --config configs/local_gpu.yaml --steps 50 --per_class 8
```

A declared bounded pilot (use a fresh output namespace for changed identities):

```sh
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MALLOC_ARENA_MAX=2 PYTHONUNBUFFERED=1
CV_COMMON_ARGS=(--config configs/local_gpu.yaml --held_out torvalds/linux --seed 0 --data_path data/processed/top10.parquet --train_per_project 32 --val_limit 128 --target_limit 128 --test_limit 256)
.venv-gpu/bin/python scripts/run_lopo.py "${CV_COMMON_ARGS[@]}" --method erm --output_dir runs/reproduce/erm
.venv-gpu/bin/python scripts/run_lopo.py "${CV_COMMON_ARGS[@]}" --method cc-dro --output_dir runs/reproduce/cc-dro
```

Wrap jobs in `scripts/run_guarded.py --out RESOURCE_DIR --timeout_seconds 900 -- COMMAND...`
using the CPU interpreter to monitor available RAM, swap growth and child peak
memory. One job at a time. The watchdog stopped one CC-DRO process after a valid
completed checkpoint; exact resume finished evaluation without more optimization.
No CUDA OOM occurred. Length 256/512 local training was not run after conservative
preflight rejection. Actual capacity depends on implementation and measured state.

## Validation-only selection and results

Add `--selection_only` for ablations: this prevents test forward passes and saves
`selection.json`. Select via `scripts/select_validation_run.py`; then evaluate
only the frozen winner by repeating its matching command without that flag.
Completed checkpoints add no new optimizer steps. Latest resume and best-validation
models are separate artifacts; optimizer, scheduler, log-duals, scaler, random
and sampler state are restored. Incompatible identities are rejected.

The measured 288/128/256-row pilot gave:

| Model | Validation AP | Test AP | Test F1 |
|---|---:|---:|---:|
| Pooled ERM | 0.176674 | 0.228204 | 0.128205 |
| CC-DRO, lambda=1 | 0.224125 | 0.118264 | 0.000000 |
| Validation-selected CC-DRO, lambda=.1 | 0.309945 | 0.129253 | 0.111111 |

CC-DRO does not beat ERM in test AP/F1 here. No arbitrary score target is claimed.
All thresholds use validation. Test PR-curve R@P50 is diagnostic; recall at the
validation-selected precision-floor threshold is stored separately. Full-dataset
training, other held-out projects and multiple seeds remain NOT RUN.

Aggregation rejects duplicate, legacy, incompatible and incomplete runs:

```sh
.venv-gpu/bin/python scripts/aggregate_results.py --runs_dir runs/real_baselines/bounded-v1 --out experiments/outputs/baseline_tables --expected_folds 1 --expected_seeds 1 --expected_methods erm cc-dro
```

For larger runs, `notebooks/colab_real_training.ipynb` uses the local source bundle
created by `scripts/package_training_bundle.py`. Upload it and authentic data
manually, inspect actual GPU resources, then run full-population baselines and
validation-only ablations. The notebook was not executed remotely. Local fixes
are uncommitted; cloning the old remote does not reproduce this source version.
No push, merge, PR or automatic cloud upload was performed.
