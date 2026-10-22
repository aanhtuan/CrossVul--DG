> Historical initial audit, before the user supplied the real dataset. Current data, native GPU evidence and measured results are in REAL_DATA_REPORT.md, TRAINING_RESULTS.md and OPTIMIZATION_REPORT.md. Initial blockers/counts below describe that earlier phase.

# CrossVul-DG local engineering audit

Date: 2026-10-20, Asia/Ho_Chi_Minh. Baseline commit:
`9ae55d80752245cfc60df577aa065f3d8c1fd742`.
Branch: `fix/training-pipeline-audit`. No commit, push, PR or merge performed.
The starting working tree was clean.

**Outcome:** implementation and CPU fixture validation completed. Authentic
MegaVul acquisition, real-model training and native CUDA verification are
**BLOCKED**. This is not evidence of improved real-world vulnerability detection.

## 1. Original problems and confirmed root causes

| Priority | Problem | Root cause and evidence | Status |
|---|---|---|---|
| P0 | Wrong vulnerable code could be labelled positive | Parser fell back from missing `func_before` to post-fix `func`. Reproduced on baseline code. | FIXED, TESTED |
| P0 | Invalid labels accepted | Numeric `2`, negative numbers and other nonzero values became `1`. Baseline probe accepted `2`. | FIXED, TESTED |
| P0 | Project identities collided | Owner prefixes were stripped: `owner1/repo` and `owner2/repo` became the same domain. Reproduced. | FIXED, TESTED |
| P0 | Leakage could not be audited | CVE/commit/code identities were discarded; same-code conflicts retained an arbitrary label; row-level time cuts could split related records. | FIXED, TESTED on fixtures |
| P0 | Stale checkpoints/results could be reused | Checkpoints checked only method; sweep scripts skipped any existing result filename. Config/data/backbone/source/software identities were absent. | FIXED, TESTED |
| P1 | ERM was not standard pooled ERM | Sources were visited as sequential project loaders. | FIXED, TESTED |
| P1 | CC-DRO initialization biased dual weights | New cells were initialized after earlier cells had already received exponential updates. Equal risks produced weights `0.2689414 / 0.7310585`, instead of equal weights. | FIXED, TESTED |
| P1 | Dual updates overflowed or lost history | Direct exponentials overflowed; probability-space storage loses historical log mass after underflow. Baseline large-risk update produced nonfinite q. | FIXED, TESTED |
| P1 | Multi-project graphs accumulated in VRAM | Every environment forward graph remained alive until the coupled objective backward pass. | FIXED, CPU gradient-equivalence TESTED |
| P1 | PD-DRO could mix CPU model/CUDA inputs | Embedding code moved inputs but did not ensure model device agreement. | FIXED, CPU/spy TESTED; native CUDA BLOCKED |
| P1 | Analysis omitted or mixed experiments | Notebook omitted CC-DRO and parsed `args.config` as a mapping; aggregation overwrote duplicate fold/seed runs and mixed incompatible configs. | FIXED, TESTED |

Baseline probes were run against source extracted directly from the baseline
Git commit into a temporary directory, not against reconstructed snippets.
Detailed evidence: `runs/audit/baseline_defects.json`.

## 2. Structure and files changed

Active code moved with `git mv` from `CrossVul-DG-Package/code/` to root-level
`configs/`, `scripts/`, `tests/`, `data/`, `notebooks/`, `figures/`, and
`src/crossvul_dg/`. Flat package modules are retained to avoid unnecessary
fragmentation. Compatibility launchers preserve five old Python script paths.
Historical proposals, papers and ZIP archives are unchanged.

Main implementation files:

- `src/crossvul_dg/preprocessing.py`: streaming schema validation, quarantine,
  provenance aliases, bounded DuckDB processing, hashes, staged output publication
  and verified idempotence. Source/library/options/date-bound signatures are
  recorded; `--as_of` makes timestamp acceptance reproducible.
- `src/crossvul_dg/data.py`: strict parquet loading, resource preflight, project
  identity preservation, time-tie-safe partitions, CVE/commit/hash purging,
  pooled ERM data and explicit sampling budgets.
- `src/crossvul_dg/losses.py`, `trainers.py`: stable log-dual updates, exact
  stochastic replay, bounded graph retention, optimization diagnostics and
  checkpoint schema v3 with algorithm/RNG/sampler state.
- `src/crossvul_dg/models.py`, `pseudo_domains.py`: pinned encoder revision,
  explicit device movement, deterministic bounded tail embeddings/clustering.
- `src/crossvul_dg/evaluate.py`: input validation, confusion matrix, prevalence,
  project means/worst scores and explicit diagnostic operating-point definitions.
- `src/crossvul_dg/config.py`, `provenance.py`, `resources.py`, `utils.py`:
  strict config overrides, software/data/source identities, resource guards,
  deterministic seeding and atomic standard JSON with undefined values as null.
- `scripts/run_lopo.py`: full protocol enforcement, validation-only selection,
  disjoint label-stripped DA target, separately labelled auxiliary-supervision
  PD-DRO, latest/best artifacts and completed-result identity checks.
- `scripts/aggregate_results.py`, `notebooks/analysis.ipynb`: compatibility
  cohorts, duplicate/incomplete rejection, CC-DRO charts and corrected config parsing.
- `scripts/prepare_data.py`, `audit_data.py`, `run_sanity_check.py`,
  `run_real_diagnostics.py`, `run_focused_ablations.py`: controlled entry points.
- `pyproject.toml`, requirements files, root README, configs and tests updated.
  Seventeen generated bytecode entries were removed from Git tracking; local
  files were preserved and caches are ignored. Dataset manifests stay trackable.

## 3. Mathematical and evaluation protocol

**VERIFIED:** CC-DRO retains
`L = sum_g q_g R_g + lambda_inv * Var_p(R_p,positive)`.
Within-cell BCE is unweighted by default; optional positive weighting is explicit.

The no-grad first pass obtains cell risks, then performs exactly one dual update.
The gradient pass replays the same batches, RNG state and buffers at unchanged
parameters, using the exact risk derivatives of the coupled objective. An
optimizer step occurs only after all environment contributions are accumulated.
CPU tests compare gradients against the full simultaneous graph with dropout.
Log-space dual history survives underflow and is checkpointed directly.

ERM pools and shuffles every source row once per epoch. Environment methods
use a uniform environment-step budget approximating one pooled pass in total
sample draws. Small projects may repeat and large projects may be partially
visited; these are consequences of the declared environment objective. Per-epoch
actual draws, unique samples, repeated draws and class exposure are recorded.
CC-DRO additionally logs classification loss, raw/weighted invariance penalty,
penalty/classification ratio, risk-coefficient norms, q min/max/entropy and
pre-clipping gradient norm. Lambda dominance on real data remains unmeasured.

Protocol `megavul-v2-strict-before-global-quarantine-temporal-purge` intentionally
changes the legacy population. Positives require pre-fix code; negatives use
non-vulnerable `func`. Ambiguous records and conflicting/cross-project exact-code
hash groups are excluded with counts. Overlaps with known positive post-fix code
are quarantined. Duplicate provenance aliases are retained, not silently lost.

LOPO sources and the held-out project are separate. Source train/validation are
chronological with rescaled train:validation ratios; timestamp ties are indivisible.
CVE/commit/code groups spanning partitions are purged. The earlier held-out portion
is DA target, the later portion is shared test data for every method. PD-DRO tail
rows must be in separate projects, precede the source-training cutoff, and have
no protected-partition metadata/code overlap. Additional labelled tail supervision
is reported separately.

Historical cross-project LOPO is **not** a forward-time test: some source training
commits may postdate held-out test commits. The runner reports this explicitly.
Missing CVE/commit metadata limits metadata-based leakage coverage.

Checkpoint selection and F1/precision-floor threshold selection use validation
only. AP is Average Precision; F1 is vulnerable-class binary F1. Test PR-curve
R@P50 is an oracle diagnostic, not a deployable threshold. Test recall/precision
at the validation-selected precision-floor threshold have separate fields.
That precision floor is not guaranteed on test. Single-class selection folds
fail explicitly; undefined subgroup AUCs are marked and serialized as null.

## 4. Tests and execution evidence

**TESTED:** final pytest run: **85 passed, 1 skipped**, 5.45 seconds reported by
pytest. `/usr/bin/time` wall time 7.63 seconds, maximum RSS 598,212 KiB
(approximately 584 MiB), exit status 0.

Coverage includes schema/date/hash/label mapping, provenance aliases, conflicting
and cross-project code quarantine, atomic failure preservation, raw array/JSONL
through parquet through LOPO, leakage checks, balanced/pooled sampling, loss and
metric definitions, stable dual updates and underflow recovery, exact replay
gradients, all seven trainers, real CLI orchestration with a mock backbone,
identity rejection, checkpoint resume/early-stop/latest-versus-best semantics,
cohort aggregation/table generation and actual CC-DRO chart rendering.

- CPU: all seven methods executed through the LOPO runner using temporary
  synthetic fixtures and a tiny mocked backbone. No pretrained model downloaded.
- A separate char-tokenized tiny Transformer sanity command executed all seven
  trainer paths. Synthetic scores are deliberately not presented as research metrics.
- Resume equivalence tested for ERM, CC-DRO and DA-DANN, including optimizer,
  scheduler, algorithm state, RNG and sampling progression via matching histories
  and final states.
- `compileall`, editable package import, both YAML configs and Git whitespace
  checks passed. Dependency check: all 50 installed packages compatible.
- The single skipped test is native CUDA pseudo-domain embedding. CPU tests
  explicitly check model movement precedes input/device use.
- Resource preflight deliberately rejected pretrained-model execution with about
  2.4 GiB available RAM, below the documented 6 GiB minimum.

Repair-loop failures were fixed and re-tested: strict leakage checks exposed
invalid old test fixtures; floating-point ratio residue required an exact zero
source test ratio; a pseudo-domain signature/import issue and a metadata-iterator
name regression were caught by expanded integration tests; PyArrow's upper bound
was aligned with the actual verified environment. None remains in the final run.

Local detailed logs are under `runs/audit/`. Lightweight evidence is in
`docs/engineering/VALIDATION.json`. These checks verify the stated CPU fixture
scope; they do not prove native CUDA or real-data behavior.

## 5. Data counts and acquisition

**BLOCKED:** no released MegaVul file was found in the searched local workspace,
Downloads/Documents or home data/dataset locations. The official release folder
redirected to `onedrive.live.com` and returned **HTTP 403**, body:
`The request is blocked.` Provider timestamp: `2026-10-19T16:51:47Z`.

Sources: [official repository](https://github.com/Icyrockton/MegaVul),
[specification](https://github.com/Icyrockton/MegaVul/blob/main/SPECIFICATION.md),
[released dataset folder](https://1drv.ms/f/s!AtzrzuojQf5sgeISZ9zN_4owVnUn9g).
The source repository was not substituted for release data.

| Population | Rows | Positive / negative |
|---|---:|---|
| Authentic raw MegaVul | Unknown; unavailable | Unknown |
| Authentic processed top/tail | Not generated | Unknown |
| Dedicated preprocessing fixture | 160, four projects | 80 / 80 |
| Fixture top-3 / tail | 120 / 40 | 60/60 and 20/20 |
| Separate Transformer sanity fixture | 480 | Synthetic; not a research population |

`data/manifests/acquisition.json` records the source, blocker, expected schema and
null release/acquisition date/file size/SHA-256 fields. The current real-data
`dataset_audit.json` is explicitly BLOCKED with unknown counts. No synthetic
parquet has been placed in the repository's production data directory. Successful
preparation of a supplied authentic file will write actual hashes/counts and
output manifests; verify and update acquisition provenance first.

The upstream repository is GPL-3.0; individual extracted source functions may
retain original project licensing. Released-file terms could not be verified
through the inaccessible provider.

## 6. Original versus corrected metrics

**No real-data metrics were measured before or after these changes.** There is
no supported F1/AP improvement claim. Baseline-versus-corrected measurements
above concern deterministic correctness invariants, not generalization scores.

## 7. Remaining limitations and next research priorities

1. **BLOCKED, P0 verification:** obtain the authentic full flattened release and
   record its release/date/size/SHA-256; run preparation and per-fold audits.
   The simple release lacks required dates unless supplemented from authentic data.
2. **BLOCKED, P1 verification:** free sufficient RAM and configure matching CUDA
   PyTorch if desired. Hardware: 15 GiB RAM, initial 21/31 GiB swap used, RTX 3050
   4 GiB. CPU fixtures do not establish 4 GiB VRAM feasibility or CUDA replay equivalence.
3. **BLOCKED:** tiny-real overfit diagnostic, one real ERM/CC-DRO LOPO fold,
   measured runtime/VRAM, and real ablations were not run. Runnable commands exist.
4. **HYPOTHESIS:** lambda_inv, long-function truncation and oversampling may affect
   real generalization. Test matched source splits/seeds, validation only. No
   setting is asserted optimal and no score target is promised.
5. **VERIFIED limitation:** near-duplicate screening is a bounded lexical SimHash
   diagnostic (default first 2,000 rows, first 256 tokens per function, candidate
   bucket cap and comparison cap). It is not an exhaustive semantic-clone audit.
6. **VERIFIED limitation:** interruption restarts streaming input, while verified
   completed outputs are reused. Raw JSON byte-offset resume is not implemented.
7. **VERIFIED limitation:** inference/sampling/gradient correctness is covered on
   CPU fixtures. The full released dataset, alternative backbones and long-running
   training remain outside executed verification. Paired significance is exploratory
   and unadjusted for multiple comparisons; insufficient/incomplete cohorts are excluded.

No failing actionable correctness case remains among the executed CPU checks.
The above boundaries prevent declaring the entire research evaluation complete.

## 8. Reproduction

The existing `.venv` uses bundled Python 3.12.14. See root README for environment
recreation with `requirements-cpu.lock` and for controlled real-data commands.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q
.venv/bin/python scripts/run_sanity_check.py
.venv/bin/python scripts/check_default_config.py --config configs/default.yaml
.venv/bin/python scripts/check_default_config.py --config configs/colab.yaml
uv pip check --python .venv/bin/python
.venv/bin/python -m compileall -q src scripts tests
.venv/bin/python scripts/prepare_data.py --raw_dir data/raw/megavul --out_dir data/processed --dedup
.venv/bin/python scripts/audit_data.py --data_path data/processed/top10.parquet
.venv/bin/python scripts/run_real_diagnostics.py --config configs/default.yaml
.venv/bin/python scripts/run_focused_ablations.py --held_out OWNER/PROJECT --config configs/default.yaml
```

The last four commands require authentic data; the real-model diagnostic also
requires resource headroom. Ablation generation only prints commands unless
`--execute` is supplied. Do not launch full sweeps before the earlier stages pass.
