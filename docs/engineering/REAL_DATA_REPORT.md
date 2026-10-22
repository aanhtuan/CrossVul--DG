# Real MegaVul integration and verification

Date: 2026-10-20 (Asia/Ho_Chi_Minh). Branch: `fix/training-pipeline-audit`.

## VERIFIED — location, preservation and provenance

The new resources were found at `megavul/megavul.json` and its parent `megavul/`
inside this repository. The folder contains only the dataset: no upstream source
checkout, Joern graphs or auxiliary resource tree was found. No third-party code
needed relocation.

- Size: **2,668,796,982 bytes**. JSON array, ASCII-encoded, one long line.
- SHA-256: `eabe8d3325be358d7a93327e3ae4f001725517a0067c4d80198f89904177419f`.
- Canonical path: `data/raw/megavul/megavul.json`.
- Integration: a hardlink on the same device/inode. Original retained unchanged;
  no 2.49 GiB duplicate copy and no destructive move.
- User states this is authentic MegaVul. All records were checked against the
  [official flattened schema](https://github.com/Icyrockton/MegaVul/blob/main/SPECIFICATION.md).
  Independent publisher checksum and release identifier were not supplied;
  schema verification and local file identity are verified, publisher provenance
  is attributed to the user's supplied file.
- Raw data, processed parquet, environments, weights, checkpoints and outputs are
  ignored. Small metadata and manifests remain trackable. No push or merge.

## VERIFIED — full streaming counts

| Population or exclusion | Count |
|---|---:|
| Raw rows | 353,873 |
| Raw vulnerable / non-vulnerable | 17,975 / 335,898 |
| Raw projects | 1,062 |
| Accepted schema rows | 353,415 |
| Rejected unchanged pre/post code | 458 |
| Quarantined rows (union of conflict/cross-project/post-fix rules) | 17,016 |
| Additional same-label, same-project duplicate rows removed | 6,054 |
| Final clean rows | 330,345 |
| Clean vulnerable / non-vulnerable | 15,733 / 314,612 |
| Clean projects | 1,054 |
| Top-10 / tail rows | 167,470 / 162,875 |

Raw positive rate: 5.0795%; clean positive rate: 4.7626%. The only schema
exclusion was `unchanged_before_after`: normalized pre/post code identical.
These 458 positive records were quarantined as ambiguous, not relabelled.
Missing/invalid labels and dates: zero observed. Missing CVE/commit IDs: zero.
CWE list and optional diff types were checked across every record. Null pre-fix
and diff fields in negatives are expected; positive code always comes from
`func_before`, never the repaired `func` fallback.

Quality counters: 146 conflicting-label hash groups; 7,069 cross-project hash
groups; 1,492 rows overlapping known post-fix code. These counters overlap and
must not be summed. Their union quarantines 17,016 rows. There were 7,559 CVE
groups and 7,939 commit groups containing multiple rows; 48 CVE groups span
projects, while no commit group spans projects. Complete per-project raw/clean
counts and all field-type counts are in `data/manifests/dataset_audit.json`.

Functions under 30 characters: 333 accepted rows; over 50,000 characters: one.
These lengths are reported, not filtered for model performance.

## VERIFIED — cleaned top projects

| Project | Clean rows | Positive | Negative | Positive rate |
|---|---:|---:|---:|---:|
| `torvalds/linux` | 82,405 | 3,338 | 79,067 | 4.051% |
| `chromium` | 22,505 | 801 | 21,704 | 3.559% |
| `wireshark` | 20,356 | 668 | 19,688 | 3.282% |
| `gpac` | 9,767 | 390 | 9,377 | 3.993% |
| `android` | 6,968 | 451 | 6,517 | 6.472% |
| `xen-project/xen` | 6,392 | 509 | 5,883 | 7.963% |
| `vim` | 5,707 | 251 | 5,456 | 4.398% |
| `ffmpeg` | 5,314 | 374 | 4,940 | 7.038% |
| `radareorg/radare2` | 4,128 | 190 | 3,938 | 4.603% |
| `tensorflow` | 3,928 | 359 | 3,569 | 9.140% |

## FIXED, TESTED — real-scale export and reproducibility

The first real preprocessing run failed safely at Parquet export: its default
row-group writer required another 128 MiB while the DuckDB 256 MB budget was
already partly occupied. Prior outputs were preserved and no completed marker
was published. The fix sets 2,048-row Parquet groups. A regression checks the
actual Parquet row-group bound.

The full rerun succeeded at the same 256 MB DuckDB limit: **115.05 seconds**, peak
RSS **554,144 KiB** (about 541 MiB). JSON was streamed in 512-row batches, with
one DuckDB thread; the 2.49 GiB input was never loaded as one Python object.
Verified idempotent rerun retained Parquet modification time and output hashes.

Processing protocol:
`megavul-v2-strict-before-global-quarantine-temporal-purge`.
Processing signature: `5d9f3aca706f4294bbe2b8d4fb2264a2b5d7ae96ff2e204f9c40295d49eb7434`.
Exact code/library/date bound and both output hashes are in `splits.json`.

## VERIFIED — leakage and LOPO

All ten full-population metadata folds passed: source project separation,
chronological source train/validation and held-out target/test boundaries,
timestamp ties, and CVE/commit/normalized-code separation. Groups crossing
partitions are purged without consulting label outcomes or model predictions.
Duplicate provenance aliases are retained for these checks.

For preregistered held-out `torvalds/linux`, the unbounded clean partition sizes
are train **75,241**, validation **7,247**, early target **40,535**, late test
**35,577**. A total of 8,870 top-project rows were excluded by fold grouping
safeguards. See `fold_audit.json` for every partition's dates and class counts.

Historical LOPO is not a global forward-time protocol. Some source training
commits postdate early held-out test commits. This pre-existing declared protocol
is reported explicitly and is not represented as future-only evaluation.

Bounded lexical near-duplicate screening examined the first 2,000 chronological
rows (first 256 tokens/function, four SimHash bands, capped buckets/comparisons).
It compared 195 candidates and retained 16 near-duplicate candidate pairs. All
16 pairs are within a project; none spans partitions in the registered Linux
fold and none has both members in the controlled benchmark cohort. This is not
an exhaustive semantic-clone guarantee. No candidate was removed to improve scores.

## VERIFIED — exact token lengths over all 330,345 clean rows

Pinned CodeBERT tokenizer: `microsoft/codebert-base` revision
`3b0952feddeffad0063f274080e3c23d75e7eb39`. Counts include special tokens.

| Statistic | Tokens |
|---|---:|
| Median | 206 |
| 90th percentile | 885 |
| 95th percentile | 1,249 |
| 99th percentile | 2,394 |
| Maximum | 42,440 |
| Mean | 388.351 |

Truncation rates at lengths 128 / 256 / 512 are **67.7355% / 42.3851% / 21.6743%**.
These are measured coverage limitations, not evidence that increasing length
will improve validation or test performance. Long examples remain in the population.
The histogram and per-project counts are in `token_lengths.json`.

## TESTED / NOT RUN

Native GPU suite: **93 passed**; CPU suite: **90 passed, 3 CUDA-only skips**.
Real data was processed and every full metadata fold audited. A pretrained
CodeBERT tiny-fit diagnostic and controlled real ERM/CC-DRO benchmark ran; see
`TRAINING_RESULTS.md`. Full-population training and multi-fold/multi-seed scores
are NOT RUN. Small-cohort counts are never substituted for these full counts.

## Reproduction

```sh
.venv/bin/python scripts/prepare_data.py --raw_dir data/raw/megavul --out_dir data/processed --dedup --batch_size 512 --memory_limit 256MB
.venv/bin/python scripts/audit_data.py --data_path data/processed/top10.parquet
HF_HOME="$PWD/.cache/huggingface" .venv/bin/python scripts/audit_token_lengths.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-gpu/bin/python -m pytest -q
```

Do not delete or overwrite either hardlink. Changing either link's contents
would change the same underlying file; ingestion opens it read-only.
