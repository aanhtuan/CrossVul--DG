# Data

This directory holds:

- `megavul_raw/` — MegaVul as released by the upstream maintainers
  (not checked in to keep the repository small).
- `processed/` — Outputs of `scripts/prepare_data.py`:
    - `top10.parquet` — functions from the top-10 projects by sample
      count, with columns `[func, label, project, commit_date]`.
    - `tail.parquet` — functions from the remaining projects (used by
      the `pd-dro` pseudo-domain variant).
    - `splits.json` — chronological cut-off timestamps for the
      8 : 1 : 1 split applied to `top10`.
- `synthetic/` — Tiny placeholder so the repo runs end-to-end without
  any download; used by `scripts/run_sanity_check.py`.

## Obtaining MegaVul

MegaVul is released publicly. As of 2024:

```bash
cd data/
git clone https://github.com/Icyrockton/MegaVul.git megavul_raw
```

The release ships per-CVE JSON files plus a metadata table. The
`_read_megavul` adapter in `scripts/prepare_data.py` is intentionally
permissive — adjust it to your snapshot's field names if the columns
above (`func`, `label`/`is_vul`, `project`, `commit_date`) are named
differently.

## Re-running preprocessing

```bash
python scripts/prepare_data.py \
    --raw_dir data/megavul_raw \
    --out_dir data/processed \
    --top_k 10 \
    --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 \
    --dedup
```

Expected wall-clock: ~3 min on a workstation.
