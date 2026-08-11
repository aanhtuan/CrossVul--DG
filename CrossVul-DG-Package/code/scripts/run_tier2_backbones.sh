#!/usr/bin/env bash
# Tier 2 -- backbone generality.
# {unixcoder, graphcodebert} x {erm, dro, cc-dro} x 10 folds x 2 seeds.
# Shows the CC-DRO gain is not specific to CodeBERT without running the full 6-method grid.

set -euo pipefail

PROJECTS=(linux chromium ffmpeg qemu php-src imagemagick openssl tensorflow wireshark systemd)
METHODS=(erm dro cc-dro)
SEEDS=(0 1)
declare -A BACKBONES=(
  [unixcoder]="microsoft/unixcoder-base"
  [graphcodebert]="microsoft/graphcodebert-base"
)

DATA=data/processed/top10.parquet
CFG=configs/default.yaml
OUT_ROOT=runs

for bb_key in "${!BACKBONES[@]}"; do
  bb="${BACKBONES[$bb_key]}"
  for method in "${METHODS[@]}"; do
    for proj in "${PROJECTS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        out="${OUT_ROOT}/${bb_key}/${method}/${proj}/seed${seed}"
        if [[ -f "${out}/results.json" ]]; then echo "[skip] ${out}"; continue; fi
        echo "=== ${bb_key} | ${method} | held_out=${proj} | seed=${seed} ==="
        python scripts/run_lopo.py \
          --config "${CFG}" --method "${method}" --held_out "${proj}" \
          --seed "${seed}" --data_path "${DATA}" --backbone "${bb}" \
          --output_dir "${out}"
      done
    done
  done
done

python scripts/aggregate_results.py --runs_dir "${OUT_ROOT}" --out paper_tables/
