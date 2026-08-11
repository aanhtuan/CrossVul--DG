#!/usr/bin/env bash
# Tier 1 -- main results table.
# CodeBERT x {erm, da-coral, da-dann, irm, dro, cc-dro, pd-dro} x 10 folds x 3 seeds.
# Expects data/processed/{top10,tail}.parquet from prepare_data.py.

set -euo pipefail

PROJECTS=(linux chromium ffmpeg qemu php-src imagemagick openssl tensorflow wireshark systemd)
METHODS=(erm da-coral da-dann irm dro cc-dro pd-dro)
SEEDS=(0 1 2)

DATA=data/processed/top10.parquet
TAIL=data/processed/tail.parquet
CFG=configs/default.yaml
BACKBONE=microsoft/codebert-base
OUT_ROOT=runs

for method in "${METHODS[@]}"; do
  for proj in "${PROJECTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      out="${OUT_ROOT}/codebert/${method}/${proj}/seed${seed}"
      if [[ -f "${out}/results.json" ]]; then
        echo "[skip] ${out}"; continue
      fi
      echo "=== ${method} | held_out=${proj} | seed=${seed} ==="
      extra=()
      [[ "${method}" == "pd-dro" ]] && extra+=("--tail_path" "${TAIL}")
      python scripts/run_lopo.py \
        --config "${CFG}" --method "${method}" --held_out "${proj}" \
        --seed "${seed}" --data_path "${DATA}" --backbone "${BACKBONE}" \
        --output_dir "${out}" "${extra[@]}"
    done
  done
done

python scripts/aggregate_results.py --runs_dir "${OUT_ROOT}" --out paper_tables/
