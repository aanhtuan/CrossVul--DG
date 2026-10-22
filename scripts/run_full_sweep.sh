#!/usr/bin/env bash
# Tier 1 -- main results table.
# CodeBERT x {erm, da-coral, da-dann, irm, dro, cc-dro, pd-dro} x 10 folds x 3 seeds.
# Expects data/processed/{top10,tail}.parquet from prepare_data.py.

set -euo pipefail

METHODS=(erm da-coral da-dann irm dro cc-dro pd-dro)
SEEDS=(0 1 2)

PROCESSED_DIR=${PROCESSED_DIR:-data/processed}
DATA=${DATA:-${PROCESSED_DIR}/top10.parquet}
TAIL=${TAIL:-${PROCESSED_DIR}/tail.parquet}
SPLITS=${SPLITS:-${PROCESSED_DIR}/splits.json}
CFG=${CFG:-configs/default.yaml}
BACKBONE=microsoft/codebert-base
OUT_ROOT=${RUNS_DIR:-runs}

if [[ -n "${PROJECTS:-}" ]]; then
  read -r -a PROJECTS <<< "${PROJECTS}"
elif [[ -f "${SPLITS}" ]]; then
  mapfile -t PROJECTS < <(
    python -c "import json; print(*json.load(open('${SPLITS}', encoding='utf-8'))['top_project_counts'].keys(), sep='\n')"
  )
else
  echo "Missing ${SPLITS}. Run preprocessing first or set PROJECTS=\"proj1 proj2 ...\"." >&2
  exit 1
fi

if [[ ! -f "${DATA}" ]]; then
  echo "Missing ${DATA}. Set PROCESSED_DIR to the folder containing top10.parquet." >&2
  exit 1
fi
if [[ " ${METHODS[*]} " == *" pd-dro "* && ! -f "${TAIL}" ]]; then
  echo "Missing ${TAIL}. pd-dro needs tail.parquet from preprocessing." >&2
  exit 1
fi

for method in "${METHODS[@]}"; do
  for proj in "${PROJECTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      out="${OUT_ROOT}/codebert/${method}/${proj}/seed${seed}"
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

python scripts/aggregate_results.py --runs_dir "${OUT_ROOT}" --out paper_tables/ \
  --expected_folds "${#PROJECTS[@]}" --expected_seeds "${#SEEDS[@]}" \
  --backbone_filter codebert-base --expected_methods "${METHODS[@]}"
