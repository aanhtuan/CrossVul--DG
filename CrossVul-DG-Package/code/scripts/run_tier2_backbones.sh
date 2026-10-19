#!/usr/bin/env bash
# Tier 2 -- backbone generality.
# {unixcoder, graphcodebert} x {erm, dro, cc-dro} x 10 folds x 2 seeds.
# Shows the CC-DRO gain is not specific to CodeBERT without running the full 6-method grid.

set -euo pipefail

METHODS=(erm dro cc-dro)
SEEDS=(0 1)
declare -A BACKBONES=(
  [unixcoder]="microsoft/unixcoder-base"
  [graphcodebert]="microsoft/graphcodebert-base"
)

PROCESSED_DIR=${PROCESSED_DIR:-data/processed}
DATA=${DATA:-${PROCESSED_DIR}/top10.parquet}
SPLITS=${SPLITS:-${PROCESSED_DIR}/splits.json}
CFG=${CFG:-configs/default.yaml}
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

python scripts/aggregate_results.py --runs_dir "${OUT_ROOT}" --out paper_tables/tier2/ \
  --expected_folds "${#PROJECTS[@]}" --expected_seeds "${#SEEDS[@]}" \
  --backbone_filter unixcoder-base graphcodebert-base
