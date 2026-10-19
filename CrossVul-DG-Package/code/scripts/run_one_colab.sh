#!/usr/bin/env bash
# Run one Colab/Kaggle-friendly LOPO job:
# one method x one held-out project x seed 0.
#
# Required:
#   PROCESSED_DIR=/path/to/processed
#
# Optional:
#   METHOD=cc-dro
#   HELD_OUT=linux
#   RUNS_DIR=/path/to/runs
#   CONFIG=configs/colab.yaml
#   BACKBONE=microsoft/codebert-base

set -euo pipefail

METHOD=${METHOD:-cc-dro}
HELD_OUT=${HELD_OUT:-linux}
SEED=${SEED:-0}
CONFIG=${CONFIG:-configs/colab.yaml}
BACKBONE=${BACKBONE:-microsoft/codebert-base}
RUNS_DIR=${RUNS_DIR:-runs}

if [[ -z "${PROCESSED_DIR:-}" ]]; then
  echo "Missing PROCESSED_DIR. Example:" >&2
  echo "  PROCESSED_DIR=/content/drive/MyDrive/CrossVul-DG/MegaVul/processed METHOD=cc-dro HELD_OUT=linux bash scripts/run_one_colab.sh" >&2
  exit 1
fi

DATA="${PROCESSED_DIR}/top10.parquet"
TAIL="${PROCESSED_DIR}/tail.parquet"
SPLITS="${PROCESSED_DIR}/splits.json"
OUT="${RUNS_DIR}/codebert/${METHOD}/${HELD_OUT}/seed${SEED}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing config: ${CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${DATA}" ]]; then
  echo "Missing data file: ${DATA}" >&2
  exit 1
fi
if [[ ! -f "${SPLITS}" ]]; then
  echo "Missing splits file: ${SPLITS}" >&2
  exit 1
fi
if [[ "${METHOD}" == "pd-dro" && ! -f "${TAIL}" ]]; then
  echo "Missing tail file for pd-dro: ${TAIL}" >&2
  exit 1
fi

echo "Processed projects:"
python -c "import json; data=json.load(open('${SPLITS}', encoding='utf-8')); print(data.get('top_project_counts', {}))"

extra=()
if [[ "${METHOD}" == "pd-dro" ]]; then
  extra+=("--tail_path" "${TAIL}")
fi

echo "=== method=${METHOD} | held_out=${HELD_OUT} | seed=${SEED} ==="
echo "output_dir=${OUT}"

python scripts/run_lopo.py \
  --config "${CONFIG}" \
  --method "${METHOD}" \
  --held_out "${HELD_OUT}" \
  --seed "${SEED}" \
  --data_path "${DATA}" \
  --backbone "${BACKBONE}" \
  --output_dir "${OUT}" \
  "${extra[@]}"

echo "Done."
echo "Results: ${OUT}/results.json"
echo "Checkpoint: ${OUT}/checkpoint.pt"
