#!/usr/bin/env bash
# Per-accident-type ErrorMLP: train then eval, looping the 5 cells. Resumable
# (skips a cell whose eval json already exists). Override the cell list with the
# CELLS env var (space-separated). Runs from src/ with the transformer_env python.
#
# Usage (from repo root or anywhere):
#   bash src/experiments/run_error_mlp_accident.sh
#   CELLS="SBO" bash src/experiments/run_error_mlp_accident.sh
#   CELLS="LLOCA_CSP LLOCA_ECSBS TLOFW_CSP TLOFW_ECSBS" bash src/experiments/run_error_mlp_accident.sh
set -euo pipefail

PY="${PY:-/data/wonung_data/miniconda3/envs/transformer_env/bin/python}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-/home/wonung/wonung_data/error_mlp_accident_out}"
CELLS="${CELLS:-SBO LLOCA_CSP LLOCA_ECSBS TLOFW_CSP TLOFW_ECSBS}"

cd "$SRC_DIR"
export NONINTERACTIVE=1

for cell in $CELLS; do
  eval_json="$OUT_ROOT/$cell/error_mlp_eval.json"
  if [[ -f "$eval_json" ]]; then
    echo "[skip] $cell (eval json exists: $eval_json)"
    continue
  fi
  echo "==================== $cell : TRAIN ===================="
  "$PY" experiments/train_error_mlp_acc.py --cell "$cell"
  echo "==================== $cell : EVAL ===================="
  "$PY" experiments/eval_error_mlp_acc.py --cell "$cell"
  echo "[done] $cell -> $eval_json"
done

echo "==================== AGGREGATE ===================="
"$PY" experiments/aggregate_error_mlp_accident.py
