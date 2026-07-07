#!/usr/bin/env bash
# Isolated dropout=0.1 ablation: 60min, k=3, 5 seeds. Trains + evals (TF & AR)
# into a SEPARATE dir (training_logs_drop01_ab) via SWEEP_LOG_ROOT, with
# dropout forced to 0.1 via SWEEP_DROPOUT. Does NOT touch the main dropout=0
# sweep (training_logs_layer8). Runs all 5 seeds concurrently (shares the GPU).
set -u
PY=/data/wonung_data/miniconda3/envs/transformer_env/bin/python
export NONINTERACTIVE=1 CUDA_VISIBLE_DEVICES=0
export SWEEP_DROPOUT=0.1
export SWEEP_LOG_ROOT=training_logs_drop01_ab
SRC=/data/wonung_data/timeseries_prediction_transformer/src
cd "$SRC" || exit 1
LOG="$SRC/training_logs_drop01_ab/_console"
mkdir -p "$LOG"
echo "[ablation] dropout=0.1 60min k=3 seeds[42 0 1 2 3] -> $SWEEP_LOG_ROOT  start $(date)"
run_one(){
  local s=$1 tag=60min_seq3_seed$1
  if "$PY" experiments/sweep_train.py --interval 60min --k 3 --seed "$s" --device 0 --num-workers 2 > "$LOG/train_$tag.log" 2>&1; then
    "$PY" experiments/sweep_eval.py --interval 60min --k 3 --seed "$s" --device cuda:0 > "$LOG/eval_$tag.log" 2>&1 \
      && echo "[done $tag]" || echo "[EVAL-FAIL $tag]"
  else echo "[TRAIN-FAIL $tag]"; fi
}
for s in 42 0 1 2 3; do run_one "$s" & done
wait
echo "[ablation] training+eval done $(date); aggregating drop01 dir"
"$PY" experiments/sweep_aggregate.py > "$LOG/aggregate.log" 2>&1 && echo "[ablation] aggregate ok" || echo "[ablation] aggregate FAIL"
echo "[ablation] all done $(date)"
