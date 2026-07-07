#!/usr/bin/env bash
# Dropout AR ablation: k=3, intervals {60min,30min}, dropout {0.1,0.2}, seeds
# {42,0,1,2,3}, early-stop patience=5. Isolated from the main dropout=0 sweep via
# per-job SWEEP_DROPOUT/SWEEP_LOG_ROOT and global SWEEP_PATIENCE=5. Train+eval
# (TF & AR on test) per run; aggregate each dropout dir at the end.
set -u
PY=/data/wonung_data/miniconda3/envs/transformer_env/bin/python
export NONINTERACTIVE=1 CUDA_VISIBLE_DEVICES=0 SWEEP_PATIENCE=5
JOBS=${JOBS:-6}
SRC=/data/wonung_data/timeseries_prediction_transformer/src
cd "$SRC" || exit 1
LOG="$SRC/training_logs_dropout_ar_ablation_console"
mkdir -p "$LOG"
export PY LOG
run_one(){
  local dropout=$1 root=$2 interval=$3 s=$4
  local tag=${interval}_drop${dropout}_seed${s}
  echo "[start $(date +%H:%M:%S)] $tag (patience=5)"
  if SWEEP_DROPOUT=$dropout SWEEP_LOG_ROOT=$root "$PY" experiments/sweep_train.py \
        --interval $interval --k 3 --seed $s --device 0 --num-workers 2 > "$LOG/train_$tag.log" 2>&1; then
    SWEEP_DROPOUT=$dropout SWEEP_LOG_ROOT=$root "$PY" experiments/sweep_eval.py \
        --interval $interval --k 3 --seed $s --device cuda:0 > "$LOG/eval_$tag.log" 2>&1 \
      && echo "[done $tag]" || echo "[EVAL-FAIL $tag]"
  else echo "[TRAIN-FAIL $tag]"; fi
}
export -f run_one
JL="$LOG/_jobs.txt"; : > "$JL"
for dropout in 0.1 0.2; do
  root=training_logs_drop$(echo $dropout | tr -d '.')_ar
  for interval in 60min 30min; do
    for s in 42 0 1 2 3; do echo "$dropout $root $interval $s" >> "$JL"; done
  done
done
echo "[ablation] $(wc -l < "$JL") jobs, JOBS=$JOBS, patience=5, dropout{0.1,0.2} x {60min,30min} k=3, seeds{42,0,1,2,3}  start $(date)"
xargs -P "$JOBS" -L1 bash -c 'run_one "$@"' _ < "$JL"
echo "[ablation] train+eval done $(date); aggregating"
for dropout in 0.1 0.2; do
  root=training_logs_drop$(echo $dropout | tr -d '.')_ar
  SWEEP_LOG_ROOT=$root "$PY" experiments/sweep_aggregate.py > "$LOG/aggregate_${root}.log" 2>&1 \
    && echo "[ablation] aggregated $root" || echo "[ablation] aggregate FAIL $root"
done
echo "[ablation] all done $(date)"
