#!/usr/bin/env bash
# Parallel layer8 sweep, processed ONE INTERVAL AT A TIME (barrier between
# intervals): all (k,seed) jobs of an interval run+eval through a pool of JOBS
# workers; the next interval starts only after the current one fully finishes.
# Order: 60min -> 30min -> 15min -> 5min. Resumable (skips runs with a checkpoint).
#
#   cd /data/wonung_data/timeseries_prediction_transformer/src
#   JOBS=16 NUM_WORKERS=2 bash experiments/run_sweep_parallel.sh
#
# Env overrides: PY INTERVALS KS SEEDS JOBS DEVICE NUM_WORKERS
set -u
PY="${PY:-/data/wonung_data/miniconda3/envs/transformer_env/bin/python}"
INTERVALS="${INTERVALS:-60min 30min 15min 5min}"
KS="${KS:-3 10 30}"
SEEDS="${SEEDS:-42 0 1 2 3}"
JOBS="${JOBS:-16}"
DEVICE="${DEVICE:-0}"
NUM_WORKERS="${NUM_WORKERS:-2}"
export NONINTERACTIVE=1
export CUDA_VISIBLE_DEVICES="${DEVICE}"

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SRC_DIR" || exit 1
LOG_DIR="$SRC_DIR/training_logs_layer8/_console_logs"
mkdir -p "$LOG_DIR"
export PY LOG_DIR SRC_DIR NUM_WORKERS

run_one() {
  local interval="$1" k="$2" seed="$3"
  local tag="${interval}_seq${k}_seed${seed}"
  echo "[start $(date +%H:%M:%S)] $tag"
  if "$PY" experiments/sweep_train.py --interval "$interval" --k "$k" --seed "$seed" \
        --device 0 --num-workers "$NUM_WORKERS" > "$LOG_DIR/train_${tag}.log" 2>&1; then
    "$PY" experiments/sweep_eval.py --interval "$interval" --k "$k" --seed "$seed" --device cuda:0 \
        > "$LOG_DIR/eval_${tag}.log" 2>&1 \
      && echo "[done  $(date +%H:%M:%S)] $tag" \
      || echo "[EVAL-FAIL $(date +%H:%M:%S)] $tag (see eval_${tag}.log)"
  else
    echo "[TRAIN-FAIL $(date +%H:%M:%S)] $tag (see train_${tag}.log)"
  fi
}
export -f run_one

echo "[parallel] intervals='$INTERVALS' jobs=$JOBS workers/job=$NUM_WORKERS device=$DEVICE start $(date)"
for interval in $INTERVALS; do
  echo "########## INTERVAL ${interval} START $(date) ##########"
  JOBLIST="$LOG_DIR/_jobs_${interval}.txt"
  : > "$JOBLIST"
  for k in $KS; do for seed in $SEEDS; do echo "$interval $k $seed" >> "$JOBLIST"; done; done
  echo "[parallel] ${interval}: $(wc -l < "$JOBLIST") jobs, ${JOBS} concurrent"
  xargs -P "$JOBS" -L 1 bash -c 'run_one "$@"' _ < "$JOBLIST"
  echo "########## INTERVAL ${interval} DONE  $(date) ##########"
  echo "-------- partial aggregate after ${interval} --------"
  "$PY" experiments/sweep_aggregate.py > "$LOG_DIR/aggregate_after_${interval}.log" 2>&1 \
    && echo "[parallel] partial aggregate ok (${interval})" || echo "[parallel] partial aggregate skipped/failed (${interval})"
done

echo "==================== FINAL AGGREGATE $(date) ===================="
"$PY" experiments/sweep_aggregate.py > "$LOG_DIR/aggregate.log" 2>&1 \
  && echo "[parallel] aggregate done" || echo "[parallel] AGGREGATE-FAIL (see aggregate.log)"
echo "[parallel] all done $(date)"
