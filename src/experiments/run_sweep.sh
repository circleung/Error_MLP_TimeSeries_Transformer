#!/usr/bin/env bash
# Orchestrate the layer8 sweep: train + evaluate every (interval, k, seed),
# then aggregate. Resumable (already-trained runs are skipped). DOES NOT run
# until you launch it. Run from the src/ directory.
#
#   cd /data/wonung_data/timeseries_prediction_transformer/src
#   bash experiments/run_sweep.sh            # full 4 x 3 x 5 = 60 runs
#   INTERVALS="60min" KS="3" SEEDS="42" bash experiments/run_sweep.sh   # one run
#
# Env overrides:
#   PY        python interpreter (default: transformer_env python)
#   INTERVALS space-separated subset (default: 60min 30min 15min 5min)
#   KS        space-separated seq lengths (default: 3 10 30)
#   SEEDS     space-separated seeds (default: 42 0 1 2 3)
#   DEVICE    cuda device index (default: 0)
#   EPOCHS    override max epochs (default: config value, 100)
set -u

PY="${PY:-/data/wonung_data/miniconda3/envs/transformer_env/bin/python}"
INTERVALS="${INTERVALS:-60min 30min 15min 5min}"
KS="${KS:-3 10 30}"
SEEDS="${SEEDS:-42 0 1 2 3}"
DEVICE="${DEVICE:-0}"
EPOCHS_ARG=""
[ -n "${EPOCHS:-}" ] && EPOCHS_ARG="--epochs ${EPOCHS}"

export NONINTERACTIVE=1
export CUDA_VISIBLE_DEVICES="${DEVICE}"

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SRC_DIR" || exit 1
LOG_DIR="$SRC_DIR/training_logs_layer8/_console_logs"
mkdir -p "$LOG_DIR"
echo "[run_sweep] src=$SRC_DIR py=$PY"
echo "[run_sweep] intervals='$INTERVALS' ks='$KS' seeds='$SEEDS' device=$DEVICE"

for interval in $INTERVALS; do
  for k in $KS; do
    for seed in $SEEDS; do
      tag="${interval}_seq${k}_seed${seed}"
      echo "==================== TRAIN $tag ===================="
      "$PY" experiments/sweep_train.py --interval "$interval" --k "$k" --seed "$seed" \
            --device 0 $EPOCHS_ARG 2>&1 | tee "$LOG_DIR/train_${tag}.log"
      echo "==================== EVAL  $tag ===================="
      "$PY" experiments/sweep_eval.py --interval "$interval" --k "$k" --seed "$seed" \
            --device cuda:0 2>&1 | tee "$LOG_DIR/eval_${tag}.log"
    done
  done
done

echo "==================== AGGREGATE ===================="
"$PY" experiments/sweep_aggregate.py 2>&1 | tee "$LOG_DIR/aggregate.log"
echo "[run_sweep] done."
