# layer8 multi-seed sweep (Tables 6 / 7 / 8 with mean ± std)

Purpose: train the ABC-Transformer across **4 intervals × 3 sequence lengths
(k=3,10,30) × 5 seeds = 60 runs**, evaluate each with **teacher-forcing**
(Table 7) and **autoregressive / rollout** (Table 8), and report per-(interval,k)
**mean / std / variance** of MAE & RMSE across seeds. Table 6 = the ABC-Transformer
row only = teacher-forcing, k=3.

Status: **code only — not launched.** Waiting on hyperparameter-optimization
results before training.

## Before running

1. **Update the model config** in `sweep_config.py -> BACKBONE_KWARGS` with the
   chosen hyperparameters once HPO finishes. Current values:
   `d_model=64, nhead=4, num_layers=8, dropout=0.1` (input_size=20, num_continuous=10).
   Seeds are `[42, 0, 1, 2, 3]` (42 first so the first run can be checked against
   the paper's published Table 7/8 numbers).
2. Confirm data paths in `sweep_config.py` (server defaults already set; override
   with `SWEEP_ROOT` to run elsewhere).

## Files

| file | role |
|---|---|
| `sweep_config.py` | single source of truth: paths, intervals, k's, seeds, fixed hyperparams |
| `sweep_train.py`  | train ONE (interval, k, seed); resumable (skips if checkpoint exists) |
| `sweep_eval.py`   | evaluate ONE run: TF + AR via `predict_batched`; writes `eval_metrics.json` |
| `sweep_aggregate.py` | combine all runs -> Table 6/7/8 (mean ± std) CSVs + stdout |
| `run_sweep.sh`    | loop train+eval over the whole matrix, then aggregate |
| `../predict_batched.py` | batched TF/AR passes (same dataset + metric as predict.py) |

`predict.py` exposes the batched passes via `from predict_batched import ...`.
Metrics match `scenario_wise_metrics`: per-step error = mean over the 10 continuous
variables; **micro** = pooled over all steps; **macro** = mean of per-scenario means
(= Table 7/8 "averaged over scenarios").

## How to run (later, on the A100 server)

```bash
cd /data/wonung_data/timeseries_prediction_transformer/src
# smoke-check a single run first (e.g. smallest interval):
INTERVALS="60min" KS="3" SEEDS="42" bash experiments/run_sweep.sh
# then the full sweep in the background:
nohup bash experiments/run_sweep.sh > training_logs_layer8/sweep.out 2>&1 &
```

Per-run outputs land in
`training_logs_layer8/<interval>/seq<k>/seed<seed>/`:
`config_used.yaml`, `csv_logs/version_0/metrics.csv`, `tb_logs/`, `checkpoints/`,
`eval_metrics.json`, `eval_summary_tf.txt`, `eval_summary_ar.txt`.
Aggregated tables land in `training_logs_layer8/` as
`table6_abc_transformer.csv`, `table7_teacher_forcing.csv`,
`table8_autoregressive.csv`, `sweep_results_long.csv`.

## Validation

Run `60min / k=3 / seed=42` first; its teacher-forcing macro MAE/RMSE should land
near the paper's Table 7 k=3 (≈ 0.0069 / 0.0116) and autoregressive near Table 8
k=3 (≈ 0.0379 / 0.0633). If so, the batched pipeline faithfully reproduces the
official evaluation.
