"""Aggregate eval_metrics.json across seeds -> per-(interval, k) mean/std/variance,
laid out like the paper's Table 7 (teacher-forcing) and Table 8 (autoregressive).

For each (interval, k, mode) it computes across the available seeds:
    mean, std (sample, ddof=1), variance (ddof=1)  of MAE and RMSE,
for both macro (= Table 7/8 "averaged over scenarios") and micro aggregation.

Outputs (into <LOG_ROOT>):
    sweep_results_long.csv   one row per (interval, k, mode, agg, metric)
    table7_teacher_forcing.csv   wide: interval x (k -> MAE/RMSE mean±std), macro
    table8_autoregressive.csv    same, AR
    table6_abc_transformer.csv   TF k=3 only (the ABC-Transformer row)
and prints them.

Usage (from src/):
    python experiments/sweep_aggregate.py
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
import experiments.sweep_config as C

MODES = [("teacher_forcing", "TF"), ("autoregressive", "AR")]


def collect():
    rows = []
    for interval in C.INTERVALS:
        for k in C.SEQ_LENS:
            for seed in C.SEEDS:
                p = os.path.join(C.run_dir(interval, k, seed), "eval_metrics.json")
                if not os.path.exists(p):
                    continue
                with open(p) as f:
                    d = json.load(f)
                for mode, _ in MODES:
                    m = d.get(mode, {})
                    rows.append(dict(
                        interval=interval, k=k, seed=seed, mode=mode,
                        micro_mae=m.get("micro_mae"), micro_rmse=m.get("micro_rmse"),
                        macro_mae=m.get("macro_mae"), macro_rmse=m.get("macro_rmse"),
                    ))
    return pd.DataFrame(rows)


def summarize(df):
    out = []
    for interval in C.INTERVALS:
        for k in C.SEQ_LENS:
            for mode, _ in MODES:
                sub = df[(df["interval"] == interval) & (df["k"] == k) & (df["mode"] == mode)]
                if sub.empty:
                    continue
                seeds = sorted(sub.seed.tolist())
                for agg in ("macro", "micro"):
                    for metric in ("mae", "rmse"):
                        vals = sub[f"{agg}_{metric}"].dropna().to_numpy(float)
                        n = vals.size
                        out.append(dict(
                            interval=interval, k=k, mode=mode, agg=agg,
                            metric=metric.upper(), n=n,
                            seeds=",".join(map(str, seeds)),
                            mean=float(np.mean(vals)) if n else float("nan"),
                            std=float(np.std(vals, ddof=1)) if n > 1 else float("nan"),
                            var=float(np.var(vals, ddof=1)) if n > 1 else float("nan"),
                            min=float(np.min(vals)) if n else float("nan"),
                            max=float(np.max(vals)) if n else float("nan"),
                        ))
    return pd.DataFrame(out)


def _cell(long_df, interval, k, mode, metric, agg="macro"):
    r = long_df[(long_df["interval"] == interval) & (long_df["k"] == k)
                & (long_df["mode"] == mode) & (long_df["agg"] == agg)
                & (long_df["metric"] == metric)]
    if r.empty or not np.isfinite(r["mean"].iloc[0]):
        return ""
    mean, std = r["mean"].iloc[0], r["std"].iloc[0]
    return f"{mean:.4f} ± {std:.4f}" if np.isfinite(std) else f"{mean:.4f}"


def table_by_k(long_df, mode, agg="macro"):
    """Table 7/8 layout: rows=interval, cols = k{3,10,30} x {MAE,RMSE} mean±std."""
    rows = []
    for interval in C.INTERVALS:
        row = {"Interval": interval}
        for k in C.SEQ_LENS:
            for metric in ("MAE", "RMSE"):
                row[f"k={k} {metric}"] = _cell(long_df, interval, k, mode, metric, agg)
        rows.append(row)
    return pd.DataFrame(rows)


def table_abc(long_df, agg="macro"):
    """Table 6 ABC-Transformer row: TF, k=3, per interval (MAE & RMSE mean±std)."""
    k = 3
    rows = []
    for metric in ("MAE", "RMSE"):
        row = {"Metric": metric}
        for interval in C.INTERVALS:
            row[interval] = _cell(long_df, interval, k, "teacher_forcing", metric, agg)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(C.ROOT, "src", C.LOG_ROOT_NAME))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = collect()
    if df.empty:
        print("No eval_metrics.json found yet. Train + eval first.")
        return
    long_df = summarize(df)
    long_df.to_csv(os.path.join(args.out_dir, "sweep_results_long.csv"), index=False)

    t7 = table_by_k(long_df, "teacher_forcing", "macro")
    t8 = table_by_k(long_df, "autoregressive", "macro")
    t6 = table_abc(long_df, "macro")
    t7.to_csv(os.path.join(args.out_dir, "table7_teacher_forcing.csv"), index=False)
    t8.to_csv(os.path.join(args.out_dir, "table8_autoregressive.csv"), index=False)
    t6.to_csv(os.path.join(args.out_dir, "table6_abc_transformer.csv"), index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n=== Table 6 (ABC-Transformer, teacher-forcing, k=3) mean ± std [macro] ===")
    print(t6.to_string(index=False))
    print("\n=== Table 7 (Teacher-forcing) mean ± std [macro, over scenarios] ===")
    print(t7.to_string(index=False))
    print("\n=== Table 8 (Autoregressive / rollout) mean ± std [macro] ===")
    print(t8.to_string(index=False))
    print(f"\nSaved CSVs into: {args.out_dir}")


if __name__ == "__main__":
    main()
