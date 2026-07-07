"""Aggregate per-accident-type GATED tail-analysis jsons into a summary table.

Reads every <out_root>/<cell>/tail_analysis.json (out_root + cell list from
configs/error_mlp_accident.yaml) and writes:
    src/experiments/results/tail_analysis_summary.csv
    src/experiments/results/tail_analysis_summary.md
Per cell -> baseline (mean, p99, max); best PREDICTED-gate operating point under
the strict mean-non-regress constraint (beta*, q*, mean, p99, max, %p99-reduction);
ORACLE p99 ceiling. Also prints the table. Cells with no tail json are skipped
(resumable).

Usage (from src/):
    NONINTERACTIVE=1 python experiments/aggregate_tail_acc.py
"""
import os
import sys
import csv
import json

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import utils

RESULTS_DIR = os.path.join(SRC_DIR, "experiments", "results")

COLUMNS = [
    "cell", "num_controls", "in_dim",
    "baseline_mean", "baseline_p99", "baseline_max",
    "op_beta", "op_q", "op_mean", "op_p99", "op_max", "op_p99_reduction_pct",
    "op_cuts_tail", "oracle_best_q", "oracle_p99", "oracle_p99_reduction_pct",
    "verdict",
]


def fnum(x, nd=6):
    if x is None:
        return "-"
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x)


def main():
    cfg = utils.load_config("error_mlp_accident")
    out_root = cfg["out_root"]
    cell_names = list(cfg["cells"].keys())

    rows = []
    for cell in cell_names:
        p = os.path.join(out_root, cell, "tail_analysis.json")
        if not os.path.exists(p):
            print(f"[aggregate] skip {cell} (no tail_analysis.json)")
            continue
        with open(p) as f:
            r = json.load(f)
        base = r["baseline"]
        op = r.get("operating_point_strict")
        oracle = r["oracle_p99_ceiling"]
        rows.append({
            "cell": cell,
            "num_controls": r.get("num_controls"),
            "in_dim": r.get("in_dim"),
            "baseline_mean": base["mean"],
            "baseline_p99": base["p99"],
            "baseline_max": base["max"],
            "op_beta": op["beta"] if op else None,
            "op_q": op["q"] if op else None,
            "op_mean": op["mean"] if op else None,
            "op_p99": op["p99"] if op else None,
            "op_max": op["max"] if op else None,
            "op_p99_reduction_pct": op["p99_reduction_pct"] if op else None,
            "op_cuts_tail": op["cuts_tail"] if op else False,
            "oracle_best_q": oracle["q"],
            "oracle_p99": oracle["p99"],
            "oracle_p99_reduction_pct": oracle["p99_reduction_pct"],
            "verdict": r["verdict"],
        })

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "tail_analysis_summary.csv")
    md_path = os.path.join(RESULTS_DIR, "tail_analysis_summary.md")

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    lines = ["# Per-accident-type GATED tail-correction summary", ""]
    lines.append("Goal: cut the TAIL (p99/max of the per-step error dist) on TEST via a "
                 "SELECTIVE (gated) ErrorMLP correction, keeping the mean (micro_mae) "
                 "non-regressed. baseline = uncorrected AR. op = best PREDICTED-gate "
                 "operating point s.t. mean <= baseline_mean (strict). oracle = "
                 "gate-on-true-error p99 ceiling (detector ceiling).")
    lines.append("")
    lines.append("| cell | baseline_mean | baseline_p99 | baseline_max | op(beta*,q*) | "
                 "op_mean | op_p99 | op_max | op %p99-red | cuts_tail | oracle_p99 | oracle %p99-red |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        op_desc = (f"beta={row['op_beta']},q={row['op_q']}"
                   if row["op_beta"] is not None else "NONE")
        lines.append(
            f"| {row['cell']} | {fnum(row['baseline_mean'])} | {fnum(row['baseline_p99'])} | "
            f"{fnum(row['baseline_max'])} | {op_desc} | {fnum(row['op_mean'])} | "
            f"{fnum(row['op_p99'])} | {fnum(row['op_max'])} | "
            f"{fnum(row['op_p99_reduction_pct'], 2)} | {row['op_cuts_tail']} | "
            f"{fnum(row['oracle_p99'])} | {fnum(row['oracle_p99_reduction_pct'], 2)} |")
    lines.append("")
    lines.append("## Per-cell verdict")
    for row in rows:
        lines.append(f"- **{row['cell']}**: {row['verdict']}")
    md = "\n".join(lines) + "\n"
    with open(md_path, "w") as f:
        f.write(md)

    print(md)
    print(f"[aggregate] wrote {csv_path}")
    print(f"[aggregate] wrote {md_path}")


if __name__ == "__main__":
    main()
