"""Aggregate per-accident-type ErrorMLP eval jsons into a summary table.

Reads every <out_root>/<cell>/error_mlp_eval.json (out_root + cell list from
configs/error_mlp_accident.yaml) and writes:
    src/experiments/results/error_mlp_accident_summary.csv
    src/experiments/results/error_mlp_accident_summary.md
Per cell -> baseline_micro, beta_star, corrected_micro, rel_reduction%, macro,
win_fraction. Also prints the table. Cells with no eval json yet are skipped
(the run is resumable), so this can be re-run at any point.

Usage (from src/):
    NONINTERACTIVE=1 python experiments/aggregate_error_mlp_accident.py
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
    "cell", "num_controls", "in_dim", "beta_star",
    "baseline_micro", "corrected_micro", "rel_reduction_pct",
    "baseline_macro", "corrected_macro", "win_fraction", "positive_result",
]


def main():
    cfg = utils.load_config("error_mlp_accident")
    out_root = cfg["out_root"]
    cell_names = list(cfg["cells"].keys())

    rows = []
    for cell in cell_names:
        p = os.path.join(out_root, cell, "error_mlp_eval.json")
        if not os.path.exists(p):
            print(f"[aggregate] skip {cell} (no eval json)")
            continue
        with open(p) as f:
            r = json.load(f)
        star = r["beta_star_test_metrics"]
        base = r["baseline_ar"]
        rows.append({
            "cell": cell,
            "num_controls": r.get("num_controls"),
            "in_dim": r.get("in_dim"),
            "beta_star": r["beta_star"],
            "baseline_micro": base["micro_mae"],
            "corrected_micro": star["micro_mae"],
            "rel_reduction_pct": r.get("rel_reduction_micro_pct"),
            "baseline_macro": base["macro_mae"],
            "corrected_macro": star["macro_mae"],
            "win_fraction": r["per_scenario_win_fraction"],
            "positive_result": r["positive_result"],
        })

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "error_mlp_accident_summary.csv")
    md_path = os.path.join(RESULTS_DIR, "error_mlp_accident_summary.md")

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    def fnum(x, nd=8):
        return f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x)

    lines = ["# Per-accident-type ErrorMLP AR-correction summary", ""]
    lines.append("| cell | ctrls | in_dim | beta* | baseline_micro | corrected_micro | rel_red% | baseline_macro | corrected_macro | win_frac | positive |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        lines.append(
            f"| {row['cell']} | {row['num_controls']} | {row['in_dim']} | {row['beta_star']} | "
            f"{fnum(row['baseline_micro'])} | {fnum(row['corrected_micro'])} | "
            f"{fnum(row['rel_reduction_pct'], 2)} | {fnum(row['baseline_macro'])} | "
            f"{fnum(row['corrected_macro'])} | {fnum(row['win_fraction'], 4)} | "
            f"{row['positive_result']} |")
    md = "\n".join(lines) + "\n"
    with open(md_path, "w") as f:
        f.write(md)

    print(md)
    print(f"[aggregate] wrote {csv_path}")
    print(f"[aggregate] wrote {md_path}")


if __name__ == "__main__":
    main()
