"""Aggregate the per-accident-type variable_analysis.json files into a summary.

Reads every <out_root>/<cell>/variable_analysis.json (out_root + cell list from
configs/error_mlp_accident.yaml) and writes
    src/experiments/results/variable_analysis_summary.md
Also PRINTS the tables. Cells with no variable_analysis.json are skipped (resumable).

Per cell:
  * top-3 most IMPORTANT input variables (dMAE, permutation, teacher forcing),
  * top-3 HARDEST output variables (AR rollout MAE and p99),
  * worst-20-scenario top error-contributor variables + top-3 cumulative share,
  * the import<->error link (Spearman + verdict).
Plus a CROSS-CELL view: which variables recur as important / hard / worst-driver
across the accident types (count of cells where each variable is a top-3 member).

Usage (from src/):
    NONINTERACTIVE=1 python experiments/aggregate_variable_analysis.py
"""
import os
import sys
import json
from collections import Counter

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import utils

RESULTS_DIR = os.path.join(SRC_DIR, "experiments", "results")
TOP_N = 3


def fnum(x, nd=6):
    if x is None:
        return "-"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def topn(ranked, n=TOP_N):
    return ranked[:n]


def desc_topn(ranked, n=TOP_N, nd=6):
    return ", ".join(f"{r['name']} ({fnum(r['value'], nd)})" for r in topn(ranked, n))


def main():
    cfg = utils.load_config("error_mlp_accident")
    out_root = cfg["out_root"]
    cell_names = list(cfg["cells"].keys())

    cells = {}
    for cell in cell_names:
        p = os.path.join(out_root, cell, "variable_analysis.json")
        if not os.path.exists(p):
            print(f"[aggregate] skip {cell} (no variable_analysis.json)")
            continue
        with open(p) as f:
            cells[cell] = json.load(f)

    if not cells:
        print("[aggregate] no variable_analysis.json found; nothing to do")
        return

    lines = ["# Per-accident-type variable-importance + error-attribution summary", ""]
    lines.append(
        "Frozen backbones, NO retraining, TEST set. "
        "**Importance** = permutation importance in teacher-forcing next-step "
        "(dMAE = increase in next-step MAE when an INPUT channel is shuffled across "
        "test windows; avg of 3 shuffles). **Output error** = per-output MAE/p99 of "
        "the baseline uncorrected AR rollout (beta=0). **Worst-scenario** = top-20 "
        "scenarios by per-scenario mean AR error; share = fraction of the pooled "
        "worst-scenario error carried by each output variable. **Link** = Spearman "
        "rank correlation over the 10 continuous vars (which are both inputs and "
        "outputs) between input importance and per-output AR error / worst-scenario "
        "share.")
    lines.append("")

    # ---- Per-cell headline table ----
    lines.append("## Per-cell headline")
    lines.append("")
    lines.append("| cell | top-3 important inputs (dMAE) | top-3 hardest outputs (MAE) | "
                 "worst-20 top contributor | top-3 cum share | import<->output rho | verdict-short |")
    lines.append("|---|---|---|---|---|---|---|")
    for cell, r in cells.items():
        imp3 = desc_topn(r["importance"]["ranked_by_dMAE_overall"])
        out3 = desc_topn(r["output_error"]["ranked_by_mae"])
        attr = r["worst_scenario_attribution"]
        top_contrib = attr["top_contributor"]
        cum = attr.get(f"top{TOP_N}_cumulative_share")
        link = r["link_importance_error"]
        rho = link["spearman_import_vs_output_error"]
        short = "ALIGNED" if link["all_aligned"] else "DIVERGENT"
        lines.append(
            f"| {cell} | {imp3} | {out3} | {top_contrib} ({fnum(attr['share_per_out'][attr['contributors_ranked'][0]['index']], 3)}) | "
            f"{fnum(cum, 3)} | {fnum(rho, 3)} | {short} |")
    lines.append("")

    # ---- Per-cell detail ----
    for cell, r in cells.items():
        lines.append(f"## {cell}")
        lines.append("")
        lines.append(f"- controls: {', '.join(r['control_cols'])}  "
                     f"(n_scenarios={r['n_scenarios']}, n_test_windows={r['n_test_windows']})")
        lines.append(f"- baseline teacher-forcing MAE: {fnum(r['importance']['baseline_tf_mae'])}")
        lines.append("")
        lines.append("**Top-3 most important INPUT variables (permutation dMAE):**")
        for x in topn(r["importance"]["ranked_by_dMAE_overall"]):
            lines.append(f"  - #{x['rank']} {x['name']}: dMAE={fnum(x['value'])}")
        lines.append("")
        lines.append("**Top-3 hardest OUTPUT variables (AR rollout):**")
        by_mae = {d["name"]: d for d in r["output_error"]["ranked_by_mae"]}
        p99_map = {d["name"]: d["value"] for d in r["output_error"]["ranked_by_p99"]}
        for x in topn(r["output_error"]["ranked_by_mae"]):
            lines.append(f"  - #{x['rank']} {x['name']}: MAE={fnum(x['value'])}, "
                         f"p99={fnum(p99_map.get(x['name']))}")
        lines.append("")
        lines.append("**Hardest OUTPUT variables by p99 (tail):**")
        for x in topn(r["output_error"]["ranked_by_p99"]):
            lines.append(f"  - #{x['rank']} {x['name']}: p99={fnum(x['value'])}")
        lines.append("")
        attr = r["worst_scenario_attribution"]
        lines.append(f"**Worst-{attr['worst_n']} scenarios error attribution** "
                     f"(avg per-scenario mean err={fnum(attr['worst_scenario_mean_avg'])}):")
        for c in topn(attr["contributors_ranked"]):
            lines.append(f"  - {c['name']}: share={fnum(c['share'], 4)}")
        lines.append(f"  - top-{TOP_N} cumulative share: "
                     f"{fnum(attr.get(f'top{TOP_N}_cumulative_share'), 4)}")
        lines.append("")
        link = r["link_importance_error"]
        lines.append("**Link importance <-> error:**")
        lines.append(f"  - most important input: {link['top_important_input']}")
        lines.append(f"  - hardest output: {link['top_hardest_output']} "
                     f"(the important input is #{int(link['top_important_input_output_error_rank'])} hardest output)")
        lines.append(f"  - top worst-scenario driver: {link['top_worst_scenario_driver']} "
                     f"(the important input carries worst-share rank #{int(link['top_important_input_worst_share_rank'])})")
        lines.append(f"  - Spearman(import, output-error) = "
                     f"{fnum(link['spearman_import_vs_output_error'], 3)} "
                     f"(p={fnum(link['spearman_import_vs_output_error_p'], 3)})")
        lines.append(f"  - Spearman(import, worst-share) = "
                     f"{fnum(link['spearman_import_vs_worst_share'], 3)} "
                     f"(p={fnum(link['spearman_import_vs_worst_share_p'], 3)})")
        lines.append(f"  - VERDICT: {link['verdict']}")
        lines.append("")

    # ---- Cross-cell recurrence ----
    imp_counter, out_counter, worst_counter = Counter(), Counter(), Counter()
    for cell, r in cells.items():
        for x in topn(r["importance"]["ranked_by_dMAE_overall"]):
            imp_counter[x["name"]] += 1
        for x in topn(r["output_error"]["ranked_by_mae"]):
            out_counter[x["name"]] += 1
        for c in topn(r["worst_scenario_attribution"]["contributors_ranked"]):
            worst_counter[c["name"]] += 1

    n_cells = len(cells)
    lines.append("## Cross-cell recurrence (how many of the "
                 f"{n_cells} cells put each variable in its top-3)")
    lines.append("")

    def counter_lines(title, counter):
        out = [f"**{title}:**"]
        if not counter:
            out.append("  - (none)")
            return out
        for name, cnt in counter.most_common():
            out.append(f"  - {name}: {cnt}/{n_cells}")
        return out

    lines += counter_lines("Recurrently IMPORTANT inputs (top-3)", imp_counter)
    lines.append("")
    lines += counter_lines("Recurrently HARDEST outputs (top-3 MAE)", out_counter)
    lines.append("")
    lines += counter_lines("Recurrent worst-scenario DRIVERS (top-3 share)", worst_counter)
    lines.append("")

    # Overall link verdict count.
    n_aligned = sum(1 for r in cells.values()
                    if r["link_importance_error"]["all_aligned"])
    lines.append(f"## Overall link verdict: {n_aligned}/{n_cells} cells fully ALIGNED "
                 "(most important input == hardest output == top worst-scenario driver)")
    for cell, r in cells.items():
        link = r["link_importance_error"]
        lines.append(f"- **{cell}**: {link['verdict']}")
    lines.append("")

    md = "\n".join(lines) + "\n"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    md_path = os.path.join(RESULTS_DIR, "variable_analysis_summary.md")
    with open(md_path, "w") as f:
        f.write(md)

    print(md)
    print(f"[aggregate] wrote {md_path}")


if __name__ == "__main__":
    main()
