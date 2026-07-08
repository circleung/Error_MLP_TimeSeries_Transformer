"""Aggregate the per-accident-type ar_permutation_importance.json files into a summary.

Reads every <out_root>/<cell>/ar_permutation_importance.json (out_root + cell list from
configs/error_mlp_accident.yaml) and writes
    src/experiments/results/ar_vs_tf_importance_summary.md
Also PRINTS the table. Cells with no ar_permutation_importance.json are skipped (resumable).

Per cell:
  * top-3 AR-rollout important inputs vs top-3 TEACHER-FORCING important inputs,
  * Spearman(TF, AR) over ALL input channels and over the 10 continuous only,
  * top-3 set overlap and the biggest per-channel rank movers,
  * whether the importance<->output-error DIVERGENCE persists under AR importance
    (AR-import<->output-error Spearman vs the committed TF one).
Plus a CROSS-CELL view: which variables recur in the AR top-3 across accident types,
and how many cells stay DIVERGENT under AR.

Usage (from src/):
    NONINTERACTIVE=1 python experiments/aggregate_ar_importance.py
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
        p = os.path.join(out_root, cell, "ar_permutation_importance.json")
        if not os.path.exists(p):
            print(f"[aggregate] skip {cell} (no ar_permutation_importance.json)")
            continue
        with open(p) as f:
            cells[cell] = json.load(f)

    if not cells:
        print("[aggregate] no ar_permutation_importance.json found; nothing to do")
        return

    lines = ["# AR-rollout vs teacher-forcing permutation-importance summary", ""]
    lines.append(
        "Frozen backbones, NO retraining, TEST set. **AR importance** = permutation "
        "importance in AR ROLLOUT: an INPUT channel is shuffled across the rollout "
        "scenarios and *held shuffled through the whole lockstep rollout*, so the "
        "corruption propagates through the model's own feedback (continuous channels are "
        "the fed-back predictions; controls are the injected known-future). dMAE_AR = "
        "increase in the rollout MAE (mean over scenario x step x 10 outputs of "
        "|pred-true|) when a channel is corrupted, averaged over the fixed shuffle "
        "seeds. **TF importance** (committed, variable_analysis.json) = the same "
        "permutation but a SINGLE teacher-forcing forward (no feedback). The question: "
        "does feedback + error compounding change WHICH variables matter, and does the "
        "TF importance<->output-error DIVERGENCE survive under AR importance?")
    lines.append("")

    # ---- Per-cell headline table ----
    lines.append("## Per-cell headline")
    lines.append("")
    lines.append("| cell | subsample | baseline AR MAE | top-3 AR important | top-3 TF important | "
                 "Spearman(TF,AR) all | Spearman(TF,AR) cont | top-3 overlap | "
                 "AR import<->output rho | TF import<->output rho | divergence under AR? |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for cell, r in cells.items():
        cmp = r["tf_vs_ar"]
        cfgc = r["config"]
        ar3 = desc_topn(r["importance"]["ranked_by_dMAE_overall_AR"])
        tf3 = ", ".join(cmp["tf_top3"])
        sub = (f"{cfgc['n_scenarios_used']}/{cfgc['n_scenarios_total']}"
               + (" (first-by-id)" if cfgc["subsample"] else " (ALL)"))
        ar_rho = cmp["ar_link"]["spearman_ar_import_vs_output_error"]
        tf_rho = cmp["tf_reference"]["spearman_tf_import_vs_output_error"]
        diverg = "YES" if not cmp["ar_link"]["ar_top_import_is_top_output"] else "no"
        lines.append(
            f"| {cell} | {sub} | {fnum(r['importance']['baseline_rollout_mae'])} | "
            f"{ar3} | {tf3} | {fnum(cmp['spearman_tf_ar_all_channels'], 3)} | "
            f"{fnum(cmp['spearman_tf_ar_continuous'], 3)} | "
            f"{cmp['n_top3_overlap']}/3 {cmp['top3_overlap']} | "
            f"{fnum(ar_rho, 3)} | {fnum(tf_rho, 3)} | {diverg} |")
    lines.append("")

    # ---- Per-cell detail ----
    for cell, r in cells.items():
        cmp = r["tf_vs_ar"]
        cfgc = r["config"]
        lines.append(f"## {cell}")
        lines.append("")
        lines.append(f"- controls: {', '.join(r['control_cols'])}")
        lines.append(f"- subsample: {cfgc['subsample_rule']} "
                     f"({cfgc['n_scenarios_used']}/{cfgc['n_scenarios_total']} scenarios), "
                     f"maxL={cfgc['maxL_rollout']}, n_shuffles={cfgc['n_shuffles']} "
                     f"seeds={cfgc['perm_seeds']}, approx GPU={fnum(cfgc['approx_gpu_seconds'], 1)}s")
        lines.append(f"- baseline AR rollout MAE (subsample): "
                     f"{fnum(r['importance']['baseline_rollout_mae'])}")
        lines.append("")
        lines.append("**Top-3 AR-rollout important INPUT variables (dMAE_AR):**")
        for x in topn(r["importance"]["ranked_by_dMAE_overall_AR"]):
            lines.append(f"  - #{x['rank']} {x['name']}: dMAE_AR={fnum(x['value'])}")
        lines.append("")
        lines.append("**Top-3 teacher-forcing important INPUT variables (committed dMAE):**")
        for i, n in enumerate(cmp["tf_top3"]):
            lines.append(f"  - #{i+1} {n}")
        lines.append("")
        lines.append(f"**Spearman(TF, AR)** over ALL {r['input_size']} input channels = "
                     f"{fnum(cmp['spearman_tf_ar_all_channels'], 3)} "
                     f"(p={fnum(cmp['spearman_tf_ar_all_channels_p'], 3)}); "
                     f"over the 10 continuous only = "
                     f"{fnum(cmp['spearman_tf_ar_continuous'], 3)} "
                     f"(p={fnum(cmp['spearman_tf_ar_continuous_p'], 3)})")
        lines.append(f"- top-3 overlap: {cmp['n_top3_overlap']}/3 {cmp['top3_overlap']}")
        lines.append("")
        lines.append("**Biggest rank movers (TF_rank - AR_rank; positive = climbed under AR):**")
        for m in cmp["biggest_movers"]:
            tag = "control" if m["is_control"] else "continuous"
            lines.append(f"  - {m['name']} ({tag}): TF #{m['tf_rank']} -> AR #{m['ar_rank']} "
                         f"(shift {m['rank_shift_tf_minus_ar']:+d})")
        lines.append("")
        arl = cmp["ar_link"]
        tfr = cmp["tf_reference"]
        lines.append("**Importance <-> output-error link under AR:**")
        lines.append(f"  - most important AR input: {arl['top_important_input_ar']} "
                     f"(it is the #{arl['top_important_input_ar_output_error_rank']} hardest output)")
        lines.append(f"  - hardest output: {arl['top_hardest_output']}")
        lines.append(f"  - Spearman(AR-import, output-error) = "
                     f"{fnum(arl['spearman_ar_import_vs_output_error'], 3)} "
                     f"(p={fnum(arl['spearman_ar_import_vs_output_error_p'], 3)})")
        lines.append(f"  - committed Spearman(TF-import, output-error) = "
                     f"{fnum(tfr['spearman_tf_import_vs_output_error'], 3)} "
                     f"(TF verdict all_aligned={tfr['tf_all_aligned']})")
        lines.append(f"  - VERDICT: {arl['verdict']}")
        lines.append("")

    # ---- Cross-cell recurrence ----
    ar_counter, tf_counter = Counter(), Counter()
    for cell, r in cells.items():
        for x in topn(r["importance"]["ranked_by_dMAE_overall_AR"]):
            ar_counter[x["name"]] += 1
        for n in r["tf_vs_ar"]["tf_top3"]:
            tf_counter[n] += 1

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

    lines += counter_lines("Recurrently AR-important inputs (top-3)", ar_counter)
    lines.append("")
    lines += counter_lines("Recurrently TF-important inputs (top-3)", tf_counter)
    lines.append("")

    # ---- Overall AR-vs-TF verdict ----
    rhos_all = [r["tf_vs_ar"]["spearman_tf_ar_all_channels"] for r in cells.values()]
    rhos_cont = [r["tf_vs_ar"]["spearman_tf_ar_continuous"] for r in cells.values()]
    n_diverg = sum(1 for r in cells.values()
                   if not r["tf_vs_ar"]["ar_link"]["ar_top_import_is_top_output"])
    mean_all = sum(rhos_all) / len(rhos_all)
    mean_cont = sum(rhos_cont) / len(rhos_cont)
    lines.append("## Overall verdict")
    lines.append("")
    lines.append(f"- mean Spearman(TF, AR) over all input channels = {fnum(mean_all, 3)} "
                 f"(per cell: {', '.join(fnum(x, 3) for x in rhos_all)})")
    lines.append(f"- mean Spearman(TF, AR) over the 10 continuous = {fnum(mean_cont, 3)} "
                 f"(per cell: {', '.join(fnum(x, 3) for x in rhos_cont)})")
    lines.append(f"- importance<->output-error DIVERGENCE persists under AR in "
                 f"{n_diverg}/{n_cells} cells "
                 "(most important AR input != hardest output)")
    lines.append("")
    for cell, r in cells.items():
        lines.append(f"- **{cell}**: {r['tf_vs_ar']['ar_link']['verdict']}")
    lines.append("")

    md = "\n".join(lines) + "\n"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    md_path = os.path.join(RESULTS_DIR, "ar_vs_tf_importance_summary.md")
    with open(md_path, "w") as f:
        f.write(md)

    print(md)
    print(f"[aggregate] wrote {md_path}")


if __name__ == "__main__":
    main()
