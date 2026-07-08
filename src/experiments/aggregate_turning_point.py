"""Aggregate the per-accident-type turning_point_analysis.json files into a summary.

Reads every <out_root>/<cell>/turning_point_analysis.json (out_root + cell list from
configs/error_mlp_accident.yaml) and writes
    src/experiments/results/turning_point_summary.md
Also PRINTS the table. Cells with no turning_point_analysis.json are skipped (resumable).

Per cell it reports the three hypothesis tests (pooled over (variable, step) events; the
error unit is the per-step abs error, pooled = mean over the 10 vars == compute_micro_macro
per-step MAE; SCALED space):
  H1: turning-point error LIFT -- TP base rate, mean error at TP vs non-TP (ratio), and the
      SHARE of the p99 error tail that falls at turning points vs the base rate (lift).
  H2: directional hit-rate overall vs at turning points (expected LOWER at TP).
  H3: MISS-vs-HIT downstream cumulative-error RATIO at H=10 (a directional MISS at a turning
      point vs a correct-direction turning point -> does the error explode downstream?).
Plus the top turning-point-failure variables per cell, and a CROSS-CELL verdict on H1/H2/H3.

Usage (from src/):
    NONINTERACTIVE=1 python experiments/aggregate_turning_point.py
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
H_KEY = "10"          # headline downstream horizon for H3.
TOP_N = 3


def fnum(x, nd=4):
    if x is None:
        return "-"
    try:
        f = float(x)
        if f != f:      # NaN
            return "nan"
        return f"{f:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def topn_names(ranked, n=TOP_N, nd=3):
    return ", ".join(f"{r['name']} ({fnum(r['value'], nd)})" for r in ranked[:n])


def h1_verdict(pH1):
    """H1 supported iff turning-point steps carry disproportionate tail error (lift>1) AND
    mean error at TP exceeds non-TP."""
    lift = pH1["lift_tail_at_tp"]
    ratio = pH1["err_ratio_tp_over_non"]
    ok = (lift is not None and lift == lift and lift > 1.0
          and ratio is not None and ratio == ratio and ratio > 1.0)
    return "SUPPORTED" if ok else "NOT-supported"


def h2_verdict(pH2):
    """H2 supported iff directional hit-rate is LOWER at turning points (drop > 0)."""
    drop = pH2["dir_hit_rate_drop"]
    return "SUPPORTED" if (drop is not None and drop == drop and drop > 0.0) else "NOT-supported"


def h3_verdict(pH3H):
    """H3 supported iff a directional MISS at a TP has larger downstream error than a HIT
    (ratio_miss_over_hit > 1)."""
    r = pH3H["ratio_miss_over_hit"]
    return "SUPPORTED" if (r is not None and r == r and r > 1.0) else "NOT-supported"


def main():
    cfg = utils.load_config("error_mlp_accident")
    out_root = cfg["out_root"]
    cell_names = list(cfg["cells"].keys())

    cells = {}
    for cell in cell_names:
        p = os.path.join(out_root, cell, "turning_point_analysis.json")
        if not os.path.exists(p):
            print(f"[aggregate] skip {cell} (no turning_point_analysis.json)")
            continue
        with open(p) as f:
            cells[cell] = json.load(f)

    if not cells:
        print("[aggregate] no turning_point_analysis.json found; nothing to do")
        return

    lines = ["# Turning-point / directional-error AR failure-mode summary", ""]
    lines.append(
        "Frozen backbones, NO retraining, TEST set, reusing the beta=0 (uncorrected) AR "
        "rollout. Hypothesis (user's insight): catastrophic AR error blowups happen at "
        "**turning points** -- steps where a true trajectory changes direction -- and a "
        "**directional miss** at a turning point feeds the wrong value back so the rollout "
        "diverges. Per continuous variable: Delta_true[t]=y_true[t]-y_true[t-1], "
        "Delta_pred[t]=yhat[t]-y_true[t-1]; a **turning point** is a sign flip of Delta_true "
        "(both steps above a per-var deadband eps = p"
        + fnum(next(iter(cells.values()))['config']['eps_percentile'], 0)
        + " of |Delta_true|); a **directional hit** is sign(Delta_pred)==sign(Delta_true). "
        "Error unit = per-step abs error, pooled = mean over the 10 vars (== "
        "compute_micro_macro per-step MAE); SCALED space. Pooled stats are over (variable, "
        "step) turning-point events (NOT 'any-var per step', which saturates). "
        "**H1**: do turning points carry disproportionate error (lift). **H2**: is "
        "directional accuracy worse at turning points. **H3** (key): does a MISS at a "
        "turning point cause a larger DOWNSTREAM error blowup than a HIT.")
    lines.append("")

    # ---- Per-cell headline table ----
    lines.append("## Per-cell headline")
    lines.append("")
    lines.append(
        "| cell | scenarios | TP base rate | H1: mean-err TP/non (ratio) | H1: p99-tail share@TP (lift) | "
        "H2: dir-hit overall / at-TP (drop) | H3: MISS/HIT downstream cum-err @H=10 (ratio) | "
        "H1 | H2 | H3 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for cell, r in cells.items():
        c = r["config"]
        pH1 = r["pooled"]["H1"]; pH2 = r["pooled"]["H2"]
        pH3 = r["pooled"]["H3"][H_KEY]
        scen = (f"{c['n_scenarios_used']}/{c['n_scenarios_total']}"
                + (" (first-by-id)" if c["subsample"] else " (ALL)"))
        h1_cell = (f"{fnum(pH1['mean_err_at_tp'])} / {fnum(pH1['mean_err_non_tp'])} "
                   f"({fnum(pH1['err_ratio_tp_over_non'], 2)}x)")
        h1_tail = (f"{fnum(pH1['share_tail_at_tp'], 3)} (base {fnum(pH1['turning_point_base_rate'], 3)}; "
                   f"lift {fnum(pH1['lift_tail_at_tp'], 2)}x)")
        h2_cell = (f"{fnum(pH2['dir_hit_rate_overall'], 3)} / {fnum(pH2['dir_hit_rate_at_tp'], 3)} "
                   f"({fnum(pH2['dir_hit_rate_drop'], 3)})")
        h3_cell = (f"{fnum(pH3['miss_mean_cum_err'])} / {fnum(pH3['hit_mean_cum_err'])} "
                   f"({fnum(pH3['ratio_miss_over_hit'], 2)}x)")
        lines.append(
            f"| {cell} | {scen} | {fnum(pH1['turning_point_base_rate'], 3)} | {h1_cell} | "
            f"{h1_tail} | {h2_cell} | {h3_cell} | {h1_verdict(pH1)} | {h2_verdict(pH2)} | "
            f"{h3_verdict(pH3)} |")
    lines.append("")

    # ---- Per-cell detail ----
    for cell, r in cells.items():
        c = r["config"]
        pH1 = r["pooled"]["H1"]; pH2 = r["pooled"]["H2"]; pH3all = r["pooled"]["H3"]
        rk = r["rankings"]
        lines.append(f"## {cell}")
        lines.append("")
        lines.append(f"- controls: {', '.join(r['control_cols'])}")
        lines.append(f"- subsample: {c['subsample_rule']} "
                     f"({c['n_scenarios_used']}/{c['n_scenarios_total']} scenarios), "
                     f"eps=p{fnum(c['eps_percentile'], 0)} of |Delta_true| (per var), "
                     f"curvature TP=top-decile |2nd diff|, near=+/-{c['near_window']}, "
                     f"tail=p{fnum(c['tail_percentile'], 0)}, "
                     f"rollout approx {fnum(c['approx_rollout_seconds'], 1)}s")
        lines.append("")
        lines.append(f"**H1 (error concentration at turning points):** turning-point base rate "
                     f"= {fnum(pH1['turning_point_base_rate'], 4)} of (var,step) events. "
                     f"Mean per-step error at TP = {fnum(pH1['mean_err_at_tp'])} vs non-TP "
                     f"= {fnum(pH1['mean_err_non_tp'])} "
                     f"(ratio {fnum(pH1['err_ratio_tp_over_non'], 2)}x; near-TP "
                     f"{fnum(pH1['err_ratio_near_over_non'], 2)}x). Turning-point steps carry "
                     f"{fnum(pH1['share_total_err_at_tp'], 3)} of the total error and "
                     f"{fnum(pH1['share_tail_at_tp'], 3)} of the p99 tail "
                     f"(lift {fnum(pH1['lift_tail_at_tp'], 2)}x; near-TP tail share "
                     f"{fnum(pH1['share_tail_near_tp'], 3)}, lift {fnum(pH1['lift_tail_near_tp'], 2)}x). "
                     f"-> **{h1_verdict(pH1)}**")
        lines.append("")
        lines.append(f"**H2 (directional accuracy at turning points):** dir hit-rate overall "
                     f"= {fnum(pH2['dir_hit_rate_overall'], 4)} vs at TP "
                     f"= {fnum(pH2['dir_hit_rate_at_tp'], 4)} "
                     f"(drop {fnum(pH2['dir_hit_rate_drop'], 4)}; n_dir={pH2['n_dir_steps']}, "
                     f"n_dir_TP={pH2['n_dir_tp_steps']}). -> **{h2_verdict(pH2)}**")
        lines.append("")
        lines.append("**H3 (directional MISS at a turning point -> downstream error blowup):**")
        lines.append("")
        lines.append("| H | MISS mean cum-err | HIT mean cum-err | ratio MISS/HIT | n_miss | n_hit |")
        lines.append("|---|---|---|---|---|---|")
        for H in c["horizons"]:
            hh = pH3all[str(H)]
            lines.append(f"| {H} | {fnum(hh['miss_mean_cum_err'])} | {fnum(hh['hit_mean_cum_err'])} | "
                         f"{fnum(hh['ratio_miss_over_hit'], 2)}x | {hh['n_miss_tp']} | {hh['n_hit_tp']} |")
        lines.append(f"\n-> **{h3_verdict(pH3all[H_KEY])}** at H=10 "
                     f"(ratio {fnum(pH3all[H_KEY]['ratio_miss_over_hit'], 2)}x)")
        # miss vs hit downstream trajectory at H=10 (pooled error at offsets 0..10)
        mt = pH3all[H_KEY].get("miss_err_trajectory", [])
        ht = pH3all[H_KEY].get("hit_err_trajectory", [])
        if mt and ht:
            lines.append("")
            lines.append("MISS downstream error trajectory (offset 0..10): "
                         + ", ".join(fnum(v) for v in mt))
            lines.append("")
            lines.append("HIT  downstream error trajectory (offset 0..10): "
                         + ", ".join(fnum(v) for v in ht))
        lines.append("")
        lines.append("**Top turning-point-failure variables:**")
        lines.append(f"  - by p99-tail lift at TP: {topn_names(rk['by_tail_lift_at_tp'])}")
        lines.append(f"  - by error ratio TP/non-TP: {topn_names(rk['by_err_ratio_tp_over_non'], nd=2)}")
        lines.append(f"  - by directional hit-rate DROP at TP: {topn_names(rk['by_dir_hit_rate_drop_at_tp'], nd=3)}")
        lines.append(f"  - by H3 MISS/HIT downstream ratio (H=10): {topn_names(rk['by_h3_miss_over_hit_ratio_H10'], nd=2)}")
        lines.append("")

    # ---- Cross-cell verdict ----
    n_cells = len(cells)
    h1_ok = sum(1 for r in cells.values() if h1_verdict(r["pooled"]["H1"]) == "SUPPORTED")
    h2_ok = sum(1 for r in cells.values() if h2_verdict(r["pooled"]["H2"]) == "SUPPORTED")
    h3_ok = sum(1 for r in cells.values()
                if h3_verdict(r["pooled"]["H3"][H_KEY]) == "SUPPORTED")
    # cross-cell recurrence of top turning-point-failure variables (by H3 ratio and dir drop).
    h3_counter, drop_counter = Counter(), Counter()
    for r in cells.values():
        for x in r["rankings"]["by_h3_miss_over_hit_ratio_H10"][:TOP_N]:
            h3_counter[x["name"]] += 1
        for x in r["rankings"]["by_dir_hit_rate_drop_at_tp"][:TOP_N]:
            drop_counter[x["name"]] += 1

    lines.append("## Cross-cell verdict")
    lines.append("")
    lines.append(f"- **H1** (errors concentrate at turning points): SUPPORTED in {h1_ok}/{n_cells} cells.")
    lines.append(f"- **H2** (directional accuracy worse at turning points): SUPPORTED in {h2_ok}/{n_cells} cells.")
    lines.append(f"- **H3** (a directional MISS at a turning point -> larger downstream error blowup than a HIT): "
                 f"SUPPORTED in {h3_ok}/{n_cells} cells.")
    lines.append("")
    lines.append("Per-cell H3 ratio (MISS/HIT downstream cum-err @H=10):")
    for cell, r in cells.items():
        lines.append(f"  - {cell}: {fnum(r['pooled']['H3'][H_KEY]['ratio_miss_over_hit'], 2)}x "
                     f"({h3_verdict(r['pooled']['H3'][H_KEY])})")
    lines.append("")
    lines.append("Cross-cell recurrence of top turning-point-failure variables:")

    def counter_lines(title, counter):
        out = [f"**{title}:**"]
        if not counter:
            out.append("  - (none)")
            return out
        for name, cnt in counter.most_common():
            out.append(f"  - {name}: {cnt}/{n_cells}")
        return out

    lines += counter_lines("Most-recurrent by H3 MISS/HIT ratio (top-3)", h3_counter)
    lines.append("")
    lines += counter_lines("Most-recurrent by directional hit-rate DROP at TP (top-3)", drop_counter)
    lines.append("")

    # overall one-line verdict
    def maj(n):
        return "SUPPORTED" if n > n_cells / 2 else ("MIXED" if n > 0 else "REFUTED")
    lines.append(f"**Overall:** H1 {maj(h1_ok)} ({h1_ok}/{n_cells}); "
                 f"H2 {maj(h2_ok)} ({h2_ok}/{n_cells}); H3 {maj(h3_ok)} ({h3_ok}/{n_cells}). "
                 "The user's hypothesis is that directional misses at turning points cause the "
                 "blowups; it is best judged by H3 (does a MISS blow up downstream more than a "
                 "HIT) together with H1 (are turning points where the error lives).")
    lines.append("")

    md = "\n".join(lines) + "\n"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    md_path = os.path.join(RESULTS_DIR, "turning_point_summary.md")
    with open(md_path, "w") as f:
        f.write(md)

    print(md)
    print(f"[aggregate] wrote {md_path}")


if __name__ == "__main__":
    main()
