"""Aggregate per-cell variable_gating.json into an efficiency summary table.

Reads every <out_root>/<cell>/variable_gating.json (out_root + cell list from
configs/error_mlp_accident.yaml) and writes:
    src/experiments/results/variable_gating_summary.md
Per cell -> baseline p99; STEP-gate strict OP (p99, mean, intervention%);
per-(step,variable) DYNAMIC-gate strict OP (p99, mean, intervention%); per-variable
FIXED-set strict OP (p99, mean); the p99 delta and intervention-rate delta between
step-gate and per-cell (variable) gate; a per-cell verdict (per-var gating
better/same/worse, more/less efficient). Plus a cross-cell verdict. Prints the table.
Cells with no variable_gating.json are skipped (resumable).

Usage (from src/):
    NONINTERACTIVE=1 python experiments/aggregate_variable_gating.py
"""
import os
import sys
import json

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import utils

RESULTS_DIR = os.path.join(SRC_DIR, "experiments", "results")


def fnum(x, nd=6):
    if x is None:
        return "-"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def pct(x, nd=2):
    if x is None:
        return "-"
    try:
        return f"{100.0 * float(x):.{nd}f}%"
    except (TypeError, ValueError):
        return str(x)


def op_desc(op, kind):
    """Short knob description for an operating-point record."""
    if op is None:
        return "NONE"
    if kind == "fixed":
        return op.get("key", "?")
    return f"b={op.get('beta')},q={op.get('q')}"


def main():
    cfg = utils.load_config("error_mlp_accident")
    out_root = cfg["out_root"]
    cell_names = list(cfg["cells"].keys())

    rows = []
    for cell in cell_names:
        p = os.path.join(out_root, cell, "variable_gating.json")
        if not os.path.exists(p):
            print(f"[aggregate] skip {cell} (no variable_gating.json)")
            continue
        with open(p) as f:
            r = json.load(f)
        rows.append(r)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    md_path = os.path.join(RESULTS_DIR, "variable_gating_summary.md")

    lines = ["# Per-(step, variable) vs STEP gated correction -- efficiency summary", ""]
    lines.append("Question: does gating the ErrorMLP correction on the VARIABLE axis "
                 "(correct only the poorly-predicted (step,variable) cells) match/beat "
                 "the STEP gate's tail (p99) while touching FEWER (step,variable) cells "
                 "(lower intervention rate), mean non-regressed? Metric = per-step MAE "
                 "(mean over 10 vars of |pred-true|); tail = p99 over (scenario,step). "
                 "Intervention rate = fraction of (step,variable) cells corrected (a "
                 "fired STEP corrects all 10 vars). OP = min p99 s.t. mean <= baseline_mean.")
    lines.append("")
    lines.append("| cell | base p99 | step OP | step p99 | step mean | step interv% | "
                 "var OP | var p99 | var mean | var interv% | fixed OP | fixed p99 | fixed mean | "
                 "dp99(var-step) | d_interv%(var-step) | verdict |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    n_better = n_same = n_worse = 0
    n_more_eff = n_less_eff = n_comp_eff = 0
    for r in rows:
        base = r["baseline"]
        os_ = r.get("operating_point_step_strict")
        ov = r.get("operating_point_var_strict")
        of = r.get("operating_point_fixed_strict")
        v = r.get("verdict", {})
        dp99 = v.get("p99_delta_var_minus_step")
        dir_ = v.get("intervention_delta_var_minus_step")
        tail_cls = v.get("tail_class", "-")
        eff_cls = v.get("efficiency_class", "-")

        if "BETTER" in tail_cls:
            n_better += 1
        elif "WORSE" in tail_cls:
            n_worse += 1
        elif "SAME" in tail_cls:
            n_same += 1
        if eff_cls.startswith("MORE"):
            n_more_eff += 1
        elif eff_cls.startswith("LESS"):
            n_less_eff += 1
        elif eff_cls.startswith("comparable"):
            n_comp_eff += 1

        lines.append(
            f"| {r['cell']} | {fnum(base['p99'])} | {op_desc(os_,'step')} | "
            f"{fnum(os_['p99']) if os_ else '-'} | {fnum(os_['mean']) if os_ else '-'} | "
            f"{pct(os_['intervention_rate']) if os_ else '-'} | "
            f"{op_desc(ov,'var')} | {fnum(ov['p99']) if ov else '-'} | "
            f"{fnum(ov['mean']) if ov else '-'} | "
            f"{pct(ov['intervention_rate']) if ov else '-'} | "
            f"{op_desc(of,'fixed')} | {fnum(of['p99']) if of else '-'} | "
            f"{fnum(of['mean']) if of else '-'} | "
            f"{fnum(dp99) if dp99 is not None else '-'} | "
            f"{pct(dir_) if dir_ is not None else '-'} | "
            f"{tail_cls}; {eff_cls} |")

    lines.append("")
    lines.append("## Per-cell verdicts")
    for r in rows:
        v = r.get("verdict", {})
        lines.append(f"- **{r['cell']}**: {v.get('summary', '-')}")

    lines.append("")
    lines.append("## Cross-cell verdict")
    n = len(rows)
    lines.append(f"- Cells analysed: {n}")
    lines.append(f"- Per-var tail vs step: BETTER={n_better}, SAME={n_same}, WORSE={n_worse}")
    lines.append(f"- Per-var efficiency vs step: MORE-efficient={n_more_eff}, "
                 f"LESS-efficient={n_less_eff}, comparable={n_comp_eff}")
    if n:
        if n_better + n_same == n and n_more_eff >= n_less_eff:
            cross = ("Per-(step,variable) gating matches or beats the step-gate tail in "
                     "every cell; on efficiency it is at least as good as the step gate "
                     f"in {n_more_eff + n_comp_eff}/{n} cells. ")
        elif n_worse > n_better:
            cross = "Per-(step,variable) gating tends to WORSEN the tail vs the step gate. "
        else:
            cross = "Mixed: per-(step,variable) gating helps in some cells, not others. "
        cross += ("This is consistent with the prior that the joint ErrorMLP already "
                  "self-restricts (outputs ~0 on well-predicted variables), so per-variable "
                  "gating adds little beyond step gating."
                  if (n_same >= n_better and n_more_eff <= n_comp_eff + n_less_eff)
                  else "Per-variable targeting delivers a measurable efficiency gain over "
                       "step gating, refuting the pure self-restriction prior.")
        lines.append(f"- {cross}")

    md = "\n".join(lines) + "\n"
    with open(md_path, "w") as f:
        f.write(md)
    print(md)
    print(f"[aggregate] wrote {md_path}")


if __name__ == "__main__":
    main()
