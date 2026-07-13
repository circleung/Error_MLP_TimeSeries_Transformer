"""Per-(step, variable) GATED tail-correction vs the STEP-axis gate (efficiency).

REUSES the already-trained, frozen backbone + ErrorMLP (NO retrain). Question: does
gating the ErrorMLP correction on the VARIABLE axis (correct only the poorly-predicted
(step, variable) cells) match/beat the current STEP-axis gate's tail (p99) while
touching FEWER (step, variable) cells (lower intervention rate) and keeping the mean
non-regressed? The joint ErrorMLP already outputs ~0 for well-predicted variables, so
the honest prior is that per-variable gating may add little over step gating -- this
script quantifies it.

Metric unit = per-step MAE = mean over the 10 continuous vars of |pred-true| (== the
tail_analysis / compute_micro_macro per-step MAE); tail = p99 pooled over (scenario,
step). Intervention rate = fraction of (step, variable) cells where a correction was
actually applied (same unit for the step gate, where a fired step corrects all 10 vars).

Variants compared, per cell on TEST:
  1. baseline  (beta=0, no correction) -- anchor.
  2. step-gate (current): correct the whole step if ||e_hat_t||_2 > tau_step, where
     tau_step = quantile(||e_hat||_2, 1-q). Sweep beta x q.  (recomputed here; the
     existing tail_analysis.json operating point is also echoed for reference.)
  3. per-(step,variable) DYNAMIC gate (the main new one): correct cell (t,j) iff
     |e_hat_{t,j}| > tau_j, tau_j = quantile(|e_hat_j|, 1-q) from a beta=0 calibration
     pass (target fraction q of that variable's cells). Feed back the mixed row.
     Sweep beta x q.
  4. per-variable FIXED-set gate (the literal idea): correct only the K hardest OUTPUT
     variables (by baseline per-output p99) at ALL steps, beta swept. K in {2,3}.

Operating-point rule = min p99 s.t. mean <= baseline_mean (strict); also relaxed
(mean <= 1.02*baseline_mean). Null-op asserts: beta=0 == baseline exactly AND a
finite-but-unreachable var gate (nothing fires) == baseline exactly.

Writes <out_root>/<cell>/variable_gating.json. Asserts the backbone is frozen.

Usage (from src/):
    NONINTERACTIVE=1 python experiments/variable_gating_acc.py --cell SBO
"""
import os
import sys
import json
import argparse

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

os.environ.setdefault("NONINTERACTIVE", "1")

import numpy as np
import torch

import utils
from accident_dataset import AccidentWindowDataset, CONTINUOUS_COLS
from models.error_mlp import ErrorMLP
from error_rollout_acc import (
    load_frozen_backbone_acc,
    baseline_var_stats_acc,
    var_gated_corrected_rollout_acc,
)
from experiments.tail_analysis_acc import step_err_from_dicts, tail_metrics
from experiments.variable_analysis_acc import per_output_ar_error


# Sweep grid (matches tail_analysis' fractions; betas per the task spec).
GATE_FRACTIONS = [0.005, 0.01, 0.02, 0.05, 0.10]
BETAS = [0.5, 1.0]
FIXED_K_LIST = [2, 3]
RELAX = 1.02              # relaxed operating-point mean tolerance
NULL_TOL = 1e-6           # float32-vs-float64 reduction tolerance


def assert_frozen(model):
    """Hard-assert the backbone carries no trainable parameters (NO retraining)."""
    n_trainable = sum(int(p.requires_grad) for p in model.parameters())
    assert n_trainable == 0, f"backbone is NOT frozen: {n_trainable} trainable params"
    assert not model.training, "backbone must be in eval() mode"


def eval_config(pdc, tdc, interv, nc):
    """Full metric bundle for one rollout config: mean/p95/p99/p999/max (per-step MAE
    pooled over scenario x step), per-output p99, and the (step,variable)-cell
    intervention rate."""
    se, ps = step_err_from_dicts(pdc, tdc)
    m = tail_metrics(se, ps)
    _, per_out_p99, _ = per_output_ar_error(pdc, tdc, nc)
    m["per_out_p99"] = [float(v) for v in per_out_p99]
    m["intervention_rate"] = float(interv["intervention_rate"])
    m["n_fired"] = int(interv["n_fired"])
    m["n_active_cells"] = int(interv["n_active_cells"])
    m["per_var_rate"] = [float(v) for v in interv["per_var_rate"]]
    return m


def op_entry(key, m, baseline_mean, baseline_p99, extra):
    """Operating-point record (metrics of the selected config + deltas vs baseline)."""
    d = {
        "key": key,
        "mean": m["mean"], "p95": m["p95"], "p99": m["p99"],
        "p999": m["p999"], "max": m["max"],
        "per_out_p99": m["per_out_p99"],
        "intervention_rate": m["intervention_rate"],
        "n_fired": m["n_fired"], "n_active_cells": m["n_active_cells"],
        "p99_reduction_pct": (100.0 * (baseline_p99 - m["p99"]) / baseline_p99
                              if baseline_p99 > 0 else float("nan")),
        "mean_delta_vs_baseline": m["mean"] - baseline_mean,
        "cuts_tail": bool(m["p99"] < baseline_p99),
    }
    d.update(extra)
    return d


def pick_op(configs, meta, mean_cap, baseline_mean, baseline_p99):
    """min-p99 config s.t. mean <= mean_cap. `meta[key]` carries the parsed knobs
    (beta/q or K/beta) folded into the operating-point record. Returns None if no
    config satisfies the mean cap."""
    cands = [(k, m) for k, m in configs.items() if m["mean"] <= mean_cap]
    if not cands:
        return None
    k, m = min(cands, key=lambda kv: kv[1]["p99"])
    return op_entry(k, m, baseline_mean, baseline_p99, meta[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    args = ap.parse_args()

    cfg = utils.load_config("error_mlp_accident")
    cells = cfg["cells"]
    assert args.cell in cells, f"--cell {args.cell} not in {list(cells)}"
    cell = cells[args.cell]
    cfg_tr = cfg["training"]
    cfg_data = cfg["data"]
    device = torch.device(cfg_tr["device"] if torch.cuda.is_available() else "cpu")

    num_workers = int(cfg_tr["num_workers"])
    collect_batch = int(cfg_tr.get("collect_batch", 2048))
    seq_len = int(cfg_data["seq_len"])
    pred_len = int(cfg_data["pred_len"])
    cache_dir = cfg_data.get("cache_dir")
    nc = int(cfg_data["num_continuous"])
    cont_cols = list(CONTINUOUS_COLS)

    run_dir = cell["run_dir"]
    model, input_size = load_frozen_backbone_acc(run_dir, device)
    assert_frozen(model)   # NO retraining: hard-assert the backbone is frozen.

    out_dir = os.path.join(cfg["out_root"], args.cell)
    pt = torch.load(os.path.join(out_dir, "error_mlp.pt"), map_location="cpu")
    in_dim = int(pt["in_dim"])
    num_controls = int(pt["num_controls"])
    step_norm_const = float(pt["step_norm_const"])
    mlp = ErrorMLP(in_dim=in_dim, **cfg["error_mlp"])
    mlp.load_state_dict(pt["state_dict"])
    mlp = mlp.to(device).eval()

    test_ds = AccidentWindowDataset(cell["test_csv"], seq_len=seq_len,
                                    pred_len=pred_len, cache_dir=cache_dir)
    assert test_ds.num_controls == num_controls and test_ds.input_size == input_size

    def roll(beta, gate, tau_vec=None, tau_step=None):
        return var_gated_corrected_rollout_acc(
            model, mlp, beta, test_ds, num_controls, step_norm_const, gate=gate,
            tau_vec=tau_vec, tau_step=tau_step, device=device, num_continuous=nc,
            collect_batch=collect_batch, num_workers=num_workers)

    # ---- 1. baseline calibration pass (per-variable |e_hat| + step-norm g) --------
    stats = baseline_var_stats_acc(
        model, mlp, test_ds, num_controls, step_norm_const, device=device,
        num_continuous=nc, collect_batch=collect_batch, num_workers=num_workers)
    abs_ehat = stats["abs_ehat"]                              # [N, 10]
    g = stats["g"]                                            # [N]
    base_stats_mean = float(np.mean(stats["step_err"]))
    base_stats_p99 = float(np.quantile(stats["step_err"], 0.99))
    base_stats_max = float(np.max(stats["step_err"]))

    # ---- CANONICAL baseline = beta=0 var-gated rollout (float64 dict path), matching
    #      every other config's reduction path for apples-to-apples comparisons. ------
    inf_vec = np.full(nc, np.inf, dtype=np.float32)
    pd0, td0, iv0 = roll(0.0, "var", tau_vec=inf_vec)
    m0 = eval_config(pd0, td0, iv0, nc)
    baseline_mean = m0["mean"]
    baseline_p99 = m0["p99"]
    base_per_out_p99 = m0["per_out_p99"]
    print(f"[{args.cell}][baseline] mean={baseline_mean:.6f} p99={baseline_p99:.6f} "
          f"p999={m0['p999']:.6f} max={m0['max']:.6f} interv={m0['intervention_rate']:.4f} "
          f"(n_steps={m0['n_steps']})")

    # ---- null-op asserts ----------------------------------------------------------
    # (a) beta=0 var-gate == stats-path baseline (both feed RAW back).
    beta0_ok = bool(abs(baseline_mean - base_stats_mean) < NULL_TOL
                    and abs(baseline_p99 - base_stats_p99) < NULL_TOL
                    and abs(m0["max"] - base_stats_max) < NULL_TOL)
    assert beta0_ok, (f"null-op broken: beta=0 dict mean/p99/max "
                      f"{baseline_mean}/{baseline_p99}/{m0['max']} != stats "
                      f"{base_stats_mean}/{base_stats_p99}/{base_stats_max}")
    # (b) beta=1.0 with a finite-but-unreachable var gate (nothing fires) == baseline
    #     EXACTLY -> exercises the torch.where masking path (not just the short-circuit).
    big_vec = np.full(nc, 1e30, dtype=np.float32)
    pde, tde, ive = roll(1.0, "var", tau_vec=big_vec)
    me = eval_config(pde, tde, ive, nc)
    empty_gate_ok = bool(me["mean"] == baseline_mean and me["p99"] == baseline_p99
                         and me["max"] == m0["max"] and me["n_fired"] == 0)
    assert empty_gate_ok, (f"empty-gate null-op broken: mean/p99/max/n_fired "
                           f"{me['mean']}/{me['p99']}/{me['max']}/{me['n_fired']} != "
                           f"baseline {baseline_mean}/{baseline_p99}/{m0['max']}/0")
    print(f"[{args.cell}][null-op] beta0={beta0_ok} empty-gate={empty_gate_ok} "
          f"(|dmean_stats|={abs(baseline_mean-base_stats_mean):.2e})")

    # ---- thresholds from the baseline distributions -------------------------------
    thresholds_step = {q: float(np.quantile(g, 1.0 - q)) for q in GATE_FRACTIONS}
    thresholds_var = {q: [float(np.quantile(abs_ehat[:, j], 1.0 - q)) for j in range(nc)]
                      for q in GATE_FRACTIONS}
    # hardest OUTPUT variables by baseline per-output p99 (for the fixed-set gate).
    hard_order = list(np.argsort(base_per_out_p99)[::-1])
    hardest_vars = [{"index": int(j), "name": cont_cols[j], "base_p99": float(base_per_out_p99[j])}
                    for j in hard_order]
    print(f"[{args.cell}][hardest-by-p99] "
          + ", ".join(f"{cont_cols[j]}({base_per_out_p99[j]:.4f})" for j in hard_order[:3]))

    # ---- 2. STEP gate -------------------------------------------------------------
    step_configs, step_meta = {}, {}
    for beta in BETAS:
        for q in GATE_FRACTIONS:
            pdc, tdc, ivc = roll(beta, "step", tau_step=thresholds_step[q])
            m = eval_config(pdc, tdc, ivc, nc)
            key = f"beta={beta},q={q}"
            step_configs[key] = m
            step_meta[key] = {"beta": beta, "q": q, "tau_step": thresholds_step[q]}
            print(f"[{args.cell}][step ] {key} mean={m['mean']:.6f} p99={m['p99']:.6f} "
                  f"interv={m['intervention_rate']:.4f} (<=base_mean: {m['mean']<=baseline_mean})")

    # ---- 3. per-(step,variable) DYNAMIC gate --------------------------------------
    var_configs, var_meta = {}, {}
    for beta in BETAS:
        for q in GATE_FRACTIONS:
            pdc, tdc, ivc = roll(beta, "var", tau_vec=np.asarray(thresholds_var[q], np.float32))
            m = eval_config(pdc, tdc, ivc, nc)
            key = f"beta={beta},q={q}"
            var_configs[key] = m
            var_meta[key] = {"beta": beta, "q": q}
            print(f"[{args.cell}][var  ] {key} mean={m['mean']:.6f} p99={m['p99']:.6f} "
                  f"interv={m['intervention_rate']:.4f} (<=base_mean: {m['mean']<=baseline_mean})")

    # ---- 4. per-variable FIXED-set gate -------------------------------------------
    fixed_configs, fixed_meta = {}, {}
    for K in FIXED_K_LIST:
        chosen = [int(j) for j in hard_order[:K]]
        tv = np.full(nc, np.inf, dtype=np.float32)
        for j in chosen:
            tv[j] = -np.inf                                   # always correct these vars
        for beta in BETAS:
            pdc, tdc, ivc = roll(beta, "var", tau_vec=tv)
            m = eval_config(pdc, tdc, ivc, nc)
            key = f"K={K},beta={beta}"
            fixed_configs[key] = m
            fixed_meta[key] = {"K": K, "beta": beta,
                               "vars": [cont_cols[j] for j in chosen],
                               "var_indices": chosen}
            print(f"[{args.cell}][fixed] {key} vars={[cont_cols[j] for j in chosen]} "
                  f"mean={m['mean']:.6f} p99={m['p99']:.6f} interv={m['intervention_rate']:.4f}")

    # ---- operating points ---------------------------------------------------------
    op_step_strict = pick_op(step_configs, step_meta, baseline_mean, baseline_mean, baseline_p99)
    op_step_relaxed = pick_op(step_configs, step_meta, baseline_mean * RELAX, baseline_mean, baseline_p99)
    op_var_strict = pick_op(var_configs, var_meta, baseline_mean, baseline_mean, baseline_p99)
    op_var_relaxed = pick_op(var_configs, var_meta, baseline_mean * RELAX, baseline_mean, baseline_p99)
    op_fixed_strict = pick_op(fixed_configs, fixed_meta, baseline_mean, baseline_mean, baseline_p99)
    op_fixed_relaxed = pick_op(fixed_configs, fixed_meta, baseline_mean * RELAX, baseline_mean, baseline_p99)

    # ---- reference: the existing tail_analysis step-gate operating point ----------
    reference_tail_op = None
    tail_path = os.path.join(out_dir, "tail_analysis.json")
    if os.path.exists(tail_path):
        with open(tail_path) as f:
            tj = json.load(f)
        reference_tail_op = tj.get("operating_point_strict")

    # ---- per-cell verdict (efficiency): var-gate vs step-gate at strict OP ---------
    verdict = build_verdict(op_step_strict, op_var_strict, op_fixed_strict, baseline_p99)
    print(f"[{args.cell}][verdict] {verdict['summary']}")

    result = {
        "cell": args.cell, "seq_len": seq_len, "num_controls": num_controls,
        "in_dim": in_dim, "step_norm_const": step_norm_const,
        "continuous_cols": cont_cols,
        "betas": BETAS, "gate_fractions": GATE_FRACTIONS, "fixed_K_list": FIXED_K_LIST,
        "baseline": {
            "mean": baseline_mean, "p95": m0["p95"], "p99": baseline_p99,
            "p999": m0["p999"], "max": m0["max"], "per_out_p99": base_per_out_p99,
            "n_steps": m0["n_steps"], "n_scen": m0["n_scen"],
        },
        "thresholds_step": {str(q): v for q, v in thresholds_step.items()},
        "thresholds_var": {str(q): v for q, v in thresholds_var.items()},
        "hardest_vars_by_p99": hardest_vars,
        "step_gate": step_configs,
        "var_gate": var_configs,
        "fixed_gate": fixed_configs,
        "fixed_meta": fixed_meta,
        "operating_point_step_strict": op_step_strict,
        "operating_point_step_relaxed": op_step_relaxed,
        "operating_point_var_strict": op_var_strict,
        "operating_point_var_relaxed": op_var_relaxed,
        "operating_point_fixed_strict": op_fixed_strict,
        "operating_point_fixed_relaxed": op_fixed_relaxed,
        "reference_tail_op": reference_tail_op,
        "nullop": {"beta0_ok": beta0_ok, "empty_gate_ok": empty_gate_ok,
                   "dmean_stats_vs_dict": abs(baseline_mean - base_stats_mean)},
        "verdict": verdict,
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "variable_gating.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{args.cell}][done] wrote {out_path}")


def build_verdict(op_step, op_var, op_fixed, baseline_p99):
    """Compare the per-(step,variable) DYNAMIC gate to the STEP gate at the strict
    (mean-non-regress) operating point: does it match/beat p99 while touching fewer
    cells? Classify better/same/worse and more/less/equally efficient."""
    def g(op, k):
        return op[k] if op else None

    step_p99 = g(op_step, "p99")
    var_p99 = g(op_var, "p99")
    step_ir = g(op_step, "intervention_rate")
    var_ir = g(op_var, "intervention_rate")

    p99_delta = (var_p99 - step_p99) if (step_p99 is not None and var_p99 is not None) else None
    ir_delta = (var_ir - step_ir) if (step_ir is not None and var_ir is not None) else None
    # relative p99 vs the step gate (var/step); <1 => var lower (better).
    p99_ratio = (var_p99 / step_p99) if (step_p99 and var_p99 is not None) else None
    ir_ratio = (var_ir / step_ir) if (step_ir and var_ir is not None) else None

    if op_step is None and op_var is None:
        tail_cls, eff_cls = "neither has a strict OP", "n/a"
    elif op_var is None:
        tail_cls, eff_cls = "var-gate has NO strict OP (step does)", "less efficient"
    elif op_step is None:
        tail_cls, eff_cls = "var-gate has a strict OP (step does NOT)", "more efficient"
    else:
        # tail class: within +-1% of step p99 => same, lower => better, higher => worse.
        if p99_ratio <= 0.99:
            tail_cls = "per-var gating BETTER p99"
        elif p99_ratio >= 1.01:
            tail_cls = "per-var gating WORSE p99"
        else:
            tail_cls = "per-var gating SAME p99 (+-1%)"
        # efficiency: fewer cells touched at no p99 cost => more efficient.
        if p99_ratio <= 1.01 and ir_ratio is not None and ir_ratio < 0.99:
            eff_cls = "MORE efficient (fewer cells, p99 not worse)"
        elif ir_ratio is not None and ir_ratio > 1.01 and p99_ratio >= 0.99:
            eff_cls = "LESS efficient (more cells, p99 not better)"
        else:
            eff_cls = "comparable efficiency"

    summary = (f"tail: {tail_cls}; efficiency: {eff_cls} "
               f"(step p99={_fmt(step_p99)} ir={_fmt(step_ir,4)} | "
               f"var p99={_fmt(var_p99)} ir={_fmt(var_ir,4)} | "
               f"fixed p99={_fmt(g(op_fixed,'p99'))})")
    return {
        "summary": summary,
        "tail_class": tail_cls,
        "efficiency_class": eff_cls,
        "p99_delta_var_minus_step": p99_delta,
        "intervention_delta_var_minus_step": ir_delta,
        "p99_ratio_var_over_step": p99_ratio,
        "intervention_ratio_var_over_step": ir_ratio,
        "step_p99": step_p99, "var_p99": var_p99, "fixed_p99": g(op_fixed, "p99"),
        "step_intervention_rate": step_ir, "var_intervention_rate": var_ir,
        "fixed_intervention_rate": g(op_fixed, "intervention_rate"),
    }


def _fmt(x, nd=6):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "-"


if __name__ == "__main__":
    main()
