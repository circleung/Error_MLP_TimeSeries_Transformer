"""GATED (selective) tail-correction analysis for a per-accident-type ErrorMLP.

REUSES the already-trained, frozen backbone + ErrorMLP (no retrain). The goal is
NOT to lower the mean per-step error (a GLOBAL beta only worsens it -- beta*=0 for
every cell), but to CUT the rare large-error moments (the tail: p99/p999/max of the
per-step error distribution) on TEST while keeping the mean (micro_mae) non-regressed.

Strategy: apply the ErrorMLP correction ONLY on the small fraction of steps where a
large error is predicted (a GATE); leave the rest uncorrected so the mean is
preserved, targeting the tail.

Per cell, on the TEST set:
  1. baseline_rollout_with_stats_acc -> baseline per-step error array + predicted-
     error L2 gate score `g` + realized-error L2 `true_g` (all aligned).
  2. tau(q) = quantile(g, 1-q) for gate fractions q; tau_true(q) = quantile(true_g,1-q).
  3. gated_corrected_rollout_acc(beta, tau, gate_on='pred') for beta in {0.5,1.0} x
     each q; ORACLE = gate_on='true' at the same q.
  4. GLOBAL beta {0.25..1.0} via autoregressive_corrected_batched_acc (diagnostic).

Metrics per config (pooled over all scenario x step): mean, p95, p99, p999, max,
plus worst-10 scenario mean-error. The per-step error unit = mean abs over the 10
continuous vars (== compute_micro_macro's per-step MAE).

Operating point (PREDICTED gate, beta>0): min p99 s.t. mean <= baseline_mean (strict);
also a relaxed pick (mean <= baseline_mean*1.02). If none gets p99 < baseline_p99
under the mean constraint -> honest negative, and we report how far the ORACLE gets.

Writes <out_root>/<cell>/tail_analysis.json. Asserts beta=0 gated == baseline.

Usage (from src/):
    NONINTERACTIVE=1 python experiments/tail_analysis_acc.py --cell SBO
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
from accident_dataset import AccidentWindowDataset
from models.error_mlp import ErrorMLP
from error_rollout_acc import (
    load_frozen_backbone_acc,
    autoregressive_corrected_batched_acc,
    baseline_rollout_with_stats_acc,
    gated_corrected_rollout_acc,
)


# Gate fractions (fraction of steps where the correction is allowed to fire).
GATE_FRACTIONS = [0.005, 0.01, 0.02, 0.05, 0.10]
GATED_BETAS = [0.5, 1.0]
GLOBAL_BETAS = [0.25, 0.5, 0.75, 1.0]
RELAX = 1.02  # relaxed operating-point mean tolerance


def step_err_from_dicts(predictions_dict, true_dict):
    """Flat per-step error array + per-scenario mean-error list from a
    (predictions_dict, true_dict) pair. Per-step error = mean abs over the 10
    continuous vars (SAME unit as compute_micro_macro's per-step MAE)."""
    flat = []
    per_scen_mean = []
    for sc in predictions_dict:
        if not predictions_dict[sc]:
            continue
        P = np.vstack([np.ravel(p) for p in predictions_dict[sc]]).astype(np.float64)
        Y = np.vstack([np.ravel(y) for y in true_dict[sc]]).astype(np.float64)
        mae_steps = np.mean(np.abs(P - Y), axis=1)   # [L] per-step MAE over 10 vars
        flat.append(mae_steps)
        per_scen_mean.append(float(mae_steps.mean()))
    flat_arr = np.concatenate(flat) if flat else np.zeros((0,), np.float64)
    return flat_arr, per_scen_mean


def tail_metrics(step_err, per_scen_mean):
    """mean/p95/p99/p999/max of a pooled per-step error array + worst-10 scenario
    mean-error (mean over each scenario's steps; take the 10 worst)."""
    step_err = np.asarray(step_err, dtype=np.float64)
    if step_err.size == 0:
        nan = float("nan")
        return dict(mean=nan, p95=nan, p99=nan, p999=nan, max=nan,
                    worst10_scen_mean=nan, n_steps=0, n_scen=0)
    worst = sorted(per_scen_mean, reverse=True)[:10]
    return dict(
        mean=float(step_err.mean()),
        p95=float(np.quantile(step_err, 0.95)),
        p99=float(np.quantile(step_err, 0.99)),
        p999=float(np.quantile(step_err, 0.999)),
        max=float(step_err.max()),
        worst10_scen_mean=float(np.mean(worst)) if worst else float("nan"),
        n_steps=int(step_err.size),
        n_scen=len(per_scen_mean),
    )


def tail_metrics_from_flat(step_err, sid):
    """Tail metrics directly from the baseline flat arrays (step_err aligned with
    scenario ids), reconstructing per-scenario means from the pooled arrays."""
    step_err = np.asarray(step_err, dtype=np.float64)
    sid = np.asarray(sid, dtype=np.int64)
    per_scen_mean = []
    for s in np.unique(sid):
        per_scen_mean.append(float(step_err[sid == s].mean()))
    return tail_metrics(step_err, per_scen_mean)


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

    run_dir = cell["run_dir"]
    model, input_size = load_frozen_backbone_acc(run_dir, device)

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

    # ---- 1. baseline pass with gate stats (uncorrected AR + aligned g/true_g) ----
    stats = baseline_rollout_with_stats_acc(
        model, mlp, test_ds, num_controls, step_norm_const, device=device,
        num_continuous=nc, collect_batch=collect_batch, num_workers=num_workers)
    g = stats["g"]
    true_g = stats["true_g"]
    # stats-path baseline metrics (float32-on-GPU reduction). Used for the gate
    # thresholds (g/true_g) and kept as a cross-check; NOT the comparison anchor.
    base_stats_metrics = tail_metrics_from_flat(stats["step_err"], stats["sid"])

    # CANONICAL baseline = the beta=0 (tau=inf) GATED rollout, computed through the
    # SAME float64 dict path as every gated/global/oracle config below, so the
    # operating-point mean/p99 comparisons are apples-to-apples (no float32-path
    # skew that could spuriously flip the strict mean<=baseline gate).
    pd0, td0 = gated_corrected_rollout_acc(
        model, mlp, 0.0, np.inf, test_ds, num_controls, step_norm_const,
        device=device, gate_on="pred", num_continuous=nc,
        collect_batch=collect_batch, num_workers=num_workers)
    se0, ps0 = step_err_from_dicts(pd0, td0)
    base_metrics = tail_metrics(se0, ps0)
    baseline_mean = base_metrics["mean"]
    baseline_p99 = base_metrics["p99"]
    print(f"[{args.cell}][baseline] mean={baseline_mean:.6f} p99={baseline_p99:.6f} "
          f"p999={base_metrics['p999']:.6f} max={base_metrics['max']:.6f} "
          f"(n_steps={base_metrics['n_steps']})")

    # ---- assert: beta=0 gated == stats-path baseline (null-op) ----
    # Both feed RAW back => SAME trajectory; only float32-vs-float64 reduction
    # rounding differs, so tolerance is set at the float32 level.
    NULL_TOL = 1e-6
    assert (abs(base_metrics["mean"] - base_stats_metrics["mean"]) < NULL_TOL
            and abs(base_metrics["max"] - base_stats_metrics["max"]) < NULL_TOL
            and abs(base_metrics["p99"] - base_stats_metrics["p99"]) < NULL_TOL), (
        f"null-op broken: gated beta=0 mean/p99/max "
        f"{base_metrics['mean']}/{base_metrics['p99']}/{base_metrics['max']} != stats "
        f"{base_stats_metrics['mean']}/{base_stats_metrics['p99']}/{base_stats_metrics['max']}")
    print(f"[{args.cell}][null-op] gated beta=0 == stats-path baseline OK "
          f"(|dmean|={abs(base_metrics['mean']-base_stats_metrics['mean']):.2e})")

    # ---- thresholds from the baseline g / true_g distributions ----
    thresholds_pred = {q: float(np.quantile(g, 1.0 - q)) for q in GATE_FRACTIONS}
    thresholds_true = {q: float(np.quantile(true_g, 1.0 - q)) for q in GATE_FRACTIONS}

    def roll_metrics_gated(beta, q, gate_on):
        tau = thresholds_pred[q] if gate_on == "pred" else thresholds_true[q]
        pdc, tdc = gated_corrected_rollout_acc(
            model, mlp, beta, tau, test_ds, num_controls, step_norm_const,
            device=device, gate_on=gate_on, num_continuous=nc,
            collect_batch=collect_batch, num_workers=num_workers)
        se, ps = step_err_from_dicts(pdc, tdc)
        return tail_metrics(se, ps)

    # ---- 2. GLOBAL beta (diagnostic; full correction, no gate) ----
    global_configs = {}
    for beta in GLOBAL_BETAS:
        pdc, tdc = autoregressive_corrected_batched_acc(
            model, mlp, beta, test_ds, num_controls, step_norm_const, device=device,
            num_continuous=nc, collect_batch=collect_batch, num_workers=num_workers)
        se, ps = step_err_from_dicts(pdc, tdc)
        gm = tail_metrics(se, ps)
        global_configs[f"beta={beta}"] = gm
        print(f"[{args.cell}][global] beta={beta} mean={gm['mean']:.6f} p99={gm['p99']:.6f} "
              f"max={gm['max']:.6f}")

    # ---- 3. GATED predicted + ORACLE ----
    gated_configs = {}   # key: "beta=..,q=.." -> metrics (predicted gate)
    oracle_configs = {}  # key: "q=.."         -> metrics (true-error gate, beta=1.0)
    for beta in GATED_BETAS:
        for q in GATE_FRACTIONS:
            gm = roll_metrics_gated(beta, q, "pred")
            key = f"beta={beta},q={q}"
            gated_configs[key] = gm
            print(f"[{args.cell}][gated ] {key} mean={gm['mean']:.6f} "
                  f"p99={gm['p99']:.6f} max={gm['max']:.6f} "
                  f"(<=base_mean: {gm['mean'] <= baseline_mean})")
    # ORACLE: gate on the TRUE error at the same fractions (detector ceiling). Use
    # the stronger beta=1.0 (best-case correction on the truly-large-error steps).
    for q in GATE_FRACTIONS:
        gm = roll_metrics_gated(1.0, q, "true")
        oracle_configs[f"q={q}"] = gm
        print(f"[{args.cell}][oracle] q={q} beta=1.0 mean={gm['mean']:.6f} "
              f"p99={gm['p99']:.6f} max={gm['max']:.6f}")

    # ---- operating-point selection (PREDICTED gate, beta>0) ----
    def pick(mean_cap):
        cands = [(k, m) for k, m in gated_configs.items() if m["mean"] <= mean_cap]
        if not cands:
            return None
        k, m = min(cands, key=lambda kv: kv[1]["p99"])
        beta_s = float(k.split("beta=")[1].split(",")[0])
        q_s = float(k.split("q=")[1])
        return {
            "key": k, "beta": beta_s, "q": q_s, "tau_pred": thresholds_pred[q_s],
            "mean": m["mean"], "p95": m["p95"], "p99": m["p99"],
            "p999": m["p999"], "max": m["max"],
            "worst10_scen_mean": m["worst10_scen_mean"],
            "p99_reduction_pct": (100.0 * (baseline_p99 - m["p99"]) / baseline_p99
                                  if baseline_p99 > 0 else float("nan")),
            "mean_delta_vs_baseline": m["mean"] - baseline_mean,
            "cuts_tail": bool(m["p99"] < baseline_p99),
        }

    op_strict = pick(baseline_mean)
    op_relaxed = pick(baseline_mean * RELAX)

    # Honest negative: does ANY strict-mean predicted-gate config cut p99?
    strict_cuts_tail = bool(op_strict is not None and op_strict["cuts_tail"])
    verdict = ("gating cuts the tail at mean-neutral operating point"
               if strict_cuts_tail else
               "NO strict-mean predicted-gate config reduces p99 below baseline")

    # ORACLE p99 ceiling (best oracle p99 across q; tells detector vs correction).
    oracle_best_q = min(oracle_configs, key=lambda k: oracle_configs[k]["p99"])
    oracle_ceiling = {
        "q": float(oracle_best_q.split("q=")[1]),
        "p99": oracle_configs[oracle_best_q]["p99"],
        "mean": oracle_configs[oracle_best_q]["mean"],
        "max": oracle_configs[oracle_best_q]["max"],
        "p99_reduction_pct": (100.0 * (baseline_p99 - oracle_configs[oracle_best_q]["p99"])
                              / baseline_p99 if baseline_p99 > 0 else float("nan")),
    }

    result = {
        "cell": args.cell, "seq_len": seq_len, "num_controls": num_controls,
        "in_dim": in_dim, "step_norm_const": step_norm_const,
        "gate_fractions": GATE_FRACTIONS, "gated_betas": GATED_BETAS,
        "global_betas": GLOBAL_BETAS,
        "thresholds_pred": {str(q): v for q, v in thresholds_pred.items()},
        "thresholds_true": {str(q): v for q, v in thresholds_true.items()},
        "baseline": base_metrics,
        "baseline_stats_path": base_stats_metrics,
        "global_beta": global_configs,
        "gated_pred": gated_configs,
        "oracle_true": oracle_configs,
        "operating_point_strict": op_strict,
        "operating_point_relaxed": op_relaxed,
        "oracle_p99_ceiling": oracle_ceiling,
        "strict_cuts_tail": strict_cuts_tail,
        "verdict": verdict,
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "tail_analysis.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{args.cell}][done] wrote {out_path}")
    if op_strict is not None:
        print(f"  strict OP: {op_strict['key']} mean={op_strict['mean']:.6f} "
              f"p99={op_strict['p99']:.6f} (base p99={baseline_p99:.6f}, "
              f"red={op_strict['p99_reduction_pct']:.2f}%) cuts_tail={op_strict['cuts_tail']}")
    else:
        print("  strict OP: NONE (no gated config with mean <= baseline_mean)")
    print(f"  ORACLE ceiling p99={oracle_ceiling['p99']:.6f} "
          f"(red={oracle_ceiling['p99_reduction_pct']:.2f}%) at q={oracle_ceiling['q']}")
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
