"""Honest operating-point protocol: select (beta*, q*, tau*) on the corrector's
held-out VALIDATION set, freeze tau*, then evaluate the frozen config ONCE on the
full TEST set.

Fixes the earlier test-set operating-point selection: the validation set is
train_csv restricted to the corrector's held-out scenarios (scenario-disjoint from
test; the corrector was NOT trained on them). tau* is the (1-q*) quantile of the
gate scores on the VALIDATION baseline rollout and is frozen for the test pass, so
the test set is never used to choose beta/q/tau.

Run from src/:
    NONINTERACTIVE=1 python experiments/op_val_select_test_eval.py --cell SBO ...
Uses error_mlp_dagger.pt (the adopted DAgger corrector) by default.
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
    load_frozen_backbone_acc, baseline_rollout_with_stats_acc, gated_corrected_rollout_acc,
)
from experiments.tail_analysis_acc import (
    step_err_from_dicts, tail_metrics, GATE_FRACTIONS, GATED_BETAS,
)

# headline reductions from the OLD (test-selected) operating point, for comparison
OLD_TEST_SELECTED_RED = {"SBO": 37.1, "LLOCA_CSP": 27.4, "LLOCA_ECSBS": 16.8,
                         "TLOFW_CSP": 25.7, "TLOFW_ECSBS": 15.9}


def gated_metrics(model, mlp, beta, tau, ds, ncontrols, snc, device, nw, cb, nc, restrict):
    pdc, tdc = gated_corrected_rollout_acc(
        model, mlp, beta, tau, ds, ncontrols, snc, device=device, gate_on="pred",
        num_continuous=nc, collect_batch=cb, num_workers=nw, restrict_scenarios=restrict)
    se, ps = step_err_from_dicts(pdc, tdc)
    return tail_metrics(se, ps)


def gated_metrics_with_fire(model, mlp, beta, tau, ds, ncontrols, snc, device, nw, cb, nc):
    pdc, tdc, _Xd, _Yd, _SID, interv = gated_corrected_rollout_acc(
        model, mlp, beta, tau, ds, ncontrols, snc, device=device, gate_on="pred",
        num_continuous=nc, collect_batch=cb, num_workers=nw, collect_dagger=True)
    se, ps = step_err_from_dicts(pdc, tdc)
    m = tail_metrics(se, ps)
    m["intervention_rate"] = float(interv["intervention_rate"])
    return m


def run_cell(cell_name, cfg, device, model_file="error_mlp_dagger.pt"):
    cell = cfg["cells"][cell_name]
    cfg_tr, cfg_data = cfg["training"], cfg["data"]
    nw = int(cfg_tr["num_workers"]); cb = int(cfg_tr.get("collect_batch", 2048))
    seq = int(cfg_data["seq_len"]); pred = int(cfg_data["pred_len"])
    cache = cfg_data.get("cache_dir"); nc = int(cfg_data["num_continuous"])
    out_dir = os.path.join(cfg["out_root"], cell_name)

    model, input_size = load_frozen_backbone_acc(cell["run_dir"], device)
    pt = torch.load(os.path.join(out_dir, model_file), map_location="cpu")
    in_dim = int(pt["in_dim"]); ncontrols = int(pt["num_controls"]); snc = float(pt["step_norm_const"])
    mlp = ErrorMLP(in_dim=in_dim, **cfg["error_mlp"]); mlp.load_state_dict(pt["state_dict"])
    mlp = mlp.to(device).eval()

    heldout = json.load(open(os.path.join(out_dir, "heldout_scenarios.json")))["heldout_scenarios"]
    heldout = [int(s) for s in heldout]

    train_ds = AccidentWindowDataset(cell["train_csv"], seq_len=seq, pred_len=pred, cache_dir=cache)
    test_ds = AccidentWindowDataset(cell["test_csv"], seq_len=seq, pred_len=pred, cache_dir=cache)

    # ===== VALIDATION (train_csv restricted to held-out; corrector never trained on it) =====
    vstats = baseline_rollout_with_stats_acc(
        model, mlp, train_ds, ncontrols, snc, device=device, num_continuous=nc,
        collect_batch=cb, num_workers=nw, restrict_scenarios=heldout)
    g_val = vstats["g"]
    vbase = gated_metrics(model, mlp, 0.0, np.inf, train_ds, ncontrols, snc, device, nw, cb, nc, heldout)
    vbmean, vbp99 = vbase["mean"], vbase["p99"]
    tau_of = {q: float(np.quantile(g_val, 1.0 - q)) for q in GATE_FRACTIONS}

    val_cfgs = {}
    for beta in GATED_BETAS:
        for q in GATE_FRACTIONS:
            m = gated_metrics(model, mlp, beta, tau_of[q], train_ds, ncontrols, snc,
                              device, nw, cb, nc, heldout)
            val_cfgs[(beta, q)] = m
            print(f"[{cell_name}][val] beta={beta} q={q} mean={m['mean']:.6f} p99={m['p99']:.6f} "
                  f"(<=vbase_mean {m['mean'] <= vbmean})")

    cands = [(k, m) for k, m in val_cfgs.items() if m["mean"] <= vbmean]
    relaxed = False
    if not cands:
        cands = [(k, m) for k, m in val_cfgs.items() if m["mean"] <= vbmean * 1.02]
        relaxed = True
    (beta_s, q_s), vsel = min(cands, key=lambda kv: kv[1]["p99"])
    tau_star = tau_of[q_s]                       # FROZEN from validation
    val_red = 100.0 * (vbp99 - vsel["p99"]) / vbp99
    print(f"[{cell_name}][VAL-PICK] beta*={beta_s} q*={q_s} tau*={tau_star:.6f} "
          f"relaxed={relaxed} | val p99 {vbp99:.6f}->{vsel['p99']:.6f} ({val_red:+.1f}%)")

    # ===== TEST (full; frozen beta*, tau*) once =====
    tbase = gated_metrics(model, mlp, 0.0, np.inf, test_ds, ncontrols, snc, device, nw, cb, nc, None)
    top = gated_metrics_with_fire(model, mlp, beta_s, tau_star, test_ds, ncontrols, snc,
                                  device, nw, cb, nc)
    red_p99 = 100.0 * (tbase["p99"] - top["p99"]) / tbase["p99"]
    red_mean = 100.0 * (tbase["mean"] - top["mean"]) / tbase["mean"]
    bw = tbase["worst10_scen_mean"]; nw10 = top["worst10_scen_mean"]
    red_w10 = 100.0 * (bw - nw10) / bw
    mean_ok = top["mean"] <= tbase["mean"]
    old = OLD_TEST_SELECTED_RED.get(cell_name, float("nan"))
    print(f"[{cell_name}][TEST] base p99={tbase['p99']:.6f} op p99={top['p99']:.6f} "
          f"red={red_p99:+.2f}% (OLD test-sel {old:+.1f}%) | mean {tbase['mean']:.6f}->"
          f"{top['mean']:.6f} ({red_mean:+.2f}%, ok={mean_ok}) | worst10 {red_w10:+.1f}% | "
          f"fire={top['intervention_rate']*100:.2f}%")

    return {
        "cell": cell_name, "beta_star": beta_s, "q_star": q_s, "tau_star": tau_star,
        "relaxed": relaxed,
        "val": {"baseline_p99": vbp99, "op_p99": vsel["p99"], "red_pct": val_red,
                "baseline_mean": vbmean, "op_mean": vsel["mean"], "n_heldout": len(heldout)},
        "test": {"baseline_p99": tbase["p99"], "op_p99": top["p99"], "red_pct": red_p99,
                 "baseline_mean": tbase["mean"], "op_mean": top["mean"], "mean_red_pct": red_mean,
                 "mean_neutral": bool(mean_ok), "worst10_red_pct": red_w10,
                 "fire_rate_pct": top["intervention_rate"] * 100.0},
        "old_test_selected_red_pct": old,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="append", required=True)
    ap.add_argument("--model-file", default="error_mlp_dagger.pt")
    args = ap.parse_args()
    cfg = utils.load_config("error_mlp_accident")
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    results = []
    for c in args.cell:
        results.append(run_cell(c, cfg, device, model_file=args.model_file))
    out = os.path.join(cfg["out_root"], "op_val_select_summary.json")
    json.dump(results, open(out, "w"), indent=2)
    print("\n===== SUMMARY: validation-selected op, evaluated on TEST =====")
    print(f"{'cell':13} {'b*':>4} {'q*':>6} | {'TEST p99 red%':>13} {'OLD(test-sel)':>13} "
          f"{'Δpp':>6} | {'mean-neutral':>12} {'worst10%':>8} {'fire%':>6}")
    for r in results:
        t = r["test"]; d = t["red_pct"] - r["old_test_selected_red_pct"]
        print(f"{r['cell']:13} {r['beta_star']:>4} {r['q_star']:>6} | {t['red_pct']:>12.1f}% "
              f"{r['old_test_selected_red_pct']:>12.1f}% {d:>+6.1f} | {str(t['mean_neutral']):>12} "
              f"{t['worst10_red_pct']:>7.1f}% {t['fire_rate_pct']:>5.2f}%")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
