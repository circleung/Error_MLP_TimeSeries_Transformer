"""Evaluate the trained ErrorMLP AR correction (60min/k3/seed42).

Pipeline:
  1. Beta selection: for each beta, run a FULL corrected AR rollout restricted to
     the persisted held-out scenarios (NOT a replay of stored open-loop CY/BY), and
     pick beta* = argmin eval.select_metric. (Single, unambiguous mechanism.)
  2. TEST reporting: for every beta, full corrected AR on the TEST dataset ->
     compute_micro_macro; the headline is beta* on TEST.
  3. Per-scenario win fraction at beta*: fraction of TEST scenarios whose corrected
     per-scenario MAE < baseline per-scenario MAE (baseline = the beta=0 corrected
     AR, which equals the canonical baseline; verified against the snapshot number).
  4. Step-index ablation at beta*: rerun TEST corrected AR with step_norm zeroed.
  5. Write error_mlp_eval.json with the full curve, selection curve, win fraction,
     ablation, non-regression flags (micro AND macro), and positive_result (AC3).

Usage (from src/):
    NONINTERACTIVE=1 python experiments/eval_error_mlp.py --interval 60min --k 3 --seed 42 --betas 0,0.25,0.5,0.75,1.0
"""
import os
import sys
import json
import argparse

# make src/ importable
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

os.environ.setdefault("NONINTERACTIVE", "1")

import numpy as np
import torch

import utils
import experiments.sweep_config as C
from models.error_mlp import ErrorMLP
from error_rollout import load_frozen_backbone, autoregressive_corrected_batched
from predict_batched import compute_micro_macro


def per_scenario_mae(predictions_dict, true_dict):
    """Per-scenario MAE (mean over the 10 vars, then mean over the scenario's steps),
    matching compute_micro_macro's macro definition. Returns {sid: mae}."""
    out = {}
    for sc in predictions_dict:
        if not predictions_dict[sc]:
            continue
        P = np.vstack([np.ravel(p) for p in predictions_dict[sc]]).astype(np.float64)
        Y = np.vstack([np.ravel(y) for y in true_dict[sc]]).astype(np.float64)
        out[int(sc)] = float(np.mean(np.abs(P - Y)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--betas", default=None, help="comma list, overrides config")
    ap.add_argument("--plot", action="store_true", help="emit error-accumulation plots")
    args = ap.parse_args()

    cfg = utils.load_config("error_mlp")
    cfg_tr = cfg["training"]
    cfg_ev = cfg["eval"]
    device = torch.device(cfg_tr["device"] if torch.cuda.is_available() else "cpu")

    betas = ([float(b) for b in args.betas.split(",")] if args.betas
             else [float(b) for b in cfg_ev["betas"]])
    select_metric = cfg_ev["select_metric"]
    num_workers = int(cfg_tr["num_workers"])

    run_dir = cfg["backbone"]["run_dir"]
    ckpt = cfg["backbone"].get("ckpt")
    model = load_frozen_backbone(run_dir, device, ckpt=ckpt)

    out_dir = cfg_tr["out_dir"]
    pt = torch.load(os.path.join(out_dir, "error_mlp.pt"), map_location="cpu")
    mlp = ErrorMLP(**cfg["error_mlp"])
    mlp.load_state_dict(pt["state_dict"])
    mlp = mlp.to(device).eval()

    with open(os.path.join(out_dir, "heldout_scenarios.json")) as f:
        heldout_ids = [int(s) for s in json.load(f)["heldout_scenarios"]]

    train_csv = cfg["data"]["train_csv"]
    test_csv = cfg["data"]["test_csv"]
    seq_len = int(cfg["data"]["seq_len"])
    pred_len = int(cfg["data"]["pred_len"])
    ptype = cfg["data"]["prediction_type"]

    # ---- 1. Beta selection on held-out (full corrected AR over TRAIN, restricted) ----
    train_ds = utils.get_dataset(train_csv, seq_len, pred_len, ptype)
    selection_curve = {}
    for beta in betas:
        pd_, td_ = autoregressive_corrected_batched(
            model, mlp, beta, train_ds, device=device, num_workers=num_workers,
            restrict_scenarios=heldout_ids)
        m = compute_micro_macro(pd_, td_)
        selection_curve[f"{beta}"] = m
        print(f"[select] beta={beta} heldout {select_metric}={m[select_metric]:.6f}")
    beta_star = min(betas, key=lambda b: selection_curve[f"{b}"][select_metric])
    print(f"[select] beta* = {beta_star}")

    # ---- 2. TEST reporting per beta ----
    test_ds = utils.get_dataset(test_csv, seq_len, pred_len, ptype)
    test_curve = {}
    per_scen_by_beta = {}
    for beta in betas:
        pd_, td_ = autoregressive_corrected_batched(
            model, mlp, beta, test_ds, device=device, num_workers=num_workers)
        m = compute_micro_macro(pd_, td_)
        test_curve[f"{beta}"] = m
        per_scen_by_beta[beta] = per_scenario_mae(pd_, td_)
        print(f"[test] beta={beta} micro_mae={m['micro_mae']:.8f} macro_mae={m['macro_mae']:.8f}")

    baseline_micro = float(cfg_ev["baseline_ar_mae"])
    baseline_macro = float(cfg_ev["baseline_ar_macro_mae"])

    # baseline per-scenario MAE = beta=0 corrected AR (== canonical baseline).
    if 0.0 not in [float(b) for b in betas]:
        pd0, td0 = autoregressive_corrected_batched(
            model, mlp, 0.0, test_ds, device=device, num_workers=num_workers)
        base_per_scen = per_scenario_mae(pd0, td0)
    else:
        base_per_scen = per_scen_by_beta[0.0]

    # verify beta=0 reproduces canonical baseline (AC2 sanity in eval too)
    beta0_micro = test_curve.get("0.0", {}).get("micro_mae")
    null_op_ok = (beta0_micro is not None and abs(beta0_micro - baseline_micro) <= 1e-6)

    # ---- 3. per-scenario win fraction at beta* ----
    star_per_scen = per_scen_by_beta[beta_star]
    common = sorted(set(star_per_scen) & set(base_per_scen))
    wins = sum(1 for s in common if star_per_scen[s] < base_per_scen[s])
    win_fraction = wins / len(common) if common else float("nan")
    print(f"[win] beta*={beta_star} win_fraction={win_fraction:.4f} ({wins}/{len(common)})")

    # ---- 4. step-index ablation at beta* on TEST (step_norm zeroed) ----
    ablation = None
    if bool(cfg_ev.get("step_index_ablation", False)):
        pda, tda = autoregressive_corrected_batched(
            model, mlp, beta_star, test_ds, device=device, num_workers=num_workers,
            step_norm_scale=0.0)
        abl_m = compute_micro_macro(pda, tda)
        star_m = test_curve[f"{beta_star}"]
        ablation = {
            "metrics_step_norm_zeroed": abl_m,
            "delta_vs_beta_star": {
                "micro_mae": abl_m["micro_mae"] - star_m["micro_mae"],
                "macro_mae": abl_m["macro_mae"] - star_m["macro_mae"],
                "micro_rmse": abl_m["micro_rmse"] - star_m["micro_rmse"],
                "macro_rmse": abl_m["macro_rmse"] - star_m["macro_rmse"],
            },
        }
        print(f"[ablation] step_norm=0 micro_mae={abl_m['micro_mae']:.8f} "
              f"(delta {ablation['delta_vs_beta_star']['micro_mae']:+.2e})")

    star_metrics = test_curve[f"{beta_star}"]
    micro_nonreg = star_metrics["micro_mae"] < baseline_micro
    macro_nonreg = star_metrics["macro_mae"] < baseline_macro
    positive_result = bool(micro_nonreg and macro_nonreg and
                           (win_fraction == win_fraction) and win_fraction > 0.5)
    clearly_worked = bool(star_metrics["micro_mae"] <= baseline_micro * 0.9)

    result = {
        "interval": cfg["data"]["interval"], "seq_len": seq_len, "seed": args.seed,
        "betas": betas, "select_metric": select_metric, "beta_star": beta_star,
        "baseline_ar": {"micro_mae": baseline_micro, "macro_mae": baseline_macro},
        "beta0_null_op_ok": null_op_ok, "beta0_micro_mae": beta0_micro,
        "test_curve": test_curve,
        "selection_curve_heldout": selection_curve,
        "beta_star_test_metrics": star_metrics,
        "per_scenario_win_fraction": win_fraction,
        "per_scenario_wins": wins, "per_scenario_total": len(common),
        "step_index_ablation": ablation,
        "non_regression": {"micro": micro_nonreg, "macro": macro_nonreg},
        "positive_result": positive_result,
        "clearly_worked_label": clearly_worked,
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "error_mlp_eval.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[done] wrote {out_path}")
    print(f"  beta*={beta_star} test micro_mae={star_metrics['micro_mae']:.8f} "
          f"macro_mae={star_metrics['macro_mae']:.8f} | baseline {baseline_micro:.8f}/{baseline_macro:.8f}")
    print(f"  positive_result={positive_result} win_fraction={win_fraction:.4f}")

    if args.plot or bool(cfg_ev.get("plot_by_default", False)):
        from predict import plot_mean_error_accumulation
        pd_star, td_star = autoregressive_corrected_batched(
            model, mlp, beta_star, test_ds, device=device, num_workers=num_workers)
        plot_mean_error_accumulation(
            pd_star, td_star, C.VARIABLE_NAMES,
            out_dir=os.path.join(out_dir, "error_accumulation_corrected"), reduction="mae")


if __name__ == "__main__":
    main()
