"""Evaluate a trained per-accident-type ErrorMLP AR correction.

ADDITIVE mirror of eval_error_mlp.py for the seq50 / variable-control cells.
Pipeline:
  1. Beta selection: for each beta, run a FULL corrected AR rollout restricted to
     the persisted held-out scenarios; pick beta* = argmin eval.select_metric.
  2. TEST reporting: for every beta, full corrected AR on the cell's TEST dataset.
     Per-cell BASELINE = the beta=0 TEST point (exact null-op == uncorrected AR).
  3. Per-scenario win fraction at beta* vs beta=0 (baseline).
  4. Step-index ablation at beta* (step_norm zeroed).
  5. Write <out_root>/<cell>/error_mlp_eval.json.

Usage (from src/):
    NONINTERACTIVE=1 python experiments/eval_error_mlp_acc.py --cell SBO
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
    in_dim_for,
)
from predict_batched import compute_micro_macro


def per_scenario_mae(predictions_dict, true_dict):
    """Per-scenario MAE (mean over 10 vars, then mean over the scenario's steps),
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
    ap.add_argument("--cell", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--betas", default=None, help="comma list, overrides config")
    args = ap.parse_args()

    cfg = utils.load_config("error_mlp_accident")
    cells = cfg["cells"]
    assert args.cell in cells, f"--cell {args.cell} not in {list(cells)}"
    cell = cells[args.cell]
    cfg_tr = cfg["training"]
    cfg_ev = cfg["eval"]
    cfg_data = cfg["data"]
    device = torch.device(cfg_tr["device"] if torch.cuda.is_available() else "cpu")

    betas = ([float(b) for b in args.betas.split(",")] if args.betas
             else [float(b) for b in cfg_ev["betas"]])
    select_metric = cfg_ev["select_metric"]
    num_workers = int(cfg_tr["num_workers"])
    collect_batch = int(cfg_tr.get("collect_batch", 2048))

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

    with open(os.path.join(out_dir, "heldout_scenarios.json")) as f:
        heldout_ids = [int(s) for s in json.load(f)["heldout_scenarios"]]

    seq_len = int(cfg_data["seq_len"])
    pred_len = int(cfg_data["pred_len"])
    cache_dir = cfg_data.get("cache_dir")
    nc = int(cfg_data["num_continuous"])

    def roll(ds, beta, restrict=None, step_scale=1.0):
        return autoregressive_corrected_batched_acc(
            model, mlp, beta, ds, num_controls, step_norm_const, device=device,
            num_continuous=nc, collect_batch=collect_batch, num_workers=num_workers,
            restrict_scenarios=restrict, step_norm_scale=step_scale)

    # ---- 1. Beta selection on held-out (full corrected AR over TRAIN, restricted) ----
    train_ds = AccidentWindowDataset(cell["train_csv"], seq_len=seq_len,
                                     pred_len=pred_len, cache_dir=cache_dir)
    assert train_ds.num_controls == num_controls and train_ds.input_size == input_size
    selection_curve = {}
    for beta in betas:
        pd_, td_ = roll(train_ds, beta, restrict=heldout_ids)
        m = compute_micro_macro(pd_, td_)
        selection_curve[f"{beta}"] = m
        print(f"[{args.cell}][select] beta={beta} heldout {select_metric}={m[select_metric]:.6f}")
    beta_star = min(betas, key=lambda b: selection_curve[f"{b}"][select_metric])
    print(f"[{args.cell}][select] beta* = {beta_star}")

    # ---- 2. TEST reporting per beta ----
    test_ds = AccidentWindowDataset(cell["test_csv"], seq_len=seq_len,
                                    pred_len=pred_len, cache_dir=cache_dir)
    test_curve = {}
    per_scen_by_beta = {}
    for beta in betas:
        pd_, td_ = roll(test_ds, beta)
        m = compute_micro_macro(pd_, td_)
        test_curve[f"{beta}"] = m
        per_scen_by_beta[beta] = per_scenario_mae(pd_, td_)
        print(f"[{args.cell}][test] beta={beta} micro_mae={m['micro_mae']:.8f} "
              f"macro_mae={m['macro_mae']:.8f}")

    # Per-cell BASELINE = the beta=0 TEST point (exact null-op == uncorrected AR).
    if 0.0 in [float(b) for b in betas]:
        base_metrics = test_curve["0.0"]
        base_per_scen = per_scen_by_beta[0.0]
    else:
        pd0, td0 = roll(test_ds, 0.0)
        base_metrics = compute_micro_macro(pd0, td0)
        base_per_scen = per_scenario_mae(pd0, td0)
    baseline_micro = float(base_metrics["micro_mae"])
    baseline_macro = float(base_metrics["macro_mae"])

    # ---- 3. per-scenario win fraction at beta* (vs beta=0 baseline) ----
    star_per_scen = per_scen_by_beta[beta_star]
    common = sorted(set(star_per_scen) & set(base_per_scen))
    wins = sum(1 for s in common if star_per_scen[s] < base_per_scen[s])
    win_fraction = wins / len(common) if common else float("nan")
    print(f"[{args.cell}][win] beta*={beta_star} win_fraction={win_fraction:.4f} "
          f"({wins}/{len(common)})")

    # ---- 4. step-index ablation at beta* on TEST (step_norm zeroed) ----
    ablation = None
    if bool(cfg_ev.get("step_index_ablation", False)) and beta_star != 0.0:
        pda, tda = roll(test_ds, beta_star, step_scale=0.0)
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
        print(f"[{args.cell}][ablation] step_norm=0 micro_mae={abl_m['micro_mae']:.8f} "
              f"(delta {ablation['delta_vs_beta_star']['micro_mae']:+.2e})")

    star_metrics = test_curve[f"{beta_star}"]
    micro_nonreg = star_metrics["micro_mae"] < baseline_micro
    macro_nonreg = star_metrics["macro_mae"] < baseline_macro
    rel_reduction = (100.0 * (baseline_micro - star_metrics["micro_mae"]) / baseline_micro
                     if baseline_micro > 0 else float("nan"))
    positive_result = bool(micro_nonreg and macro_nonreg and
                           (win_fraction == win_fraction) and win_fraction > 0.5)
    clearly_worked = bool(star_metrics["micro_mae"] <= baseline_micro * 0.9)

    result = {
        "cell": args.cell, "seq_len": seq_len, "seed": args.seed,
        "num_controls": num_controls, "in_dim": in_dim,
        "step_norm_const": step_norm_const,
        "betas": betas, "select_metric": select_metric, "beta_star": beta_star,
        "baseline_ar": {"micro_mae": baseline_micro, "macro_mae": baseline_macro},
        "test_curve": test_curve,
        "selection_curve_heldout": selection_curve,
        "beta_star_test_metrics": star_metrics,
        "rel_reduction_micro_pct": rel_reduction,
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
    print(f"[{args.cell}][done] wrote {out_path}")
    print(f"  beta*={beta_star} test micro_mae={star_metrics['micro_mae']:.8f} "
          f"macro_mae={star_metrics['macro_mae']:.8f} | baseline "
          f"{baseline_micro:.8f}/{baseline_macro:.8f} | rel_red={rel_reduction:.2f}%")
    print(f"  positive_result={positive_result} win_fraction={win_fraction:.4f}")


if __name__ == "__main__":
    main()
