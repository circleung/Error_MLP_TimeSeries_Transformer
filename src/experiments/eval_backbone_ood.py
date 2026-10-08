"""OOD / compounding-ratio eval tool for a per-cell backbone run_dir (Plan #4, D3).

Builds ENTIRELY from committed primitives. For a given run_dir + a FROZEN scenario
set it computes, on that same set (identically for B0 and any NEW backbone):
  (a) single-step teacher-forced MSE + MAE (.eval(): model(true_window) vs next true
      continuous, one step from every window);
  (b) AR beta=0 metrics via autoregressive_corrected_batched_acc(error_mlp=None,
      beta=0) (exact null-op) -> compute_micro_macro (micro-MAE) + step_err_from_dicts
      / tail_metrics (p99, per-step MAE);
  (c) CR = AR-MAE / single-step-MAE (the identifiable compounding ratio, Plan D4).

Frozen fixed test-tail (B0-ranked, Plan D3): with --make_fixed_tail, run B0 open-loop
(beta=0) on TEST, rank per-scenario baseline error, take the top --tail_frac, and
persist the scenario ids to --out. Score BOTH backbones on that same frozen set later
via --fixed_set.

Self-check: asserts beta=0 is a byte-identical null-op (error_mlp=None vs a random
ErrorMLP at beta=0 give identical AR trajectories).

Usage (from src/, NONINTERACTIVE=1):
  # build the frozen B0-ranked test tail once (from the B0 run_dir):
  python experiments/eval_backbone_ood.py --cell SBO --run_dir <B0_run_dir> \
      --make_fixed_tail --tail_frac 0.10 --out results/arbb/fixed_test_tail_SBO.json
  # score a backbone on a frozen set (in-dist gate set or the fixed test tail):
  python experiments/eval_backbone_ood.py --cell SBO --run_dir <run_dir> \
      --fixed_set results/arbb/fixed_test_tail_SBO.json --split test
"""
import os
import sys
import json
import math
import argparse

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

os.environ.setdefault("NONINTERACTIVE", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader

import utils
from accident_dataset import AccidentWindowDataset
from models.error_mlp import ErrorMLP
from error_rollout_acc import (
    load_frozen_backbone_acc,
    autoregressive_corrected_batched_acc,
    in_dim_for,
)
from predict_batched import compute_micro_macro
from tail_analysis_acc import step_err_from_dicts, tail_metrics


@torch.inference_mode()
def single_step_tf_metrics(model, dataset, device, num_continuous=10,
                           batch_size=4096, num_workers=4, restrict=None):
    """Single-step teacher-forced MSE + MAE over windows (optionally restricted to a
    scenario set). MSE = mean over all elements of (pred-cy)^2 (== nn.MSELoss, the
    crown-jewel metric). MAE = mean over all elements of |pred-cy| (== the per-step
    MAE unit pooled over windows, comparable to AR micro-MAE)."""
    model = model.to(device=device, dtype=torch.float32).eval()
    restrict = None if restrict is None else set(int(s) for s in restrict)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    se_sum, ae_sum, n_elem = 0.0, 0.0, 0
    for batch in dl:
        pv = batch["past_values"].to(device, torch.float32)
        cy = batch["continuous_y"].to(device, torch.float32)
        scn = batch["scenario_id"].numpy().astype(np.int64)
        out = model(pv)                                       # [B, nc]
        if restrict is not None:
            keep = torch.from_numpy(np.array([s in restrict for s in scn])).to(device)
            if not bool(keep.any()):
                continue
            out = out[keep]
            cy = cy[keep]
        diff = out - cy
        se_sum += float((diff * diff).sum().item())
        ae_sum += float(diff.abs().sum().item())
        n_elem += int(diff.numel())
    if n_elem == 0:
        return {"mse": float("nan"), "mae": float("nan"), "n_elem": 0}
    return {"mse": se_sum / n_elem, "mae": ae_sum / n_elem, "n_elem": n_elem}


def per_scenario_baseline_err(predictions_dict, true_dict):
    """Per-scenario open-loop mean-abs error keyed by scenario id (for tail ranking)."""
    out = {}
    for s in predictions_dict:
        if not predictions_dict[s]:
            continue
        P = np.vstack([np.ravel(p) for p in predictions_dict[s]]).astype(np.float64)
        Y = np.vstack([np.ravel(y) for y in true_dict[s]]).astype(np.float64)
        out[int(s)] = float(np.mean(np.abs(P - Y)))
    return out


def ar_beta0_metrics(model, dataset, num_controls, step_norm_const, device,
                     num_continuous=10, collect_batch=2048, num_workers=4, restrict=None):
    """AR beta=0 (exact null-op) -> (micro-MAE, tail metrics incl p99, per-step MAE,
    predictions_dict, true_dict)."""
    pd_, td_ = autoregressive_corrected_batched_acc(
        model, None, 0.0, dataset, num_controls, step_norm_const, device=device,
        num_continuous=num_continuous, collect_batch=collect_batch,
        num_workers=num_workers, restrict_scenarios=restrict)
    mm = compute_micro_macro(pd_, td_)
    se, ps = step_err_from_dicts(pd_, td_)
    tm = tail_metrics(se, ps)
    return mm, tm, pd_, td_


def assert_beta0_null_op(model, dataset, num_controls, step_norm_const, device, in_dim,
                         num_continuous=10, collect_batch=2048, num_workers=4, restrict=None):
    """beta=0 must be a byte-identical null-op: error_mlp=None vs a random ErrorMLP at
    beta=0 produce identical AR trajectories (the correction term is skipped)."""
    pd_none, td_none = autoregressive_corrected_batched_acc(
        model, None, 0.0, dataset, num_controls, step_norm_const, device=device,
        num_continuous=num_continuous, collect_batch=collect_batch,
        num_workers=num_workers, restrict_scenarios=restrict)
    torch.manual_seed(0)
    dummy = ErrorMLP(in_dim=in_dim).to(device).eval()
    pd_mlp, td_mlp = autoregressive_corrected_batched_acc(
        model, dummy, 0.0, dataset, num_controls, step_norm_const, device=device,
        num_continuous=num_continuous, collect_batch=collect_batch,
        num_workers=num_workers, restrict_scenarios=restrict)
    max_abs = 0.0
    for s in pd_none:
        A = np.vstack([np.ravel(p) for p in pd_none[s]])
        B = np.vstack([np.ravel(p) for p in pd_mlp[s]])
        max_abs = max(max_abs, float(np.max(np.abs(A - B))) if A.size else 0.0)
    assert max_abs == 0.0, f"beta=0 null-op broken: max|diff|={max_abs}"
    return max_abs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--run_dir", required=True,
                    help="Backbone run_dir (has config_used.yaml + ckpt subdir).")
    ap.add_argument("--config", default="error_mlp_accident",
                    help="Config providing cells/data/step_norm_const (default error_mlp_accident).")
    ap.add_argument("--split", choices=["train", "test"], default="test",
                    help="Which csv the scenario set / tail ranking lives on.")
    ap.add_argument("--fixed_set", type=str, default=None,
                    help="JSON list (or {'scenarios':[...]}) of scenario ids to score on.")
    ap.add_argument("--make_fixed_tail", action="store_true",
                    help="Build a B0-ranked fixed tail from --split csv and write --out.")
    ap.add_argument("--tail_frac", type=float, default=0.10)
    ap.add_argument("--out", type=str, default=None,
                    help="Where to write the fixed-tail JSON (with --make_fixed_tail).")
    ap.add_argument("--results_out", type=str, default=None,
                    help="Where to write the eval-results JSON (default results/arbb/<cell>/).")
    ap.add_argument("--max-scenarios", type=int, default=None, help="SMOKE ONLY cap.")
    args = ap.parse_args()

    cfg = utils.load_config(args.config)
    cells = cfg["cells"]
    assert args.cell in cells, f"--cell {args.cell} not in {list(cells)}"
    cell = cells[args.cell]
    cfg_tr, cfg_data = cfg["training"], cfg["data"]
    device = torch.device(cfg_tr["device"] if torch.cuda.is_available() else "cpu")
    num_workers = int(cfg_tr["num_workers"])
    collect_batch = int(cfg_tr.get("collect_batch", 2048))
    seq_len, pred_len = int(cfg_data["seq_len"]), int(cfg_data["pred_len"])
    cache_dir = cfg_data.get("cache_dir")
    nc = int(cfg_data["num_continuous"])
    step_norm_const = float(cell["step_norm_const"])

    model, input_size = load_frozen_backbone_acc(args.run_dir, device)
    csv_path = cell["test_csv"] if args.split == "test" else cell["train_csv"]
    ds = AccidentWindowDataset(csv_path, seq_len=seq_len, pred_len=pred_len,
                               cache_dir=cache_dir, max_scenarios=args.max_scenarios)
    num_controls = ds.num_controls
    in_dim = in_dim_for(num_controls)
    assert ds.input_size == input_size, (
        f"csv input_size {ds.input_size} != backbone input_size {input_size}")

    # ---- build the frozen B0-ranked test tail (and exit) ----
    if args.make_fixed_tail:
        assert args.out is not None, "--make_fixed_tail requires --out"
        _mm, _tm, pd_, td_ = ar_beta0_metrics(
            model, ds, num_controls, step_norm_const, device, nc, collect_batch, num_workers)
        per_scen = per_scenario_baseline_err(pd_, td_)
        ranked = sorted(per_scen.items(), key=lambda kv: kv[1], reverse=True)
        k = max(1, int(math.ceil(args.tail_frac * len(ranked))))
        tail_ids = sorted(int(s) for s, _ in ranked[:k])
        payload = {"cell": args.cell, "split": args.split, "tail_frac": args.tail_frac,
                   "run_dir": args.run_dir, "n_total": len(ranked), "n_tail": len(tail_ids),
                   "scenarios": tail_ids,
                   "per_scenario_err": {str(s): per_scen[s] for s in tail_ids}}
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[{args.cell}][fixed_tail] wrote {len(tail_ids)}/{len(ranked)} scenarios "
              f"(top {args.tail_frac:.0%}) -> {args.out}")
        return

    # ---- resolve the frozen scenario set to score on ----
    restrict = None
    fixed_tag = "all"
    if args.fixed_set is not None:
        with open(args.fixed_set) as f:
            obj = json.load(f)
        restrict = obj["scenarios"] if isinstance(obj, dict) else list(obj)
        restrict = [int(s) for s in restrict]
        fixed_tag = os.path.splitext(os.path.basename(args.fixed_set))[0]
        print(f"[{args.cell}] scoring on {len(restrict)} frozen scenarios from {args.fixed_set}")

    # ---- (a) single-step TF, (b) AR beta0, (c) CR ----
    ss = single_step_tf_metrics(model, ds, device, num_continuous=nc,
                                batch_size=int(cfg_tr.get("batch_size", 4096)),
                                num_workers=num_workers, restrict=restrict)
    mm, tm, _pd, _td = ar_beta0_metrics(model, ds, num_controls, step_norm_const, device,
                                        nc, collect_batch, num_workers, restrict=restrict)
    cr = (mm["micro_mae"] / ss["mae"]) if ss["mae"] and ss["mae"] > 0 else float("nan")

    # ---- self-check: beta=0 byte-identical null-op ----
    null_max = assert_beta0_null_op(model, ds, num_controls, step_norm_const, device,
                                    in_dim, nc, collect_batch, num_workers, restrict=restrict)

    result = {
        "cell": args.cell, "run_dir": args.run_dir, "split": args.split,
        "fixed_set": args.fixed_set, "fixed_tag": fixed_tag,
        "n_scored_scenarios": (len(restrict) if restrict is not None else tm["n_scen"]),
        "num_controls": num_controls, "in_dim": in_dim, "step_norm_const": step_norm_const,
        "single_step_tf": ss,
        "ar_beta0": {"micro_mae": mm["micro_mae"], "micro_rmse": mm["micro_rmse"],
                     "macro_mae": mm["macro_mae"], "p99": tm["p99"], "p95": tm["p95"],
                     "p999": tm["p999"], "max": tm["max"], "mean_step_mae": tm["mean"],
                     "n_steps": tm["n_steps"], "n_scen": tm["n_scen"]},
        "compounding_ratio": cr,
        "null_op_max_abs_diff": null_max,
    }
    out_path = args.results_out
    if out_path is None:
        rdir = os.path.join(SRC_DIR, "results", "arbb", args.cell)
        os.makedirs(rdir, exist_ok=True)
        out_path = os.path.join(rdir, f"ood_eval_{args.split}_{fixed_tag}.json")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{args.cell}] single-step TF: MSE={ss['mse']:.3e} MAE={ss['mae']:.6f} "
          f"(n_elem={ss['n_elem']})")
    print(f"[{args.cell}] AR beta0: micro-MAE={mm['micro_mae']:.6f} p99={tm['p99']:.6f} "
          f"(n_steps={tm['n_steps']}, n_scen={tm['n_scen']})")
    print(f"[{args.cell}] CR = AR-MAE/single-step-MAE = {cr:.2f}")
    print(f"[{args.cell}] null-op self-check max|diff|={null_max:.1e} (must be 0)")
    print(f"[{args.cell}] wrote {out_path}")


if __name__ == "__main__":
    main()
