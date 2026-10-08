"""Baseline vs ErrorMLP-corrected (gated) MAE / RMSE bar charts across cells (TEST).

For each cell: one beta=0 rollout (baseline) and one gated rollout with the adopted
DAgger corrector at the validation-selected, frozen operating point (beta*, tau*) read
from `<out_root>/op_val_select_summary.json`. Metrics follow the ABC-Transformer
definitions (macro over scenarios, 10 continuous channels):
    MAE  = 1/(T*C) * sum_t sum_c |err|
    RMSE = 1/T * sum_t sqrt(1/C * sum_c err^2)      # root inside the time average
plus per-variable MAE. Writes figures/fig_metrics_* and a JSON with the raw numbers.

Run from src/:
    NONINTERACTIVE=1 python experiments/plot_cell_metrics.py
Read-only on data; backbones frozen. Nothing trained.
"""
import os
import sys
import json
import argparse

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(SRC_DIR)
FIG_DIR = os.path.join(REPO_ROOT, "figures")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
os.environ.setdefault("NONINTERACTIVE", "1")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import utils
from accident_dataset import AccidentWindowDataset, CONTINUOUS_COLS
from models.error_mlp import ErrorMLP
from error_rollout_acc import (
    load_frozen_backbone_acc, autoregressive_corrected_batched_acc,
    gated_corrected_rollout_acc,
)

BASE_C = "#D55E00"      # baseline AR (backbone only)
CORR_C = "#0072B2"      # ErrorMLP-corrected (gated)
CELLS = ["SBO", "LLOCA_CSP", "LLOCA_ECSBS", "TLOFW_CSP", "TLOFW_ECSBS"]


def metrics(pred, true, nc):
    """Macro (per-scenario, then averaged) MAE / RMSE and per-variable MAE."""
    mae, rmse, var_abs, var_n = [], [], np.zeros(nc), 0
    for s in sorted(true):
        e = np.asarray(pred[s], np.float64)[:, :nc] - np.asarray(true[s], np.float64)[:, :nc]
        mae.append(np.abs(e).mean())
        rmse.append(np.sqrt((e ** 2).mean(axis=1)).mean())
        var_abs += np.abs(e).sum(axis=0); var_n += len(e)
    return dict(mae=float(np.mean(mae)), rmse=float(np.mean(rmse)),
                var_mae=(var_abs / var_n).tolist(), n=len(mae))


def bars(ax, labels, b, c, ylabel, title):
    x = np.arange(len(labels)); w = 0.38
    ax.bar(x - w / 2, b, w, color=BASE_C, label="Baseline AR", edgecolor="white", linewidth=1)
    ax.bar(x + w / 2, c, w, color=CORR_C, label="+ Error-MLP (gated)", edgecolor="white", linewidth=1)
    for i in range(len(labels)):
        d = 100.0 * (c[i] - b[i]) / b[i]
        ax.text(x[i], max(b[i], c[i]) * 1.03, f"{d:+.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    ax.set_ylim(0, max(max(b), max(c)) * 1.18)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6); ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=CELLS)
    ap.add_argument("--model-file", default="error_mlp_dagger.pt")
    ap.add_argument("--detail-cell", default="SBO", help="cell for the per-variable MAE figure")
    args = ap.parse_args()

    cfg = utils.load_config("error_mlp_accident")
    cfg_tr, cfg_data = cfg["training"], cfg["data"]
    device = torch.device(cfg_tr["device"] if torch.cuda.is_available() else "cpu")
    nw = int(cfg_tr["num_workers"]); cb = int(cfg_tr.get("collect_batch", 2048))
    nc = int(cfg_data["num_continuous"])
    ops = {o["cell"]: o for o in json.load(open(os.path.join(cfg["out_root"], "op_val_select_summary.json")))}

    res = {}
    for name in args.cells:
        cell = cfg["cells"][name]
        snc = float(cell.get("step_norm_const", cfg_data.get("step_norm_const", 1000.0)))
        model, _ = load_frozen_backbone_acc(cell["run_dir"], device)
        ds = AccidentWindowDataset(cell["test_csv"], seq_len=int(cfg_data["seq_len"]),
                                   pred_len=int(cfg_data["pred_len"]), cache_dir=cfg_data.get("cache_dir"))
        base, true = autoregressive_corrected_batched_acc(
            model, None, 0.0, ds, ds.num_controls, snc, device=device, num_continuous=nc,
            collect_batch=cb, num_workers=nw)
        ptc = torch.load(os.path.join(cfg["out_root"], name, args.model_file), map_location="cpu")
        mlp = ErrorMLP(in_dim=int(ptc["in_dim"]), **cfg["error_mlp"])
        mlp.load_state_dict(ptc["state_dict"]); mlp = mlp.to(device).eval()
        beta, tau = float(ops[name]["beta_star"]), float(ops[name]["tau_star"])
        corr, _ = gated_corrected_rollout_acc(
            model, mlp, beta, tau, ds, ds.num_controls, snc, device=device,
            gate_on="pred", num_continuous=nc, collect_batch=cb, num_workers=nw)
        res[name] = dict(beta=beta, tau=tau, baseline=metrics(base, true, nc), corrected=metrics(corr, true, nc))
        b, c = res[name]["baseline"], res[name]["corrected"]
        print(f"[{name}] n={b['n']} MAE {b['mae']:.5f}->{c['mae']:.5f}  RMSE {b['rmse']:.5f}->{c['rmse']:.5f}", flush=True)

    os.makedirs(FIG_DIR, exist_ok=True)
    with open(os.path.join(FIG_DIR, "metrics_baseline_vs_errormlp.json"), "w") as f:
        json.dump(dict(variables=CONTINUOUS_COLS[:nc], cells=res), f, indent=1)

    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.family": "sans-serif",
                         "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    labels = [n.replace("_", "-") for n in args.cells]

    # (1) MAE and RMSE per cell, baseline vs corrected (two panels, one unit each)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, nm in ((axes[0], "mae", "MAE"), (axes[1], "rmse", "RMSE")):
        bars(ax, labels, [res[n]["baseline"][key] for n in args.cells],
             [res[n]["corrected"][key] for n in args.cells], f"AR rollout {nm} (scaled)",
             f"{nm}: baseline vs Error-MLP (test)")
    axes[0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    stem = os.path.join(FIG_DIR, "fig_metrics_mae_rmse")
    fig.savefig(stem + ".pdf", bbox_inches="tight"); fig.savefig(stem + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig); print(f"[done] wrote {stem}.png")

    # (2) per-variable MAE for one cell
    if args.detail_cell in res:
        r = res[args.detail_cell]
        fig, ax = plt.subplots(figsize=(10, 3.8))
        bars(ax, CONTINUOUS_COLS[:nc], r["baseline"]["var_mae"], r["corrected"]["var_mae"],
             "AR rollout MAE (scaled)", f"{args.detail_cell}: per-variable MAE, baseline vs Error-MLP (test)")
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()
        stem = os.path.join(FIG_DIR, f"fig_metrics_pervar_{args.detail_cell}")
        fig.savefig(stem + ".pdf", bbox_inches="tight"); fig.savefig(stem + ".png", dpi=200, bbox_inches="tight")
        plt.close(fig); print(f"[done] wrote {stem}.png")


if __name__ == "__main__":
    main()
