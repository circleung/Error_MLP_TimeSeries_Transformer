"""Before/after ErrorMLP comparison for one continuous variable (default SBO PEX0(17)).

Figures (all on the TEST set, scaled space):
  (1) _scen   : per-scenario trajectories GT vs baseline-AR vs ErrorMLP-corrected for the
                worst baseline scenarios (+ the least-improved among the worst-10)
  (2) _all    : all-scenario spaghetti, baseline vs corrected side-by-side (GT shown faint)
  (3) _errband: per-step |error| median / p99 across scenarios, baseline vs corrected

Corrector = adopted DAgger ErrorMLP at the validation-frozen operating point (beta*, tau*).

Run from src/:
    NONINTERACTIVE=1 python experiments/plot_before_after.py --cell SBO --var "PEX0(17)"
Read-only on data; backbone frozen. Nothing trained.
"""
import os
import sys
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

GT_C = "#000000"        # ground truth
BASE_C = "#D55E00"      # baseline AR (backbone only)
CORR_C = "#0072B2"      # ErrorMLP-corrected (gated)


def save(fig, stem):
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    fig.savefig(stem + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote {stem}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="SBO")
    ap.add_argument("--var", default="PEX0(17)")
    ap.add_argument("--model-file", default="error_mlp_dagger.pt")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.092902)
    ap.add_argument("--n-worst", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    j = CONTINUOUS_COLS.index(args.var)
    tag = f"{args.cell}_{args.var.replace('(', '').replace(')', '')}"

    cfg = utils.load_config("error_mlp_accident")
    cell = cfg["cells"][args.cell]
    cfg_tr, cfg_data = cfg["training"], cfg["data"]
    device = torch.device(cfg_tr["device"] if torch.cuda.is_available() else "cpu")
    nw = int(cfg_tr["num_workers"]); cb = int(cfg_tr.get("collect_batch", 2048))
    nc = int(cfg_data["num_continuous"])
    snc = float(cell.get("step_norm_const", cfg_data.get("step_norm_const", 1000.0)))

    model, _ = load_frozen_backbone_acc(cell["run_dir"], device)
    ds = AccidentWindowDataset(cell["test_csv"], seq_len=int(cfg_data["seq_len"]),
                               pred_len=int(cfg_data["pred_len"]), cache_dir=cfg_data.get("cache_dir"))
    base, true = autoregressive_corrected_batched_acc(
        model, None, 0.0, ds, ds.num_controls, snc, device=device, num_continuous=nc,
        collect_batch=cb, num_workers=nw)
    ptc = torch.load(os.path.join(cfg["out_root"], args.cell, args.model_file), map_location="cpu")
    mlp = ErrorMLP(in_dim=int(ptc["in_dim"]), **cfg["error_mlp"])
    mlp.load_state_dict(ptc["state_dict"]); mlp = mlp.to(device).eval()
    corr, _ = gated_corrected_rollout_acc(
        model, mlp, args.beta, args.tau, ds, ds.num_controls, snc, device=device,
        gate_on="pred", num_continuous=nc, collect_batch=cb, num_workers=nw)

    sids = sorted(true)
    L = min(len(true[s]) for s in sids)
    Y = np.stack([np.asarray(true[s], float)[:L, j] for s in sids])
    B = np.stack([np.asarray(base[s], float)[:L, j] for s in sids])
    C = np.stack([np.asarray(corr[s], float)[:L, j] for s in sids])
    AB, AC = np.abs(B - Y), np.abs(C - Y)
    sb, sc = AB.mean(1), AC.mean(1)
    n, x = len(sids), np.arange(L)
    print(f"[{args.cell}] {args.var} n={n} L={L} MAE {AB.mean():.5f}->{AC.mean():.5f} "
          f"p99 {np.percentile(AB, 99):.4f}->{np.percentile(AC, 99):.4f}")

    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.family": "sans-serif",
                         "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    lo = min(Y.min(), B.min(), C.min()); hi = max(Y.max(), B.max(), C.max())
    pad = 0.03 * (hi - lo)
    os.makedirs(FIG_DIR, exist_ok=True)

    # (1) worst baseline scenarios + least-improved among the worst-10
    worst10 = np.argsort(sb)[::-1][:10]
    pick = list(worst10[:args.n_worst])
    least = max(worst10, key=lambda i: sc[i] / sb[i])
    if least not in pick:
        pick.append(least)
    ncol = 3; nrow = int(np.ceil(len(pick) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 3.0 * nrow), sharex=True, squeeze=False)
    for ax, i in zip(axes.flat, pick):
        ax.plot(x, Y[i], color=GT_C, lw=1.6, label="Ground truth", zorder=3)
        ax.plot(x, B[i], color=BASE_C, lw=1.2, ls="--", label="Baseline AR", zorder=2)
        ax.plot(x, C[i], color=CORR_C, lw=1.2, label="+ ErrorMLP (gated)", zorder=2)
        note = "  (least improved)" if i == least and i not in worst10[:args.n_worst] else ""
        ax.set_title(f"scenario {sids[i]}{note}\nMAE {sb[i]:.3f} → {sc[i]:.3f}", fontsize=9)
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(alpha=0.3, linewidth=0.6); ax.set_axisbelow(True); ax.margins(x=0.01)
    for ax in axes.flat[len(pick):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("rollout step")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"{args.var} (scaled)")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{args.cell} {args.var}: worst baseline test scenarios — before vs after ErrorMLP "
                 f"(β*={args.beta}, τ* frozen)", y=1.06, fontsize=10)
    fig.tight_layout()
    save(fig, os.path.join(FIG_DIR, f"fig_ba_{tag}_scen"))

    # (2) all-scenario spaghetti: baseline | corrected, GT median overlaid
    gt_med = np.median(Y, axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    for ax, P, col, name in ((axes[0], B, BASE_C, "Baseline AR"),
                             (axes[1], C, CORR_C, "+ ErrorMLP (gated)")):
        for y in P:
            ax.plot(x, y, color=col, lw=0.4, alpha=args.alpha, zorder=1)
        ax.plot(x, np.median(P, axis=0), color=col, lw=1.8, zorder=3, label=f"{name} median")
        ax.plot(x, gt_med, color=GT_C, lw=1.4, ls="--", zorder=4, label="Ground-truth median")
        ax.set_title(f"{name} — all {n} test scenarios", fontsize=10)
        ax.set_xlabel("rollout step"); ax.set_ylim(lo - pad, hi + pad)
        ax.grid(alpha=0.3, linewidth=0.6); ax.set_axisbelow(True); ax.margins(x=0.01)
        ax.legend(frameon=False, loc="best")
    axes[0].set_ylabel(f"{args.var} (scaled)")
    fig.tight_layout()
    save(fig, os.path.join(FIG_DIR, f"fig_ba_{tag}_all"))

    # (3) per-step |error| across scenarios: median & p99, baseline vs corrected
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for A, col, name in ((AB, BASE_C, "Baseline AR"), (AC, CORR_C, "+ ErrorMLP")):
        ax.plot(x, np.percentile(A, 99, axis=0), color=col, lw=1.6, label=f"{name} p99")
        ax.plot(x, np.median(A, axis=0), color=col, lw=1.2, ls="--", label=f"{name} median")
    ax.set_xlabel("rollout step"); ax.set_ylabel(f"|prediction − GT|  {args.var} (scaled)")
    ax.grid(alpha=0.3, linewidth=0.6); ax.set_axisbelow(True); ax.margins(x=0.01)
    ax.set_title(f"{args.cell} {args.var}: per-step error across {n} test scenarios", fontsize=10)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    fig.tight_layout()
    save(fig, os.path.join(FIG_DIR, f"fig_ba_{tag}_errband"))


if __name__ == "__main__":
    main()
