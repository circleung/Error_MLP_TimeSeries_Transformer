"""4-way (baseline / baseline+ErrorMLP / SS / SS+ErrorMLP) SBO TEST figures, in the
plot_cell_metrics / plot_before_after style. Reads the outputs of ss_scratch_4way_eval.py
(--out-json and --out-npz); nothing is re-rolled.

Figures (TEST, scaled space):
  fig_4way_SBO_metrics       : step-level MAE / RMSE (ABC macro) / p99 / max bars, % vs arm1
  fig_4way_SBO_PEX017_errband: per-step |error| median / p99 across scenarios, 4 arms
  fig_4way_SBO_PEX017_all    : all-scenario PEX0(17) spaghetti per arm, GT median overlaid

Run from src/:
  python experiments/plot_4way_pex.py --json <..>/ss_scratch_sbo_4way.json --npz <..>/pex_4way.npz
"""
import os
import sys
import json
import argparse

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(SRC_DIR)
FIG_DIR = os.path.join(REPO_ROOT, "figures")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GT_C = "#000000"
ARMS = [("arm1", "Baseline AR", "#D55E00", "--"),
        ("arm2", "Baseline + Error-MLP", "#E69F00", "-"),
        ("arm3", "SS backbone AR", "#009E73", "--"),
        ("arm4", "SS + Error-MLP", "#0072B2", "-")]


def save(fig, stem):
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    fig.savefig(stem + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote {stem}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    res = json.load(open(args.json))
    d = np.load(args.npz)
    cell = res["cell"]
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.family": "sans-serif",
                         "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    os.makedirs(FIG_DIR, exist_ok=True)

    # (1) step-level metric bars, one panel per metric (one unit each)
    panels = [("macro_mae", "MAE"), ("macro_rmse", "RMSE"), ("p99", "step p99"), ("max", "step max")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, (key, name) in zip(axes, panels):
        vals = [res["arms"][a]["step"][key] for a, *_ in ARMS]
        x = np.arange(len(ARMS))
        ax.bar(x, vals, 0.7, color=[c for *_, c, _ in ARMS], edgecolor="white", linewidth=1)
        for i, v in enumerate(vals):
            lab = f"{v:.4f}" if i == 0 else f"{v:.4f}\n{100.0 * (v - vals[0]) / vals[0]:+.1f}%"
            ax.text(x[i], v * 1.02, lab, ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x); ax.set_xticklabels([l for _, l, *_ in ARMS], rotation=25, ha="right")
        ax.set_title(name, fontsize=10); ax.set_ylim(0, max(vals) * 1.25)
        ax.grid(axis="y", alpha=0.3, linewidth=0.6); ax.set_axisbelow(True)
    axes[0].set_ylabel("AR rollout error (scaled)")
    fig.suptitle(f"{cell}: baseline vs scheduled-sampling backbone, ± Error-MLP "
                 f"(test, {res['n_test_scen']} scenarios; % vs baseline AR)", y=1.03, fontsize=10)
    fig.tight_layout()
    save(fig, os.path.join(FIG_DIR, f"fig_4way_{cell}_metrics"))

    # (2) PEX0(17) per-step |error| across scenarios: median & p99
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for key, name, col, _ in ARMS:
        A = d[key]
        x = np.arange(A.shape[1])
        ax.plot(x, np.nanpercentile(A, 99, axis=0), color=col, lw=1.6, label=f"{name} p99")
        ax.plot(x, np.nanmedian(A, axis=0), color=col, lw=1.1, ls=":", label=f"{name} median")
    ax.set_xlabel("rollout step"); ax.set_ylabel("|prediction − GT|  PEX0(17) (scaled)")
    ax.grid(alpha=0.3, linewidth=0.6); ax.set_axisbelow(True); ax.margins(x=0.01)
    ax.set_title(f"{cell} PEX0(17): per-step error across {d['arm1'].shape[0]} test scenarios",
                 fontsize=10)
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=7.5)
    fig.tight_layout()
    save(fig, os.path.join(FIG_DIR, f"fig_4way_{cell}_PEX017_errband"))

    # (3) all-scenario spaghetti per arm, GT median overlaid
    Y = d["true"]
    gt_med = np.nanmedian(Y, axis=0)
    lo = np.nanmin([np.nanmin(Y)] + [np.nanmin(d[f"{a}_pred"]) for a, *_ in ARMS])
    hi = np.nanmax([np.nanmax(Y)] + [np.nanmax(d[f"{a}_pred"]) for a, *_ in ARMS])
    pad = 0.03 * (hi - lo)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.0), sharex=True, sharey=True)
    for ax, (key, name, col, _) in zip(axes.flat, ARMS):
        P = d[f"{key}_pred"]
        x = np.arange(P.shape[1])
        for y in P:
            ax.plot(x, y, color=col, lw=0.4, alpha=args.alpha, zorder=1)
        ax.plot(x, np.nanmedian(P, axis=0), color=col, lw=1.8, zorder=3, label=f"{name} median")
        ax.plot(x, gt_med, color=GT_C, lw=1.4, ls="--", zorder=4, label="Ground-truth median")
        pv = res["arms"][key]["per_var"]["PEX0(17)"]
        ax.set_title(f"{name} — MAE {pv['mae']:.4f}, p99 {pv['p99']:.4f}", fontsize=10)
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(alpha=0.3, linewidth=0.6); ax.set_axisbelow(True); ax.margins(x=0.01)
        ax.legend(frameon=False, loc="best")
    for ax in axes[-1]:
        ax.set_xlabel("rollout step")
    for ax in axes[:, 0]:
        ax.set_ylabel("PEX0(17) (scaled)")
    fig.suptitle(f"{cell} PEX0(17): all {P.shape[0]} test scenarios", y=1.01, fontsize=10)
    fig.tight_layout()
    save(fig, os.path.join(FIG_DIR, f"fig_4way_{cell}_PEX017_all"))


if __name__ == "__main__":
    main()
