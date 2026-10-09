"""4-way TEST bar charts (SBO): baseline / baseline+ErrorMLP / SS / SS+ErrorMLP.

Reads backbones, correctors and the validation-frozen operating points from the
`ss_scratch_4way_eval.py` JSON, re-runs the four rollouts and plots MAE / RMSE / p99
(10-var, ABC-Transformer definitions; see plot_cell_metrics.metrics) plus the same
for one variable. Writes figures/fig_4way_metrics_<cell>.png and a JSON next to it.

Run from src/:
    NONINTERACTIVE=1 python experiments/plot_4way_metrics.py \
        --eval-json experiments/results/ss_scratch_sbo_4way.json
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
from error_rollout_acc import load_frozen_backbone_acc, gated_corrected_rollout_acc

ARMS = [("arm1", "Baseline", "#D55E00"), ("arm2", "Baseline + Error-MLP", "#E69F00"),
        ("arm3", "SS backbone", "#009E73"), ("arm4", "SS + Error-MLP", "#0072B2")]


def metrics(pred, true, j=None):
    """Macro MAE / RMSE and pooled per-step p99. All 10 variables: RMSE with the root
    inside the time average (ABC-Transformer). Single variable j: per-scenario
    sqrt(mean_t err^2) (a per-step root over one channel would just equal MAE)."""
    mae, rmse, steps = [], [], []
    for s in sorted(true):
        e = np.asarray(pred[s], np.float64)[:, :10] - np.asarray(true[s], np.float64)[:, :10]
        if j is not None:
            e = e[:, j:j + 1]
        mae.append(np.abs(e).mean())
        rmse.append(np.sqrt((e ** 2).mean()) if j is not None else np.sqrt((e ** 2).mean(axis=1)).mean())
        steps.append(np.abs(e).mean(axis=1))
    return dict(mae=float(np.mean(mae)), rmse=float(np.mean(rmse)),
                p99=float(np.percentile(np.concatenate(steps), 99)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", required=True)
    ap.add_argument("--var", default="PEX0(17)")
    args = ap.parse_args()
    ev = json.load(open(args.eval_json))
    name = ev["cell"]; j = CONTINUOUS_COLS.index(args.var)

    cfg = utils.load_config("error_mlp_accident")
    cell, cfg_tr, cfg_data = cfg["cells"][name], cfg["training"], cfg["data"]
    device = torch.device(cfg_tr["device"] if torch.cuda.is_available() else "cpu")
    nw = int(cfg_tr["num_workers"]); cb = int(cfg_tr.get("collect_batch", 2048))
    nc = int(cfg_data["num_continuous"])
    snc = float(cell.get("step_norm_const", cfg_data.get("step_norm_const", 1000.0)))
    ds = AccidentWindowDataset(cell["test_csv"], seq_len=int(cfg_data["seq_len"]),
                               pred_len=int(cfg_data["pred_len"]), cache_dir=cfg_data.get("cache_dir"))

    res = {}
    for key, label, _ in ARMS:
        arm = ev["arms"][key]
        model, _ = load_frozen_backbone_acc(arm["run_dir"], device)
        mlp, beta, tau = None, 0.0, float("inf")
        if arm.get("mlp") and arm.get("op"):
            ptc = torch.load(arm["mlp"], map_location="cpu")
            mlp = ErrorMLP(in_dim=int(ptc["in_dim"]), **cfg["error_mlp"])
            mlp.load_state_dict(ptc["state_dict"]); mlp = mlp.to(device).eval()
            beta, tau = float(arm["op"]["beta_star"]), float(arm["op"]["tau_star"])
        pred, true = gated_corrected_rollout_acc(
            model, mlp, beta, tau, ds, ds.num_controls, snc, device=device, gate_on="pred",
            num_continuous=nc, collect_batch=cb, num_workers=nw)
        res[key] = dict(label=label, beta=beta, tau=tau, all=metrics(pred, true), var=metrics(pred, true, j))
        print(f"[{key}] {label}: {res[key]['all']} | {args.var}: {res[key]['var']}", flush=True)

    os.makedirs(FIG_DIR, exist_ok=True)
    stem = os.path.join(FIG_DIR, f"fig_4way_metrics_{name}")
    with open(stem + ".json", "w") as f:
        json.dump(dict(cell=name, var=args.var, arms=res), f, indent=1)

    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.family": "sans-serif",
                         "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
    for row, (scope, title) in enumerate((("all", "10 variables"), ("var", args.var))):
        for ax, m in zip(axes[row], ("mae", "rmse", "p99")):
            vals = [res[k][scope][m] for k, _, _ in ARMS]
            ax.bar(np.arange(4), vals, 0.7, color=[c for _, _, c in ARMS], edgecolor="white", linewidth=1)
            for i, v in enumerate(vals):
                note = f"{v:.4f}" if i == 0 else f"{v:.4f}\n({100 * (v - vals[0]) / vals[0]:+.1f}%)"
                ax.text(i, v * 1.02, note, ha="center", va="bottom", fontsize=7.5)
            ax.set_ylim(0, max(vals) * 1.25); ax.set_xticks([])
            ax.set_title(f"{title}: {m.upper()}", fontsize=10)
            ax.grid(axis="y", alpha=0.3, linewidth=0.6); ax.set_axisbelow(True)
        axes[row][0].set_ylabel("AR rollout error (scaled)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in ARMS]
    fig.legend(handles, [l for _, l, _ in ARMS], loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(f"{name} test set: baseline vs scheduled-sampling backbone, with / without Error-MLP",
                 y=1.07, fontsize=10)
    fig.tight_layout()
    fig.savefig(stem + ".pdf", bbox_inches="tight"); fig.savefig(stem + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig); print(f"[done] wrote {stem}.png")


if __name__ == "__main__":
    main()
