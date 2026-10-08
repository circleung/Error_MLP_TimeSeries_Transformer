"""Prediction-trajectory publication figure: GT vs baseline AR vs gated ErrorMLP-corrected AR.

ADDITIVE. REUSES the already-trained, frozen per-accident-type backbone + the trained
ErrorMLP (NO retrain; the backbone is hard-asserted frozen). Imports the canonical
rollouts from error_rollout_acc WITHOUT modifying them:
  * baseline AR (beta=0)      via autoregressive_corrected_batched_acc(model, None, 0.0, ...)
  * step-gated corrected AR   at the cell's tail-analysis operating point (beta*, tau_pred)
                              via gated_corrected_rollout_acc(..., gate_on='pred').

For the figure annotation ("where did the gate fire?") a byte-faithful MIRROR of the
step-gated rollout (`_gated_rollout_with_fires`) additionally records the per-step gate
fire mask along the corrected trajectory; it is cross-validated for exact equality against
the canonical gated_corrected_rollout_acc before use (assert), so the marked steps are the
real fires that produced the plotted blue line.

Scenario selection: pick ONE representative TAIL scenario near the 90-95th percentile of
per-scenario baseline MAE (a clear but not pathological drift case). Variable selection:
plot the 6 continuous variables with the largest per-variable baseline error in that
scenario, so the correction is visible where it matters.

Writes <repo>/figures/fig6_trajectories_<cell>.{pdf,png} (300-dpi PNG + vector PDF).

Usage (from src/):
    NONINTERACTIVE=1 python experiments/plot_trajectories.py --cell SBO
    NONINTERACTIVE=1 python experiments/plot_trajectories.py --cell SBO --cell TLOFW_CSP
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
import matplotlib.transforms as mtransforms

import utils
from accident_dataset import AccidentWindowDataset, CONTINUOUS_COLS
from models.error_mlp import ErrorMLP
from error_rollout_acc import (
    load_frozen_backbone_acc,
    autoregressive_corrected_batched_acc,
    gated_corrected_rollout_acc,
    build_error_features_acc,
    _collect_pass,
    _pack,
)


# Okabe-Ito palette (matches make_paper_figures.py; colorblind-safe).
GT_C = "#000000"        # ground truth: black solid
BASE_C = "#D55E00"      # baseline AR: vermillion dashed
CORR_C = "#0072B2"      # ErrorMLP-corrected AR: blue solid
RUG_C = "#009E73"       # gate-fired rug ticks: bluish-green
# SAMG-intervention markers (small, on the x-axis): distinct from GT(black)/baseline
# (vermillion)/corrected(blue)/gate(green) -> purple / magenta / gold / brown / slate.
SAMG_COLORS = ["#762A83", "#E7298A", "#E6AB02", "#A6761D", "#5E4FA2", "#666666"]


def samg_edges(test_ds, scenario_id, seq_len, L):
    """When each SAMG control changes state in this scenario (the ±1 signal flips,
    |Δ|>0.5 = the operator-action / intervention point), mapped to the plotted rollout.
    AR step t injects the TRUE control for absolute time t+seq_len, so
    rollout_step = absolute_edge - seq_len; keep only edges inside [0, L].
    Returns [(name, rollout_step, control_index, direction)]."""
    names = list(test_ds.feature_cols[test_ds.num_continuous:])
    m = (test_ds._scenario == scenario_id)
    ctrl = test_ds._feats[m][:, test_ds.num_continuous:]      # [L_full, num_controls]
    out = []
    for k, nm in enumerate(names):
        col = ctrl[:, k]
        d = np.diff(col)
        for e in (np.where(np.abs(d) > 0.5)[0] + 1):
            rs = int(e) - seq_len
            if 0 <= rs <= L:
                out.append((nm, rs, k, "on→off" if col[e] < col[e - 1] else "off→on"))
    return out


def set_style():
    """Publication rcParams (mirrors make_paper_figures.set_style)."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def assert_frozen(model):
    """Hard-assert the backbone carries no trainable params and is in eval mode."""
    n_trainable = sum(int(p.requires_grad) for p in model.parameters())
    assert n_trainable == 0, f"backbone is NOT frozen: {n_trainable} trainable params"
    assert not model.training, "backbone must be in eval() mode"


def per_scenario_mae(predictions_dict, true_dict):
    """Per-scenario MAE (mean over 10 vars then over steps); matches eval_error_mlp_acc."""
    out = {}
    for sc in predictions_dict:
        if not predictions_dict[sc]:
            continue
        P = np.vstack([np.ravel(p) for p in predictions_dict[sc]]).astype(np.float64)
        Y = np.vstack([np.ravel(y) for y in true_dict[sc]]).astype(np.float64)
        out[int(sc)] = float(np.mean(np.abs(P - Y)))
    return out


def _stack(dict_of_lists, sid):
    """[L, 10] float array for one scenario from a predictions/true dict-of-lists."""
    return np.vstack([np.ravel(v) for v in dict_of_lists[sid]]).astype(np.float64)


@torch.inference_mode()
def _gated_rollout_with_fires(model, error_mlp, beta, tau, dataset, num_controls,
                              step_norm_const, device, num_continuous=10,
                              collect_batch=2048, num_workers=4, restrict_scenarios=None):
    """Byte-faithful MIRROR of error_rollout_acc.gated_corrected_rollout_acc (gate_on='pred',
    the non-null-op path: beta!=0, finite tau) that ALSO records, per scenario, the per-step
    boolean gate-fire mask (||e_hat_t||_2 > tau) along the CORRECTED trajectory. Cross-validated
    for exact equality against the canonical function before its fires are trusted.

    Returns (pred_dict[sid]->[L,10], true_dict[sid]->[L,10], fire_dict[sid]->[L] bool)."""
    model = model.to(device=device, dtype=torch.float32).eval()
    error_mlp = error_mlp.to(device=device, dtype=torch.float32).eval()

    order, init_win, cont, ctrl = _collect_pass(
        dataset, collect_batch, num_workers, restrict_scenarios)
    S, lengths, maxL, W, CY, UY = _pack(order, init_win, cont, ctrl, num_continuous, num_controls)
    window = torch.from_numpy(W).to(device)
    UY_t = torch.from_numpy(UY).to(device)
    preds = np.zeros((S, maxL, num_continuous), np.float32)
    fires = np.zeros((S, maxL), dtype=bool)
    tau_t = torch.as_tensor(float(tau), device=device, dtype=torch.float32)

    for t in range(maxL):
        raw = model(window)                                        # [S, 10]
        feats = build_error_features_acc(
            raw, window[:, -1, :num_continuous], UY_t[:, t, :], t, step_norm_const)
        e_hat = error_mlp(feats)                                   # [S, 10]
        score = torch.linalg.vector_norm(e_hat, ord=2, dim=1)      # [S]
        fire = (score > tau_t).unsqueeze(1)                        # [S, 1]
        chosen = torch.where(fire, raw + beta * e_hat, raw)        # gated select
        preds[:, t, :] = chosen.detach().cpu().numpy()
        fires[:, t] = fire.squeeze(1).detach().cpu().numpy()
        next_row = torch.cat([chosen, UY_t[:, t, :]], dim=1)       # [S, input]
        window = torch.cat([window[:, 1:, :], next_row[:, None, :]], dim=1)

    pred_dict, true_dict, fire_dict = {}, {}, {}
    for i, s in enumerate(order):
        L = int(lengths[i])
        pred_dict[int(s)] = preds[i, :L].astype(np.float64).copy()
        true_dict[int(s)] = CY[i, :L].astype(np.float64).copy()
        fire_dict[int(s)] = fires[i, :L].copy()
    return pred_dict, true_dict, fire_dict


def choose_tail_scenario(base_mae, lo_q=0.90, hi_q=0.95):
    """Pick ONE representative tail scenario near the (lo_q, hi_q) band midpoint of the
    per-scenario baseline MAE distribution (clear drift, not pathological max)."""
    sids = sorted(base_mae)
    vals = np.array([base_mae[s] for s in sids], dtype=np.float64)
    lo, hi = np.quantile(vals, lo_q), np.quantile(vals, hi_q)
    mid = np.quantile(vals, 0.5 * (lo_q + hi_q))
    band = [s for s in sids if lo <= base_mae[s] <= hi]
    pool = band if band else sids
    chosen = min(pool, key=lambda s: abs(base_mae[s] - mid))
    pct = 100.0 * float(np.mean(vals <= base_mae[chosen]))
    return int(chosen), pct, float(lo), float(hi)


def choose_worst_scenario(base_mae, rank=0):
    """Pick the rank-th ACTUAL worst tail scenario (rank 0 = the single worst) by
    per-scenario baseline MAE -- the scenarios that drive p99 / worst-10."""
    sids = sorted(base_mae, key=lambda s: base_mae[s], reverse=True)
    rank = max(0, min(int(rank), len(sids) - 1))
    chosen = sids[rank]
    vals = np.array(list(base_mae.values()), dtype=np.float64)
    pct = 100.0 * float(np.mean(vals <= base_mae[chosen]))
    return int(chosen), pct


def choose_best_scenario(base_mae, corr_mae, rank=0, min_base_q=0.60):
    """Pick the rank-th SUCCESS scenario where the corrector helps MOST: largest
    absolute MAE reduction (baseline - corrected), among scenarios whose baseline MAE
    is >= the min_base_q quantile (a real drift to fix -> a clean before/after where
    our method visibly recovers GT). rank 0 = the single most-improved."""
    vals = np.array([base_mae[s] for s in base_mae], dtype=np.float64)
    thr = float(np.quantile(vals, min_base_q))
    cand = [s for s in base_mae if base_mae[s] >= thr and s in corr_mae]
    cand.sort(key=lambda s: base_mae[s] - corr_mae[s], reverse=True)
    rank = max(0, min(int(rank), len(cand) - 1))
    chosen = cand[rank]
    red = 100.0 * (base_mae[chosen] - corr_mae[chosen]) / base_mae[chosen]
    pct = 100.0 * float(np.mean(vals <= base_mae[chosen]))
    return int(chosen), float(red), pct


def run_cell(cell_name, cfg, device, model_file="error_mlp.pt", worst_rank=None,
             best_rank=None, beta_override=None, tau_override=None, var_name=None,
             ylim=None):
    cells = cfg["cells"]
    assert cell_name in cells, f"--cell {cell_name} not in {list(cells)}"
    cell = cells[cell_name]
    cfg_tr, cfg_data = cfg["training"], cfg["data"]
    num_workers = int(cfg_tr["num_workers"])
    collect_batch = int(cfg_tr.get("collect_batch", 2048))
    seq_len = int(cfg_data["seq_len"])
    pred_len = int(cfg_data["pred_len"])
    cache_dir = cfg_data.get("cache_dir")
    nc = int(cfg_data["num_continuous"])
    out_dir = os.path.join(cfg["out_root"], cell_name)

    # --- frozen backbone (NO retrain) ---
    model, input_size = load_frozen_backbone_acc(cell["run_dir"], device)
    assert_frozen(model)

    # --- trained ErrorMLP (model_file: error_mlp.pt one-shot | error_mlp_dagger.pt) ---
    pt = torch.load(os.path.join(out_dir, model_file), map_location="cpu")
    in_dim = int(pt["in_dim"])
    num_controls = int(pt["num_controls"])
    step_norm_const = float(pt["step_norm_const"])
    mlp = ErrorMLP(in_dim=in_dim, **cfg["error_mlp"])
    mlp.load_state_dict(pt["state_dict"])
    mlp = mlp.to(device).eval()

    # --- operating point (beta*, q*, tau_pred) from the matching tail_analysis json ---
    tail_name = ("tail_analysis.json" if model_file == "error_mlp.pt"
                 else f"tail_analysis_{os.path.splitext(model_file)[0]}.json")
    with open(os.path.join(out_dir, tail_name)) as f:
        tail = json.load(f)
    op = tail["operating_point_strict"]
    beta_star = float(op["beta"])
    q_star = float(op["q"])
    tau_pred = float(op["tau_pred"])
    # validation-frozen operating point override (Fig 4 final: beta*/tau* from
    # op_val_select_test_eval.py, so the trajectory matches the reported headline).
    if beta_override is not None:
        beta_star = float(beta_override)
    if tau_override is not None:
        tau_pred = float(tau_override)
    print(f"[{cell_name}] operating point: beta*={beta_star} q*={q_star} "
          f"tau_pred={tau_pred:.6f}"
          + (" (validation-frozen override)" if (beta_override is not None
                                                 or tau_override is not None) else ""))

    # --- TEST dataset ---
    test_ds = AccidentWindowDataset(cell["test_csv"], seq_len=seq_len,
                                    pred_len=pred_len, cache_dir=cache_dir)
    assert test_ds.num_controls == num_controls and test_ds.input_size == input_size

    # --- 1. full baseline AR (beta=0; exact uncorrected null-op) over TEST ---
    base_pred, true_dict = autoregressive_corrected_batched_acc(
        model, None, 0.0, test_ds, num_controls, step_norm_const, device=device,
        num_continuous=nc, collect_batch=collect_batch, num_workers=num_workers)

    base_mae = per_scenario_mae(base_pred, true_dict)
    if best_rank is not None:
        # rank by how much the corrector IMPROVES each scenario -> need corrected MAE
        # over the FULL test (one gated rollout), then pick the biggest before/after win.
        corr_full, ctrue_full = gated_corrected_rollout_acc(
            model, mlp, beta_star, tau_pred, test_ds, num_controls, step_norm_const,
            device=device, gate_on="pred", num_continuous=nc,
            collect_batch=collect_batch, num_workers=num_workers)
        corr_mae_all = per_scenario_mae(corr_full, ctrue_full)
        chosen, red_sel, pct = choose_best_scenario(base_mae, corr_mae_all, best_rank)
        print(f"[{cell_name}] BEST-improvement rank-{best_rank} scenario={chosen} "
              f"(~{pct:.1f} pct baseline MAE; baseline={base_mae[chosen]:.5f} -> "
              f"corrected={corr_mae_all[chosen]:.5f}, {red_sel:.1f}%)")
    elif worst_rank is not None:
        chosen, pct = choose_worst_scenario(base_mae, worst_rank)
        print(f"[{cell_name}] WORST-rank-{worst_rank} scenario={chosen} at ~{pct:.1f} pct "
              f"of baseline per-scenario MAE (actual tail)")
    else:
        chosen, pct, lo, hi = choose_tail_scenario(base_mae)
        print(f"[{cell_name}] scenario={chosen} at ~{pct:.1f} pct of baseline per-scenario "
              f"MAE (90-95 band=[{lo:.5f},{hi:.5f}])")

    # --- 2. step-gated corrected AR at (beta*, tau_pred), restricted to chosen scenario,
    #        recording gate fires; cross-validated against the canonical rollout. ---
    corr_pred, corr_true, fire_dict = _gated_rollout_with_fires(
        model, mlp, beta_star, tau_pred, test_ds, num_controls, step_norm_const,
        device=device, num_continuous=nc, collect_batch=collect_batch,
        num_workers=num_workers, restrict_scenarios=[chosen])
    canon_pred, _ = gated_corrected_rollout_acc(
        model, mlp, beta_star, tau_pred, test_ds, num_controls, step_norm_const,
        device=device, gate_on="pred", num_continuous=nc, collect_batch=collect_batch,
        num_workers=num_workers, restrict_scenarios=[chosen])
    canon = _stack(canon_pred, chosen)
    assert np.allclose(canon, corr_pred[chosen], atol=1e-6), \
        "gate-fire mirror diverges from canonical gated_corrected_rollout_acc"

    # --- per-scenario MAE: baseline vs corrected (confirm the correction helps here) ---
    gt = _stack(true_dict, chosen)                    # [L, 10]
    bl = _stack(base_pred, chosen)                    # [L, 10]
    cr = corr_pred[chosen]                            # [L, 10]
    fr = fire_dict[chosen]                            # [L] bool
    base_mae_scen = float(np.mean(np.abs(bl - gt)))
    corr_mae_scen = float(np.mean(np.abs(cr - gt)))
    rel = 100.0 * (base_mae_scen - corr_mae_scen) / base_mae_scen if base_mae_scen > 0 else float("nan")
    n_fire = int(fr.sum())
    print(f"[{cell_name}] scenario={chosen} L={gt.shape[0]} baseline_MAE={base_mae_scen:.6f} "
          f"corrected_MAE={corr_mae_scen:.6f} rel_reduction={rel:.2f}% gate_fires={n_fire}/{gt.shape[0]}")

    # --- variable selection: 6 vars with largest per-variable baseline error here ---
    per_var_err = np.mean(np.abs(bl - gt), axis=0)    # [10]
    top6 = list(np.argsort(per_var_err)[::-1][:6])
    print(f"[{cell_name}] top-6 drifting vars: "
          + ", ".join(f"{CONTINUOUS_COLS[j]}({per_var_err[j]:.4f})" for j in top6))

    # --- SAMG intervention timing for this scenario (control state changes) ---
    samg = samg_edges(test_ds, chosen, seq_len, gt.shape[0])
    if samg:
        print(f"[{cell_name}] SAMG interventions (rollout step): "
              + ", ".join(f"{nm}@{rs}({d})" for nm, rs, k, d in samg))
    else:
        print(f"[{cell_name}] no SAMG state change within the rollout window")

    set_style()
    steps = np.arange(gt.shape[0])

    # --- SINGLE-VARIABLE mode (--var): (1) surrogate prediction overlay + (2) GT-only ---
    if var_name is not None:
        assert var_name in CONTINUOUS_COLS, \
            f"--var {var_name} not in {CONTINUOUS_COLS}"
        j = CONTINUOUS_COLS.index(var_name)
        which = ("one-shot" if model_file == "error_mlp.pt"
                 else os.path.splitext(model_file)[0].replace("error_mlp_", ""))
        vj_err = float(np.mean(np.abs(bl[:, j] - gt[:, j])))
        cj_err = float(np.mean(np.abs(cr[:, j] - gt[:, j])))
        vsafe = var_name.replace("(", "").replace(")", "")

        def _draw_samg_gate(ax):
            tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
            for (nm, rs, ci, direction) in samg:
                ax.plot(rs, 0.0, marker="^", ms=7, mec="none",
                        color=SAMG_COLORS[ci % len(SAMG_COLORS)], transform=tr,
                        clip_on=False, zorder=5, label=f"{nm} (on)")
            fsteps = steps[fr]
            if fsteps.size:
                a = max(0.12, min(0.65, 8.0 / len(fsteps)))
                for i, fs in enumerate(fsteps):
                    ax.axvline(fs, color=RUG_C, lw=1.1, alpha=a, zorder=1.8,
                               label="gate fired (step)" if i == 0 else None)

        written = []
        # (1) surrogate prediction overlay: GT vs baseline vs ErrorMLP-corrected
        figp, axp = plt.subplots(figsize=(7.2, 4.0))
        axp.plot(steps, gt[:, j], color=GT_C, lw=2.0, solid_capstyle="round",
                 label="Ground truth", zorder=3)
        axp.plot(steps, bl[:, j], color=BASE_C, lw=1.5, ls="--",
                 label="Baseline AR (backbone)", zorder=2)
        axp.plot(steps, cr[:, j], color=CORR_C, lw=1.8, ls="-",
                 label="ErrorMLP-corrected AR", zorder=2.5)
        _draw_samg_gate(axp)
        axp.set_xlabel("rollout step"); axp.set_ylabel(f"{var_name} (scaled)")
        if ylim is not None:
            axp.set_ylim(ylim[0], ylim[1])
        axp.grid(alpha=0.3, linewidth=0.6); axp.set_axisbelow(True); axp.margins(x=0.01)
        axp.set_title(f"{cell_name}, scenario {chosen} (~{pct:.0f} pct): {which} surrogate "
                      f"prediction of {var_name} (MAE {vj_err:.4f}→{cj_err:.4f})", fontsize=9.5)
        axp.legend(frameon=False, fontsize=8, ncol=2, loc="best")
        figp.tight_layout()
        stem_p = os.path.join(FIG_DIR, f"fig_traj_{cell_name}_{vsafe}_s{chosen}_pred")
        os.makedirs(FIG_DIR, exist_ok=True)
        figp.savefig(stem_p + ".pdf", bbox_inches="tight")
        figp.savefig(stem_p + ".png", dpi=300, bbox_inches="tight")
        plt.close(figp); written.append(stem_p)

        # (2) GT-only trajectory (the actual mapped-variable trajectory, standalone)
        figg, axg = plt.subplots(figsize=(7.2, 4.0))
        axg.plot(steps, gt[:, j], color=GT_C, lw=2.0, solid_capstyle="round",
                 label="Ground truth", zorder=3)
        _draw_samg_gate(axg)
        axg.set_xlabel("rollout step"); axg.set_ylabel(f"{var_name} (scaled)")
        if ylim is not None:
            axg.set_ylim(ylim[0], ylim[1])
        axg.grid(alpha=0.3, linewidth=0.6); axg.set_axisbelow(True); axg.margins(x=0.01)
        axg.set_title(f"{cell_name}, scenario {chosen} (~{pct:.0f} pct): "
                      f"ground-truth {var_name} trajectory", fontsize=9.5)
        axg.legend(frameon=False, fontsize=8, ncol=2, loc="best")
        figg.tight_layout()
        stem_g = os.path.join(FIG_DIR, f"fig_traj_{cell_name}_{vsafe}_s{chosen}_gt")
        figg.savefig(stem_g + ".pdf", bbox_inches="tight")
        figg.savefig(stem_g + ".png", dpi=300, bbox_inches="tight")
        plt.close(figg); written.append(stem_g)

        for st in written:
            print(f"[{cell_name}][done] wrote {st}.png")
        return {"cell": cell_name, "scenario": chosen, "var": var_name,
                "baseline_var_mae": vj_err, "corrected_var_mae": cj_err,
                "beta_star": beta_star, "tau_pred": tau_pred}

    # --- 3. plot small-multiples 2x3 ---
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.4), sharex=True)
    axes = axes.ravel()
    for k, (ax, j) in enumerate(zip(axes, top6)):
        ax.plot(steps, gt[:, j], color=GT_C, lw=2.0, solid_capstyle="round",
                label="Ground truth", zorder=3)
        ax.plot(steps, bl[:, j], color=BASE_C, lw=1.4, ls="--",
                label="Baseline AR (backbone)", zorder=2)
        ax.plot(steps, cr[:, j], color=CORR_C, lw=1.6, ls="-",
                label="ErrorMLP-corrected AR", zorder=2.5)
        # SAMG intervention lines (control state change = operator action point).
        # SAMG onsets: SMALL upward triangles on the x-axis (not full-height lines).
        samg_tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        for (nm, rs, ci, direction) in samg:
            ax.plot(rs, 0.0, marker="^", ms=6, mec="none",
                    color=SAMG_COLORS[ci % len(SAMG_COLORS)], transform=samg_tr,
                    clip_on=False, zorder=5, label=f"{nm} (on)")
        # gate-fired: FULL-HEIGHT green lines (solid, distinct from the dotted SAMG
        # lines). alpha adapts to the fire count so a few fires are bold and hundreds
        # blend into a faint band instead of blacking out the panel.
        fire_steps = steps[fr]
        if fire_steps.size:
            a = max(0.12, min(0.65, 8.0 / len(fire_steps)))
            for i, fs in enumerate(fire_steps):
                ax.axvline(fs, color=RUG_C, lw=1.1, alpha=a, zorder=1.8,
                           label="gate fired (step)" if i == 0 else None)
        ax.set_title(CONTINUOUS_COLS[j], fontsize=9)
        ax.grid(axis="both", alpha=0.3, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.margins(x=0.01)
        if k % 3 == 0:
            ax.set_ylabel("value (scaled)")
        if k >= 3:
            ax.set_xlabel("rollout step")

    # --- one shared legend (de-duplicated across panels) ---
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4),
               frameon=False, bbox_to_anchor=(0.5, -0.03))

    which = "one-shot" if model_file == "error_mlp.pt" else os.path.splitext(model_file)[0].replace("error_mlp_", "")
    if best_rank is not None:
        kind = f"BEST-corrected tail (rank {best_rank})"
    elif worst_rank is not None:
        kind = f"WORST tail (rank {worst_rank})"
    else:
        kind = "representative tail"
    title = (f"{cell_name}, {kind} scenario {chosen} (~{pct:.0f} pct): {which} "
             f"ErrorMLP-corrected AR tracks GT, suppresses drift "
             f"(MAE {base_mae_scen:.4f}→{corr_mae_scen:.4f}, {rel:+.0f}%)")
    fig.suptitle(title, fontsize=9.5, y=0.995)
    fig.tight_layout(rect=(0, 0.09, 1, 0.97))

    os.makedirs(FIG_DIR, exist_ok=True)
    if model_file == "error_mlp.pt" and worst_rank is None:
        stem_name = f"fig6_trajectories_{cell_name}"            # backward-compatible default
    else:
        tag = os.path.splitext(model_file)[0].replace("error_mlp", "").strip("_") or "oneshot"
        if best_rank is not None:
            stag = f"best{best_rank}"
        elif worst_rank is not None:
            stag = f"worst{worst_rank}"
        else:
            stag = "band"
        stem_name = f"fig_traj_{cell_name}_{tag}_{stag}"
    stem = os.path.join(FIG_DIR, stem_name)
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    fig.savefig(stem + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[{cell_name}][done] wrote {stem}.pdf and {stem}.png")

    return {
        "cell": cell_name, "scenario": chosen, "percentile": pct,
        "beta_star": beta_star, "q_star": q_star, "tau_pred": tau_pred,
        "L": int(gt.shape[0]), "gate_fires": n_fire,
        "baseline_mae": base_mae_scen, "corrected_mae": corr_mae_scen,
        "rel_reduction_pct": rel,
        "vars": [CONTINUOUS_COLS[j] for j in top6],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="append", required=True,
                    help="cell name; repeat to render several (e.g. --cell SBO --cell TLOFW_CSP)")
    ap.add_argument("--model-file", default="error_mlp.pt",
                    help="ErrorMLP under <out_root>/<cell>/ (e.g. error_mlp_dagger.pt for the "
                         "2-round tail-loss DAgger corrector). Its matching tail_analysis json "
                         "supplies the operating point.")
    ap.add_argument("--worst-rank", type=int, default=None,
                    help="Plot the rank-th ACTUAL worst tail scenario (0=the single worst) "
                         "instead of the representative 90-95 pct band.")
    ap.add_argument("--best-rank", type=int, default=None,
                    help="Plot the rank-th SUCCESS scenario where the corrector helps most "
                         "(0=biggest baseline->corrected MAE reduction, among high-drift "
                         "scenarios). Overrides --worst-rank.")
    ap.add_argument("--beta", type=float, default=None,
                    help="Override correction gain beta* (e.g. validation-frozen op).")
    ap.add_argument("--tau-pred", type=float, default=None,
                    help="Override gate threshold tau* (e.g. validation-frozen op).")
    ap.add_argument("--var", default=None,
                    help="Plot a SINGLE continuous variable (e.g. 'PEX0(17)'): writes a "
                         "surrogate-prediction overlay (_pred) AND a GT-only trajectory "
                         "(_gt), instead of the 2x3 top-6 panel.")
    ap.add_argument("--ylim", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="Fix the y-axis range (e.g. --ylim 0 1) on the --var figures.")
    args = ap.parse_args()

    cfg = utils.load_config("error_mlp_accident")
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"device={device} torch={torch.__version__} matplotlib={matplotlib.__version__}")

    summaries = [run_cell(c, cfg, device, model_file=args.model_file,
                          worst_rank=args.worst_rank, best_rank=args.best_rank,
                          beta_override=args.beta, tau_override=args.tau_pred,
                          var_name=args.var, ylim=args.ylim)
                 for c in args.cell]
    print("\n=== SUMMARY ===")
    for s in summaries:
        print(json.dumps(s))


if __name__ == "__main__":
    main()
