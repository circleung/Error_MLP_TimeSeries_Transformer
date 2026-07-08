"""Per-accident-type TURNING-POINT / directional-error AR-rollout failure-mode analysis.

ADDITIVE, read-only-on-the-backbone. NO retraining: it REUSES the frozen per-cell
backbone and the EXACT baseline beta=0 AR rollout (autoregressive_corrected_batched_acc
with error_mlp=None, beta=0.0), which is byte-identical to the uncorrected baseline AR.
It imports (does not modify) load_frozen_backbone_acc, autoregressive_corrected_batched_acc,
_collect_pass, _pack, AccidentWindowDataset, infer_schema_from_csv, CONTINUOUS_COLS,
NUM_CONTINUOUS.

Hypothesis under test (user's insight): the catastrophic AR error blowups happen at
TURNING POINTS -- steps where a variable's true trajectory changes direction
(up<->down inflection). If the model gets the DIRECTION wrong there, the wrong value
feeds back and the rollout diverges. Concretely:
  H1: large / tail errors concentrate at/near turning points (error LIFT vs base rate).
  H2: directional hit-rate is worse at turning points than overall.
  H3: a directional MISS at a turning point predicts a large DOWNSTREAM error blowup
      over the next H steps, vs a correct-direction (HIT) turning point.

Definitions (per continuous output variable k, per scenario, over the ordered rollout
steps t=0..L-1; SCALED space -- the data is scaled):
  y_true[t] = true value (from true_dict);  yhat[t] = AR prediction (predictions_dict).
  y_true[-1] = last continuous row of the scenario's init window (the true observation
               immediately before step 0), so Delta at t=0 is well-defined.
  Delta_true[t] = y_true[t] - y_true[t-1]           (true change)
  Delta_pred[t] = yhat[t]   - y_true[t-1]           (predicted change vs PREVIOUS TRUTH)
  Turning point at t (per var): sign(Delta_true[t]) != sign(Delta_true[t-1]) with a
    deadband |Delta_true|>eps on BOTH steps (eps = a small percentile of |Delta_true|
    over that variable's pooled steps, to ignore flat noise; recorded). Curvature
    variant: |second difference| = |Delta_true[t] - Delta_true[t-1]| in the top decile.
  Directional hit at t: sign(Delta_pred[t]) == sign(Delta_true[t]).
  Per-step abs error e[t,k] = |yhat[t,k] - y_true[t,k]|; pooled per-step error =
    mean over the 10 vars (== compute_micro_macro per-step MAE).

Metrics (per cell; per-variable AND pooled):
  H1: mean per-step error at turning-point steps and within +/-2, vs non-turning; the
      SHARE of total error and of p99-tail-exceeding steps that fall at/near turning
      points vs the base rate of turning steps (lift = concentration = share/base_rate).
  H2: directional hit-rate overall vs at turning points.
  H3: partition turning-point steps into MISS vs HIT (by directional hit at the TP).
      For H in {5,10,20}, downstream cumulative error = sum_{i=1..H} e[t+i] (pooled over
      the 10 vars per step). Report mean over MISS-TPs vs HIT-TPs and the ratio; plus the
      mean-error trajectory at offsets 0..H for the two groups.
  Which variables have the most turning-point-concentrated / directional failures (rank).

Writes <out_root>/<cell>/turning_point_analysis.json.

Usage (from src/):
    NONINTERACTIVE=1 python experiments/turning_point_analysis_acc.py --cell SBO
    NONINTERACTIVE=1 python experiments/turning_point_analysis_acc.py --cell SBO --subsample 800
"""
import os
import sys
import json
import time
import argparse

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

os.environ.setdefault("NONINTERACTIVE", "1")

import numpy as np
import torch

import utils
from accident_dataset import (
    AccidentWindowDataset,
    infer_schema_from_csv,
    CONTINUOUS_COLS,
    NUM_CONTINUOUS,
)
from error_rollout_acc import (
    load_frozen_backbone_acc,
    autoregressive_corrected_batched_acc,
    _collect_pass,
    _pack,
)

SUBSAMPLE = 800                # fixed representative scenario subsample (first N ids asc).
EPS_PCTL = 25.0                # deadband eps = this percentile of |Delta_true| (per var).
CURV_TOP_DECILE = 90.0         # curvature TP = |2nd diff| above this percentile (per var).
HORIZONS = (5, 10, 20)         # downstream horizons for H3.
NEAR_WINDOW = 2                # "near a turning point" = within +/- this many steps.
TAIL_PCTL = 99.0               # p99 tail on the pooled per-step error.


def assert_frozen(model):
    """Hard-assert the backbone carries no trainable parameters (NO retraining)."""
    n_trainable = sum(int(p.requires_grad) for p in model.parameters())
    assert n_trainable == 0, f"backbone is NOT frozen: {n_trainable} trainable params"
    assert not model.training, "backbone must be in eval() mode"


def _signed_deadband(delta, eps):
    """sign(delta) with a deadband: 0 where |delta| <= eps, else +/-1. Array-safe."""
    s = np.sign(delta)
    s[np.abs(delta) <= eps] = 0.0
    return s


def analyze_cell(preds_by_scn, true_by_scn, prev_true_by_scn, kept_ids, cont_cols):
    """Core turning-point / directional analysis over the kept scenarios.

    preds_by_scn[s], true_by_scn[s]: ordered lists of per-step [10] vectors (t=0..L-1).
    prev_true_by_scn[s]: [10] true observation immediately BEFORE step 0 (init-window
    last continuous row), so Delta at t=0 is defined.

    Returns a dict with H1/H2/H3 per-variable AND pooled, the eps/curvature thresholds,
    and the variable rankings."""
    K = NUM_CONTINUOUS
    # ---- Build per-scenario aligned arrays (variable-length L per scenario). ----
    scen = []
    for s in kept_ids:
        s = int(s)
        if s not in preds_by_scn or len(preds_by_scn[s]) == 0:
            continue
        yhat = np.vstack([np.ravel(p) for p in preds_by_scn[s]]).astype(np.float64)  # [L,10]
        ytru = np.vstack([np.ravel(y) for y in true_by_scn[s]]).astype(np.float64)   # [L,10]
        assert yhat.shape == ytru.shape and yhat.shape[1] == K
        y_prev0 = np.asarray(prev_true_by_scn[s], dtype=np.float64).reshape(K)       # [10]
        scen.append((s, yhat, ytru, y_prev0))
    assert scen, "no scenarios to analyze after alignment"

    # y_true augmented with y_true[-1] prepended -> yt_aug[0]=y_prev0, yt_aug[t+1]=ytru[t].
    # Delta_true[t] = yt_aug[t+1]-yt_aug[t]; Delta_pred[t] = yhat[t]-yt_aug[t].
    # ---- eps deadband per variable (percentile of |Delta_true| pooled over all steps). ----
    abs_dtrue_pool = [[] for _ in range(K)]
    for (s, yhat, ytru, y_prev0) in scen:
        yt_aug = np.vstack([y_prev0[None, :], ytru])           # [L+1, 10]
        dtrue = np.diff(yt_aug, axis=0)                        # [L, 10] = Delta_true[t]
        for k in range(K):
            abs_dtrue_pool[k].append(np.abs(dtrue[:, k]))
    eps_per_var = np.array([
        float(np.percentile(np.concatenate(abs_dtrue_pool[k]), EPS_PCTL))
        if len(abs_dtrue_pool[k]) else 0.0 for k in range(K)], dtype=np.float64)
    # curvature threshold per var (top-decile of |2nd diff| of the true trajectory).
    abs_curv_pool = [[] for _ in range(K)]
    for (s, yhat, ytru, y_prev0) in scen:
        yt_aug = np.vstack([y_prev0[None, :], ytru])
        dtrue = np.diff(yt_aug, axis=0)                        # [L,10]
        curv = np.abs(np.diff(dtrue, axis=0))                 # [L-1,10] = |2nd diff|, t>=1
        for k in range(K):
            if curv.shape[0] > 0:
                abs_curv_pool[k].append(curv[:, k])
    curv_thr_per_var = np.array([
        float(np.percentile(np.concatenate(abs_curv_pool[k]), CURV_TOP_DECILE))
        if len(abs_curv_pool[k]) else np.inf for k in range(K)], dtype=np.float64)

    # ---- Per-variable accumulators. ----
    # H1
    tp_err_sum = np.zeros(K); tp_n = np.zeros(K, np.int64)
    near_err_sum = np.zeros(K); near_n = np.zeros(K, np.int64)
    non_err_sum = np.zeros(K); non_n = np.zeros(K, np.int64)
    tot_err_sum = np.zeros(K); tot_n = np.zeros(K, np.int64)
    tp_err_at_tp_sum = np.zeros(K)   # error carried BY tp steps (for share of total error)
    # tail (p99 per var over all steps): computed after pooling errors -> stash per-var errs
    all_err_per_var = [[] for _ in range(K)]
    all_is_tp_per_var = [[] for _ in range(K)]
    all_is_near_per_var = [[] for _ in range(K)]
    # curvature-TP error accumulators
    ctp_err_sum = np.zeros(K); ctp_n = np.zeros(K, np.int64)
    # H2
    hit_all = np.zeros(K, np.int64); n_all = np.zeros(K, np.int64)
    hit_tp = np.zeros(K, np.int64); n_tp_dir = np.zeros(K, np.int64)
    # H3 per-variable downstream (pooled-over-vars error at t+i is handled at pooled level;
    # per-variable H3 uses the per-variable error e[.,k] downstream).
    maxH = max(HORIZONS)
    h3_pv = {k: {"miss_cum": {H: [] for H in HORIZONS},
                 "hit_cum": {H: [] for H in HORIZONS}} for k in range(K)}

    # ---- Pooled accumulators.
    # H1/H2 pooled = pooled over (variable, step) TURNING-POINT EVENTS (NOT "any-var per
    # step", which saturates with 10 vars). We accumulate, over ALL (scenario, var, step):
    #   - the per-variable abs error e[t,k] split into {TP, near-TP, non-near} events,
    #   - the p99 tail flag (per variable, over that var's pooled errors) crossed with
    #     the TP/near flag (event counts) -> pooled tail share / lift,
    #   - directional hit counts at TP vs overall (via the per-var hit_all/hit_tp sums).
    # H3 pooled uses the POOLED per-step MAE downstream (mean over 10 vars) because the
    # user's hypothesis is that a directional miss at a turning point makes the WHOLE
    # rollout diverge, not just that one variable.
    pool_ev_err = []          # per (var,step) error, ALL steps
    pool_ev_is_tp = []        # per (var,step): step is a TP for THAT var
    pool_ev_is_near = []      # per (var,step): step is near a TP for THAT var
    pool_ev_var = []          # per (var,step): variable index (for per-var tail thresh)
    h3_pool = {"miss_cum": {H: [] for H in HORIZONS},
               "hit_cum": {H: [] for H in HORIZONS},
               "miss_traj": {H: [] for H in HORIZONS},   # per-offset trajectories
               "hit_traj": {H: [] for H in HORIZONS}}

    for (s, yhat, ytru, y_prev0) in scen:
        L = ytru.shape[0]
        yt_aug = np.vstack([y_prev0[None, :], ytru])          # [L+1,10]
        dtrue = np.diff(yt_aug, axis=0)                        # [L,10] Delta_true[t]
        dpred = yhat - yt_aug[:-1, :]                          # [L,10] Delta_pred[t]
        e = np.abs(yhat - ytru)                                # [L,10] per-step abs error
        e_pool = e.mean(axis=1)                                # [L] pooled per-step error

        # signed direction with deadband
        s_true = np.stack([_signed_deadband(dtrue[:, k], eps_per_var[k]) for k in range(K)], axis=1)  # [L,10]
        s_pred = np.stack([_signed_deadband(dpred[:, k], eps_per_var[k]) for k in range(K)], axis=1)

        # turning point at t (per var): sign flip vs previous step, both above deadband.
        # need s_true[t] and s_true[t-1]; valid t>=1. Both nonzero and opposite sign.
        is_tp = np.zeros((L, K), dtype=bool)
        for k in range(K):
            st = s_true[:, k]
            prev = st[:-1]; cur = st[1:]                      # t-1 vs t, aligned to t=1..L-1
            flip = (prev != 0) & (cur != 0) & (np.sign(prev) != np.sign(cur))
            is_tp[1:, k] = flip
        # curvature-based TP variant (per var): |2nd diff| above top-decile threshold, t>=1.
        curv = np.abs(np.diff(dtrue, axis=0))                 # [L-1,10] aligned to t=1..L-1
        is_ctp = np.zeros((L, K), dtype=bool)
        for k in range(K):
            if curv.shape[0] > 0:
                is_ctp[1:, k] = curv[:, k] > curv_thr_per_var[k]

        # "near a TP" (per var): within +/- NEAR_WINDOW of any TP step for that var.
        is_near = np.zeros((L, K), dtype=bool)
        for k in range(K):
            tp_idx = np.where(is_tp[:, k])[0]
            for ti in tp_idx:
                lo = max(0, ti - NEAR_WINDOW); hi = min(L, ti + NEAR_WINDOW + 1)
                is_near[lo:hi, k] = True

        # directional hit at t (per var): sign(Delta_pred)==sign(Delta_true); count only
        # where the TRUE step has a real direction (deadband nonzero) so a "hit" is meaningful.
        hit = (s_pred == s_true)                               # [L,10]

        for k in range(K):
            ek = e[:, k]
            tpk = is_tp[:, k]; nk = is_near[:, k]
            # H1 accumulators (per var)
            tot_err_sum[k] += ek.sum(); tot_n[k] += L
            tp_err_sum[k] += ek[tpk].sum(); tp_n[k] += int(tpk.sum())
            tp_err_at_tp_sum[k] += ek[tpk].sum()
            near_err_sum[k] += ek[nk].sum(); near_n[k] += int(nk.sum())
            non_err_sum[k] += ek[~nk].sum(); non_n[k] += int((~nk).sum())
            ctpk = is_ctp[:, k]
            ctp_err_sum[k] += ek[ctpk].sum(); ctp_n[k] += int(ctpk.sum())
            all_err_per_var[k].append(ek)
            all_is_tp_per_var[k].append(tpk)
            all_is_near_per_var[k].append(nk)
            # H2 accumulators (only where true direction is defined, deadband nonzero)
            valid_dir = s_true[:, k] != 0
            hit_all[k] += int(hit[valid_dir, k].sum()); n_all[k] += int(valid_dir.sum())
            tp_valid = tpk & valid_dir
            hit_tp[k] += int(hit[tp_valid, k].sum()); n_tp_dir[k] += int(tp_valid.sum())
            # H3 per var: at each TP step (with defined direction), MISS/HIT by hit[t,k],
            # downstream cumulative per-var error over next H steps.
            for ti in np.where(tp_valid)[0]:
                for H in HORIZONS:
                    if ti + H < L:
                        cum = float(ek[ti + 1: ti + 1 + H].sum())
                        if hit[ti, k]:
                            h3_pv[k]["hit_cum"][H].append(cum)
                        else:
                            h3_pv[k]["miss_cum"][H].append(cum)

        # ---- pooled level (event-pooled over (var, step)) ----
        # flatten per-variable errors + TP/near flags across the 10 vars for this scenario.
        for k in range(K):
            pool_ev_err.append(e[:, k])
            pool_ev_is_tp.append(is_tp[:, k])
            pool_ev_is_near.append(is_near[:, k])
            pool_ev_var.append(np.full(L, k, dtype=np.int64))
        # H3 pooled: iterate per-(var) TP events, classify by that var's direction hit,
        # measure the POOLED downstream error (mean over 10 vars) -> "does a directional
        # miss at a turning point -> pooled error explodes downstream".
        for k in range(K):
            tp_valid_k = is_tp[:, k] & (s_true[:, k] != 0)
            for ti in np.where(tp_valid_k)[0]:
                grp = "hit" if hit[ti, k] else "miss"
                for H in HORIZONS:
                    if ti + H < L:
                        cum = float(e_pool[ti + 1: ti + 1 + H].sum())
                        h3_pool[grp + "_cum"][H].append(cum)
                        # per-offset trajectory (offsets 0..H): pooled error at ti..ti+H
                        traj = e_pool[ti: ti + H + 1]
                        if traj.shape[0] == H + 1:
                            h3_pool[grp + "_traj"][H].append(traj)

    # ---- finalize per-variable H1 ----
    def safe_div(a, b):
        return float(a) / float(b) if b else float("nan")

    # per-var p99 tail: fraction of tail steps that are at/near TP vs base rate.
    per_var = []
    tp_share_of_tail = np.zeros(K); tp_base_rate = np.zeros(K)
    for k in range(K):
        ek = np.concatenate(all_err_per_var[k])
        tpk = np.concatenate(all_is_tp_per_var[k])
        nk = np.concatenate(all_is_near_per_var[k])
        n = ek.shape[0]
        base_rate = safe_div(int(tpk.sum()), n)                # TP base rate
        near_base_rate = safe_div(int(nk.sum()), n)
        # share of TOTAL error carried by TP / near-TP steps
        share_err_tp = safe_div(ek[tpk].sum(), ek.sum())
        share_err_near = safe_div(ek[nk].sum(), ek.sum())
        # p99 tail steps
        thr = float(np.percentile(ek, TAIL_PCTL)) if n else float("nan")
        tail_mask = ek >= thr
        n_tail = int(tail_mask.sum())
        share_tail_at_tp = safe_div(int((tail_mask & tpk).sum()), n_tail)
        share_tail_near_tp = safe_div(int((tail_mask & nk).sum()), n_tail)
        lift_tail_tp = safe_div(share_tail_at_tp, base_rate)
        lift_tail_near = safe_div(share_tail_near_tp, near_base_rate)
        tp_share_of_tail[k] = share_tail_at_tp
        tp_base_rate[k] = base_rate

        mean_tp = safe_div(tp_err_sum[k], tp_n[k])
        mean_near = safe_div(near_err_sum[k], near_n[k])
        mean_non = safe_div(non_err_sum[k], non_n[k])
        mean_all = safe_div(tot_err_sum[k], tot_n[k])
        mean_ctp = safe_div(ctp_err_sum[k], ctp_n[k])
        err_ratio_tp_non = safe_div(mean_tp, mean_non)
        err_ratio_near_non = safe_div(mean_near, mean_non)

        # H2 per var
        hit_rate_all = safe_div(hit_all[k], n_all[k])
        hit_rate_tp = safe_div(hit_tp[k], n_tp_dir[k])

        # H3 per var (ratio at each H)
        h3 = {}
        for H in HORIZONS:
            mc = np.asarray(h3_pv[k]["miss_cum"][H], dtype=np.float64)
            hc = np.asarray(h3_pv[k]["hit_cum"][H], dtype=np.float64)
            mm = float(mc.mean()) if mc.size else float("nan")
            hm = float(hc.mean()) if hc.size else float("nan")
            h3[str(H)] = {
                "miss_mean_cum_err": mm, "hit_mean_cum_err": hm,
                "ratio_miss_over_hit": (mm / hm) if (hc.size and hm != 0) else float("nan"),
                "n_miss": int(mc.size), "n_hit": int(hc.size),
            }

        per_var.append({
            "index": k, "name": cont_cols[k],
            "eps_deadband": float(eps_per_var[k]),
            "curv_threshold": float(curv_thr_per_var[k]),
            "n_steps": int(n),
            "n_turning_points": int(tp_n[k]),
            "turning_point_base_rate": base_rate,
            "near_tp_base_rate": near_base_rate,
            "H1": {
                "mean_err_at_tp": mean_tp,
                "mean_err_near_tp": mean_near,
                "mean_err_non_tp": mean_non,
                "mean_err_all": mean_all,
                "mean_err_curv_tp": mean_ctp,
                "err_ratio_tp_over_non": err_ratio_tp_non,
                "err_ratio_near_over_non": err_ratio_near_non,
                "share_total_err_at_tp": share_err_tp,
                "share_total_err_near_tp": share_err_near,
                "p99_threshold": thr,
                "n_tail_steps": n_tail,
                "share_tail_at_tp": share_tail_at_tp,
                "share_tail_near_tp": share_tail_near_tp,
                "lift_tail_at_tp": lift_tail_tp,
                "lift_tail_near_tp": lift_tail_near,
            },
            "H2": {
                "dir_hit_rate_overall": hit_rate_all,
                "dir_hit_rate_at_tp": hit_rate_tp,
                "dir_hit_rate_drop": (hit_rate_all - hit_rate_tp)
                if not (np.isnan(hit_rate_all) or np.isnan(hit_rate_tp)) else float("nan"),
                "n_dir_steps": int(n_all[k]),
                "n_dir_tp_steps": int(n_tp_dir[k]),
            },
            "H3": h3,
        })

    # ---- finalize pooled H1 (event-pooled over (var, step)) ----
    ev_err = np.concatenate(pool_ev_err)                    # [N_events] per-(var,step) error
    ev_tp = np.concatenate(pool_ev_is_tp)                   # [N_events] bool: TP for that var
    ev_near = np.concatenate(pool_ev_is_near)              # [N_events] bool: near-TP
    ev_var = np.concatenate(pool_ev_var)                   # [N_events] var index
    n_ev = ev_err.shape[0]
    base_rate_tp = safe_div(int(ev_tp.sum()), n_ev)
    base_rate_near = safe_div(int(ev_near.sum()), n_ev)
    mean_tp_p = safe_div(ev_err[ev_tp].sum(), int(ev_tp.sum())) if ev_tp.any() else float("nan")
    mean_near_p = safe_div(ev_err[ev_near].sum(), int(ev_near.sum())) if ev_near.any() else float("nan")
    mean_non_p = safe_div(ev_err[~ev_near].sum(), int((~ev_near).sum())) if (~ev_near).any() else float("nan")
    share_err_tp_p = safe_div(ev_err[ev_tp].sum(), ev_err.sum())
    share_err_near_p = safe_div(ev_err[ev_near].sum(), ev_err.sum())
    # p99 tail is defined PER VARIABLE (different scales), then pooled at the event level.
    ev_tail = np.zeros(n_ev, dtype=bool)
    for k in range(K):
        mk = ev_var == k
        if mk.any():
            thr_k = np.percentile(ev_err[mk], TAIL_PCTL)
            ev_tail |= mk & (ev_err >= thr_k)
    n_tail_p = int(ev_tail.sum())
    share_tail_tp_p = safe_div(int((ev_tail & ev_tp).sum()), n_tail_p)
    share_tail_near_p = safe_div(int((ev_tail & ev_near).sum()), n_tail_p)

    pooled_H1 = {
        "n_events": int(n_ev),
        "pooling": "over (variable, step) turning-point events; per-var p99 tail threshold",
        "turning_point_base_rate": base_rate_tp,
        "near_tp_base_rate": base_rate_near,
        "mean_err_at_tp": mean_tp_p,
        "mean_err_near_tp": mean_near_p,
        "mean_err_non_tp": mean_non_p,
        "err_ratio_tp_over_non": safe_div(mean_tp_p, mean_non_p),
        "err_ratio_near_over_non": safe_div(mean_near_p, mean_non_p),
        "share_total_err_at_tp": share_err_tp_p,
        "share_total_err_near_tp": share_err_near_p,
        "n_tail_events": n_tail_p,
        "share_tail_at_tp": share_tail_tp_p,
        "share_tail_near_tp": share_tail_near_p,
        "lift_tail_at_tp": safe_div(share_tail_tp_p, base_rate_tp),
        "lift_tail_near_tp": safe_div(share_tail_near_p, base_rate_near),
    }

    # pooled H2 = mean over vars of the per-var directional hit rates (weighted by n_dir).
    hr_all_num = int(hit_all.sum()); hr_all_den = int(n_all.sum())
    hr_tp_num = int(hit_tp.sum()); hr_tp_den = int(n_tp_dir.sum())
    pooled_H2 = {
        "dir_hit_rate_overall": safe_div(hr_all_num, hr_all_den),
        "dir_hit_rate_at_tp": safe_div(hr_tp_num, hr_tp_den),
        "dir_hit_rate_drop": safe_div(hr_all_num, hr_all_den) - safe_div(hr_tp_num, hr_tp_den),
        "n_dir_steps": hr_all_den,
        "n_dir_tp_steps": hr_tp_den,
    }

    pooled_H3 = {}
    for H in HORIZONS:
        mc = np.asarray(h3_pool["miss_cum"][H], dtype=np.float64)
        hc = np.asarray(h3_pool["hit_cum"][H], dtype=np.float64)
        mm = float(mc.mean()) if mc.size else float("nan")
        hm = float(hc.mean()) if hc.size else float("nan")
        mt = np.asarray(h3_pool["miss_traj"][H], dtype=np.float64)
        ht = np.asarray(h3_pool["hit_traj"][H], dtype=np.float64)
        pooled_H3[str(H)] = {
            "miss_mean_cum_err": mm,
            "hit_mean_cum_err": hm,
            "ratio_miss_over_hit": (mm / hm) if (hc.size and hm != 0) else float("nan"),
            "n_miss_tp": int(mc.size),
            "n_hit_tp": int(hc.size),
            "miss_err_trajectory": [float(v) for v in mt.mean(axis=0)] if mt.size else [],
            "hit_err_trajectory": [float(v) for v in ht.mean(axis=0)] if ht.size else [],
        }

    # ---- variable rankings ----
    rank_lift_tail = sorted(per_var, key=lambda d: (d["H1"]["lift_tail_at_tp"]
                            if not np.isnan(d["H1"]["lift_tail_at_tp"]) else -1),
                            reverse=True)
    rank_err_ratio = sorted(per_var, key=lambda d: (d["H1"]["err_ratio_tp_over_non"]
                            if not np.isnan(d["H1"]["err_ratio_tp_over_non"]) else -1),
                            reverse=True)
    rank_dir_drop = sorted(per_var, key=lambda d: (d["H2"]["dir_hit_rate_drop"]
                           if not np.isnan(d["H2"]["dir_hit_rate_drop"]) else -1),
                           reverse=True)
    H10 = str(10 if 10 in HORIZONS else HORIZONS[0])
    rank_h3 = sorted(per_var, key=lambda d: (d["H3"][H10]["ratio_miss_over_hit"]
                     if not np.isnan(d["H3"][H10]["ratio_miss_over_hit"]) else -1),
                     reverse=True)

    def rk(items, path):
        out = []
        for i, d in enumerate(items):
            v = d
            for p in path:
                v = v[p]
            out.append({"rank": i + 1, "name": d["name"], "value": float(v)})
        return out

    rankings = {
        "by_tail_lift_at_tp": rk(rank_lift_tail, ["H1", "lift_tail_at_tp"]),
        "by_err_ratio_tp_over_non": rk(rank_err_ratio, ["H1", "err_ratio_tp_over_non"]),
        "by_dir_hit_rate_drop_at_tp": rk(rank_dir_drop, ["H2", "dir_hit_rate_drop"]),
        "by_h3_miss_over_hit_ratio_H10": rk(rank_h3, ["H3", H10, "ratio_miss_over_hit"]),
    }

    return {
        "thresholds": {
            "eps_percentile": EPS_PCTL,
            "eps_per_var": [float(v) for v in eps_per_var],
            "curvature_top_decile_percentile": CURV_TOP_DECILE,
            "curv_threshold_per_var": [float(v) for v in curv_thr_per_var],
            "near_window": NEAR_WINDOW,
            "tail_percentile": TAIL_PCTL,
            "horizons": list(HORIZONS),
        },
        "pooled": {"H1": pooled_H1, "H2": pooled_H2, "H3": pooled_H3},
        "per_variable": per_var,
        "rankings": rankings,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--subsample", type=int, default=SUBSAMPLE,
                    help="fixed scenario subsample (first N ids asc); 0 = all scenarios")
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
    assert nc == NUM_CONTINUOUS

    run_dir = cell["run_dir"]
    step_norm_const = float(cell["step_norm_const"])
    model, input_size = load_frozen_backbone_acc(run_dir, device)
    assert_frozen(model)   # NO retraining: hard-assert the backbone is frozen.

    feature_cols, schema_nc, num_controls = infer_schema_from_csv(cell["test_csv"])
    assert schema_nc == nc and len(feature_cols) == input_size, (
        f"schema mismatch: nc={schema_nc}/{nc} in={len(feature_cols)}/{input_size}")
    cont_cols = list(CONTINUOUS_COLS)
    control_cols = feature_cols[nc:]
    print(f"[{args.cell}] input_size={input_size} num_controls={num_controls} "
          f"controls={control_cols}")

    test_ds = AccidentWindowDataset(cell["test_csv"], seq_len=seq_len,
                                    pred_len=pred_len, cache_dir=cache_dir)
    assert test_ds.num_controls == num_controls and test_ds.input_size == input_size

    # ---- Fixed deterministic subsample: first N scenario ids ascending. ----
    # Pass-1 collect (SAME deterministic dataset order as the rollout) to get the scenario
    # set AND each scenario's init-window last continuous row (= y_true[-1] for step 0).
    print(f"[{args.cell}] Pass-1 collect (scenario order + init windows)...")
    order, init_win, cont, ctrl = _collect_pass(test_ds, collect_batch, num_workers)
    order_arr = np.asarray([int(s) for s in order], dtype=np.int64)
    sorted_ids = np.sort(order_arr)
    if args.subsample and 0 < int(args.subsample) < len(sorted_ids):
        kept_ids = sorted_ids[: int(args.subsample)]
        subsample_used = True
    else:
        kept_ids = sorted_ids
        subsample_used = False
    restrict = set(int(s) for s in kept_ids.tolist())
    S_full = len(order)
    S = len(kept_ids)
    print(f"[{args.cell}] scenarios total={S_full} using={S} "
          f"(subsample={'first %d by id asc' % args.subsample if subsample_used else 'ALL'})")

    # y_true[-1] per kept scenario = last continuous row of the init window.
    prev_true_by_scn = {}
    for s in kept_ids:
        s = int(s)
        w = init_win[s]                                       # [seq, input]
        prev_true_by_scn[s] = np.asarray(w[-1, :nc], dtype=np.float64)

    # ---- Baseline beta=0 AR rollout (REUSED, byte-identical to uncorrected AR). ----
    print(f"[{args.cell}] baseline beta=0 AR rollout (frozen backbone, error_mlp=None)...")
    t0 = time.time()
    preds_by_scn, true_by_scn = autoregressive_corrected_batched_acc(
        model, None, 0.0, test_ds, num_controls, step_norm_const,
        device=device, num_continuous=nc, collect_batch=collect_batch,
        num_workers=num_workers, restrict_scenarios=restrict)
    rollout_seconds = time.time() - t0
    # sanity: every kept scenario present
    got = set(int(k) for k in preds_by_scn.keys())
    missing = restrict - got
    assert not missing, f"rollout missing {len(missing)} kept scenarios (e.g. {list(missing)[:5]})"
    print(f"[{args.cell}] rollout done ({rollout_seconds:.1f}s), {len(got)} scenarios")

    # ---- Turning-point / directional analysis. ----
    print(f"[{args.cell}] turning-point / directional analysis...")
    ta = analyze_cell(preds_by_scn, true_by_scn, prev_true_by_scn, kept_ids, cont_cols)

    pH1 = ta["pooled"]["H1"]; pH2 = ta["pooled"]["H2"]; pH3 = ta["pooled"]["H3"]
    print(f"[{args.cell}][H1] TP base-rate={pH1['turning_point_base_rate']:.4f} "
          f"| mean err TP={pH1['mean_err_at_tp']:.6f} non={pH1['mean_err_non_tp']:.6f} "
          f"(ratio {pH1['err_ratio_tp_over_non']:.2f}x) "
          f"| share of p99 tail at TP={pH1['share_tail_at_tp']:.3f} "
          f"(lift {pH1['lift_tail_at_tp']:.2f}x)")
    print(f"[{args.cell}][H2] dir hit overall={pH2['dir_hit_rate_overall']:.4f} "
          f"at TP={pH2['dir_hit_rate_at_tp']:.4f} (drop {pH2['dir_hit_rate_drop']:+.4f})")
    H10 = str(10 if 10 in HORIZONS else HORIZONS[0])
    print(f"[{args.cell}][H3] H=10 downstream cum-err MISS={pH3[H10]['miss_mean_cum_err']:.6f} "
          f"HIT={pH3[H10]['hit_mean_cum_err']:.6f} "
          f"ratio={pH3[H10]['ratio_miss_over_hit']:.2f}x "
          f"(n_miss={pH3[H10]['n_miss_tp']} n_hit={pH3[H10]['n_hit_tp']})")

    result = {
        "cell": args.cell,
        "method": ("turning-point / directional-error AR-rollout failure-mode analysis "
                   "(frozen backbone, reuse beta=0 rollout, NO retraining)"),
        "seq_len": seq_len,
        "num_controls": num_controls,
        "input_size": input_size,
        "continuous_cols": cont_cols,
        "control_cols": control_cols,
        "step_norm_const": step_norm_const,
        "config": {
            "n_scenarios_total": int(S_full),
            "n_scenarios_used": int(S),
            "subsample": bool(subsample_used),
            "subsample_size": int(args.subsample) if subsample_used else None,
            "subsample_rule": ("first %d scenario ids ascending" % args.subsample
                               if subsample_used else "ALL scenarios"),
            "subsample_scenario_ids": [int(s) for s in kept_ids.tolist()],
            "approx_rollout_seconds": float(rollout_seconds),
            "eps_percentile": EPS_PCTL,
            "curvature_top_decile_percentile": CURV_TOP_DECILE,
            "near_window": NEAR_WINDOW,
            "tail_percentile": TAIL_PCTL,
            "horizons": list(HORIZONS),
            "error_unit": ("per-step abs error per var; pooled = mean over 10 vars "
                           "(== compute_micro_macro per-step MAE); SCALED space"),
            "definitions": {
                "delta_true": "y_true[t]-y_true[t-1] (y_true[-1]=init-window last cont row)",
                "delta_pred": "yhat[t]-y_true[t-1]",
                "turning_point": "sign(delta_true[t])!=sign(delta_true[t-1]), both |.|>eps",
                "directional_hit": "sign(delta_pred[t])==sign(delta_true[t])",
            },
        },
        "thresholds": ta["thresholds"],
        "pooled": ta["pooled"],
        "per_variable": ta["per_variable"],
        "rankings": ta["rankings"],
    }

    out_dir = os.path.join(cfg["out_root"], args.cell)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "turning_point_analysis.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{args.cell}][done] wrote {out_path}")


if __name__ == "__main__":
    main()
