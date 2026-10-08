"""Per-accident-type rollout-error dataset generation + faithful corrected AR.

ADDITIVE mirror of error_rollout.py for the per-accident-type backbones
(SimpleDecoderOnlyTransformer, seq_len=50, pred_len=1, 10 continuous + variable
controls). It does NOT modify error_rollout.py, predict_batched.py, or any of the
existing 60min-path functions. `compute_micro_macro` is REUSED from predict_batched.

Key differences vs the existing (60min, seq3, 10 continuous + 10 BINARY) path:
  * variable-width controls (num_controls = 4 for SBO/LLOCA, 5 for TLOFW), fed as
    GROUND TRUTH each rollout step (analogous to the old "binary" known-future);
  * seq_len == 50 (window slides over the full 10+controls feature row);
  * per-cell feature width  in_dim = 20 + num_controls + 1  (25 for 4-control
    cells, 26 for 5-control cells);
    features = backbone_pred(10) + last_obs_cont(10) + current controls(num_controls)
               + step_norm(1);
  * per-cell STEP_NORM_CONST (a fixed round number >= the max rollout length in the
    cell's test set; recorded in the config so step_norm means the same thing
    across scenarios and runs).

Byte-faithful lockstep: Pass-1 collects order / init window / CY / control-Y in
dataset order; Pass-2 rolls in lockstep. At beta == 0 the correction term is
SKIPPED entirely, so the corrected rollout is byte-identical to the uncorrected
baseline AR (exact null-op gate).

Runs from src/ (relative import: model_selector).
"""
from __future__ import annotations

import os
import re
import glob
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
import tqdm
import yaml

from model_selector import ModelSelector


NUM_CONTINUOUS = 10
CKPT_SUBDIR = "transformer_decoder_wonung_checkpoints_absolute"


def in_dim_for(num_controls: int) -> int:
    """Per-cell ErrorMLP input width: pred(10) + last_obs_cont(10) + controls + step(1)."""
    return 20 + int(num_controls) + 1


def build_error_features_acc(backbone_pred, last_obs_cont, cur_controls, step_idx,
                             step_norm_const, step_norm_scale=1.0):
    """Feature vector for the per-accident-type ErrorMLP.

    backbone_pred [S,10], last_obs_cont [S,10], cur_controls [S,num_controls].
    step_norm = (step_idx / step_norm_const) * step_norm_scale (absolute rollout
    step, 0-based; same meaning for every scenario / run within a cell). Returns
    [S, 20 + num_controls + 1]. `step_norm_scale`=0.0 zeroes ONLY the step feature
    (step-index ablation)."""
    S = backbone_pred.shape[0]
    step_norm = torch.full((S, 1), (step_idx / float(step_norm_const)) * step_norm_scale,
                           device=backbone_pred.device, dtype=backbone_pred.dtype)
    return torch.cat([backbone_pred, last_obs_cont, cur_controls, step_norm], dim=1)


def find_best_ckpt_acc(run_dir):
    """Pick the epoch=*.ckpt inside `<run_dir>/transformer_decoder_wonung_checkpoints_absolute/`
    (prefer the epoch ckpt, not last.ckpt; fall back to last.ckpt). If multiple
    epoch ckpts exist, take the lowest val_loss parsed from the filename."""
    ckpt_dir = os.path.join(run_dir, CKPT_SUBDIR)
    cands = [p for p in glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
             if os.path.basename(p) != "last.ckpt"]
    if not cands:
        cands = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not cands:
        return None

    def val_of(p):
        m = re.search(r"val_loss=([0-9.]+)", os.path.basename(p))
        return float(m.group(1)) if m else float("inf")
    return min(cands, key=val_of)


def load_frozen_backbone_acc(run_dir, device, ckpt=None):
    """Load the frozen per-cell backbone from `<run_dir>/config_used.yaml`
    (SINGLE SOURCE OF TRUTH for backbone_kwargs / lightning_kwargs / model.name)
    and the best epoch ckpt in the checkpoints subdir. Every param is frozen."""
    os.environ.setdefault("NONINTERACTIVE", "1")
    with open(os.path.join(run_dir, "config_used.yaml")) as f:
        cfg = yaml.safe_load(f)
    model_name = cfg["model"].get("name", "transformer_decoder")
    backbone_kwargs = cfg["model"]["backbone_kwargs"]
    lightning_kwargs = cfg["model"].get("lightning_kwargs", {})
    _, lit = ModelSelector(model_name, backbone_kwargs=backbone_kwargs,
                           lightning_kwargs=lightning_kwargs)
    ckpt_path = os.path.join(run_dir, ckpt) if ckpt else find_best_ckpt_acc(run_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint under {os.path.join(run_dir, CKPT_SUBDIR)}")
    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    lit.load_state_dict(state)                     # strict=True: fail loudly on mismatch
    model = lit.backbone.to(device=device, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, int(backbone_kwargs["input_size"])


def _collect_pass(dataset, collect_batch, num_workers, restrict_scenarios=None):
    """Pass-1 collect: per-scenario ordered init window, continuous-Y, control-Y.
    Optional scenario restriction (used for held-out beta selection)."""
    if restrict_scenarios is not None:
        restrict_scenarios = set(int(s) for s in restrict_scenarios)
    dl = DataLoader(dataset, batch_size=collect_batch, shuffle=False,
                    num_workers=num_workers)
    order, init_win = [], {}
    cont, ctrl = defaultdict(list), defaultdict(list)
    for batch in tqdm.tqdm(dl, desc="AR-collect"):
        pv = batch["past_values"].numpy()                 # [B, seq, input]
        cy = batch["continuous_y"].numpy()                # [B, 10]
        uy = batch["control_y"].numpy()                   # [B, num_controls]
        scn = batch["scenario_id"].numpy().astype(np.int64)
        for b in range(pv.shape[0]):
            s = int(scn[b])
            if restrict_scenarios is not None and s not in restrict_scenarios:
                continue
            if s not in init_win:
                init_win[s] = pv[b]
                order.append(s)
            cont[s].append(cy[b])
            ctrl[s].append(uy[b])
    return order, init_win, cont, ctrl


def _pack(order, init_win, cont, ctrl, num_continuous, num_controls):
    """Pack per-scenario lists into dense [S, maxL, .] arrays."""
    S = len(order)
    lengths = np.array([len(cont[s]) for s in order])
    maxL = int(lengths.max())
    W = np.stack([init_win[s] for s in order]).astype(np.float32)        # [S, seq, input]
    CY = np.zeros((S, maxL, num_continuous), np.float32)
    UY = np.zeros((S, maxL, num_controls), np.float32)
    for i, s in enumerate(order):
        L = lengths[i]
        CY[i, :L] = np.vstack(cont[s])
        UY[i, :L] = np.vstack(ctrl[s])
    return S, lengths, maxL, W, CY, UY


@torch.inference_mode()
def generate_rollout_error_dataset_acc(model, dataset, num_controls, step_norm_const,
                                       device="cuda", num_continuous=10,
                                       collect_batch=2048, num_workers=4,
                                       restrict_scenarios=None):
    """OPEN-LOOP (beta=0) round-0 collection. Lockstep rolls the frozen backbone
    with PREDICTED (uncorrected) continuous fed back and controls=truth. Per step t:
        feats_t = build_error_features_acc(out, window[:,-1,:10], UY[:,t,:], t, K)
        err_t   = CY[:,t,:] - out             # 10-dim error of the RAW backbone
    Returns (X[N,in_dim], Y[N,10], SID[N])."""
    in_dim = in_dim_for(num_controls)
    model = model.to(device=device, dtype=torch.float32).eval()
    order, init_win, cont, ctrl = _collect_pass(
        dataset, collect_batch, num_workers, restrict_scenarios)
    if len(order) == 0:
        return (np.zeros((0, in_dim), np.float32),
                np.zeros((0, num_continuous), np.float32),
                np.zeros((0,), np.int64))

    S, lengths, maxL, W, CY, UY = _pack(order, init_win, cont, ctrl, num_continuous, num_controls)
    window = torch.from_numpy(W).to(device)
    UY_t = torch.from_numpy(UY).to(device)
    CY_t = torch.from_numpy(CY).to(device)
    lengths_t = torch.from_numpy(lengths.astype(np.int64)).to(device)
    order_t = torch.tensor([int(s) for s in order], dtype=torch.int64, device=device)

    X_chunks, Y_chunks, SID_chunks = [], [], []
    for t in tqdm.tqdm(range(maxL), desc="AR-roll(gen)"):
        out = model(window)                                   # [S, 10]
        active = t < lengths_t
        if active.any():
            feats = build_error_features_acc(
                out, window[:, -1, :num_continuous], UY_t[:, t, :], t, step_norm_const)
            err = CY_t[:, t, :] - out                         # [S, 10]
            X_chunks.append(feats[active].detach().cpu().numpy().astype(np.float32))
            Y_chunks.append(err[active].detach().cpu().numpy().astype(np.float32))
            SID_chunks.append(order_t[active].detach().cpu().numpy().astype(np.int64))
        next_row = torch.cat([out, UY_t[:, t, :]], dim=1)     # [S, input]
        window = torch.cat([window[:, 1:, :], next_row[:, None, :]], dim=1)

    X = np.concatenate(X_chunks, axis=0) if X_chunks else np.zeros((0, in_dim), np.float32)
    Y = np.concatenate(Y_chunks, axis=0) if Y_chunks else np.zeros((0, num_continuous), np.float32)
    SID = np.concatenate(SID_chunks, axis=0) if SID_chunks else np.zeros((0,), np.int64)
    return X, Y, SID


@torch.inference_mode()
def autoregressive_corrected_batched_acc(model, error_mlp, beta, dataset, num_controls,
                                         step_norm_const, device="cuda",
                                         num_continuous=10, collect_batch=2048,
                                         num_workers=4, restrict_scenarios=None,
                                         collect_dagger=False, step_norm_scale=1.0):
    """Corrected AR eval. Lockstep rollout; the fed-back / reported value is:
        raw   = model(window)
        feats = build_error_features_acc(raw, window[:,-1,:10], UY[:,t,:], t, K)
        corr  = raw + beta * error_mlp(feats)
        next_row = cat([corr, UY[:,t]])                # CORRECTED continuous FED BACK
    beta == 0 -> correction term SKIPPED entirely -> corr is `raw` -> byte-identical
    to the uncorrected baseline AR (exact null-op).

    collect_dagger=True ALSO returns (X_d, Y_d, SID_d) collected on the CORRECTED
    trajectory (Y_d = CY - raw), enabling Phase-2 DAgger (gated; default off)."""
    in_dim = in_dim_for(num_controls)
    model = model.to(device=device, dtype=torch.float32).eval()
    if error_mlp is not None:
        error_mlp = error_mlp.to(device=device, dtype=torch.float32).eval()

    predictions_dict, true_dict = defaultdict(list), defaultdict(list)

    order, init_win, cont, ctrl = _collect_pass(
        dataset, collect_batch, num_workers, restrict_scenarios)
    if len(order) == 0:
        if collect_dagger:
            return (predictions_dict, true_dict,
                    np.zeros((0, in_dim), np.float32),
                    np.zeros((0, num_continuous), np.float32),
                    np.zeros((0,), np.int64))
        return predictions_dict, true_dict

    S, lengths, maxL, W, CY, UY = _pack(order, init_win, cont, ctrl, num_continuous, num_controls)
    window = torch.from_numpy(W).to(device)
    UY_t = torch.from_numpy(UY).to(device)
    preds = np.zeros((S, maxL, num_continuous), np.float32)

    do_corr = (beta != 0.0) and (error_mlp is not None)

    if collect_dagger:
        CY_t = torch.from_numpy(CY).to(device)
        lengths_t = torch.from_numpy(lengths.astype(np.int64)).to(device)
        order_t = torch.tensor([int(s) for s in order], dtype=torch.int64, device=device)
        Xd_chunks, Yd_chunks, SIDd_chunks = [], [], []

    for t in tqdm.tqdm(range(maxL), desc="AR-roll(corr)"):
        raw = model(window)                                   # [S, 10]
        if do_corr or collect_dagger:
            feats = build_error_features_acc(
                raw, window[:, -1, :num_continuous], UY_t[:, t, :], t,
                step_norm_const, step_norm_scale=step_norm_scale)
        if do_corr:
            corr = raw + beta * error_mlp(feats)              # [S, 10]
        else:
            corr = raw                                        # exact null-op at beta==0
        preds[:, t, :] = corr.detach().cpu().numpy()
        if collect_dagger:
            active = t < lengths_t
            if active.any():
                err = CY_t[:, t, :] - raw
                Xd_chunks.append(feats[active].detach().cpu().numpy().astype(np.float32))
                Yd_chunks.append(err[active].detach().cpu().numpy().astype(np.float32))
                SIDd_chunks.append(order_t[active].detach().cpu().numpy().astype(np.int64))
        next_row = torch.cat([corr, UY_t[:, t, :]], dim=1)    # [S, input]
        window = torch.cat([window[:, 1:, :], next_row[:, None, :]], dim=1)

    for i, s in enumerate(order):
        L = lengths[i]
        for t in range(L):
            predictions_dict[s].append(preds[i, t])
            true_dict[s].append(CY[i, t])

    if collect_dagger:
        Xd = np.concatenate(Xd_chunks, axis=0) if Xd_chunks else np.zeros((0, in_dim), np.float32)
        Yd = np.concatenate(Yd_chunks, axis=0) if Yd_chunks else np.zeros((0, num_continuous), np.float32)
        SIDd = np.concatenate(SIDd_chunks, axis=0) if SIDd_chunks else np.zeros((0,), np.int64)
        return predictions_dict, true_dict, Xd, Yd, SIDd
    return predictions_dict, true_dict


# ---------------------------------------------------------------------------
# Gated (selective) tail-correction primitives (ADDITIVE; used only by
# experiments/tail_analysis_acc.py). These do NOT change the behavior of any
# function above -- they add two new read-only-on-the-backbone rollouts.
# ---------------------------------------------------------------------------


@torch.inference_mode()
def baseline_rollout_with_stats_acc(model, error_mlp, dataset, num_controls,
                                    step_norm_const, device="cuda",
                                    num_continuous=10, collect_batch=2048,
                                    num_workers=4, restrict_scenarios=None,
                                    step_norm_scale=1.0):
    """UNCORRECTED (beta=0) lockstep AR that ALSO collects, per ACTIVE step, the
    aligned flat arrays needed for gated tail analysis. Computing e_hat here does
    NOT change the trajectory: the fed-back value is always the RAW backbone
    prediction (exact beta=0 baseline), so `step_err` below is the true baseline
    per-step error distribution.

    Per active step t the following aligned scalars are collected:
        step_err = mean_k |CY[:,t,k] - raw[:,k]|   (per-step error, SAME unit as
                   compute_micro_macro's per-step MAE)
        g        = || error_mlp(feats) ||_2        (predicted-error L2; the gate
                   score for gate_on='pred')
        true_g   = || CY[:,t] - raw ||_2           (realized-error L2; the gate
                   score for gate_on='true' = ORACLE)
        sid      = scenario id (for worst-scenario aggregation)

    Returns dict with float32 1-D arrays {step_err, g, true_g, sid} (sid int64),
    all aligned and pooled over (scenario, step). error_mlp must be non-None (g
    needs it)."""
    assert error_mlp is not None, "baseline_rollout_with_stats_acc needs error_mlp for g"
    model = model.to(device=device, dtype=torch.float32).eval()
    error_mlp = error_mlp.to(device=device, dtype=torch.float32).eval()

    order, init_win, cont, ctrl = _collect_pass(
        dataset, collect_batch, num_workers, restrict_scenarios)
    if len(order) == 0:
        z = np.zeros((0,), np.float32)
        return {"step_err": z, "g": z.copy(), "true_g": z.copy(),
                "sid": np.zeros((0,), np.int64)}

    S, lengths, maxL, W, CY, UY = _pack(order, init_win, cont, ctrl, num_continuous, num_controls)
    window = torch.from_numpy(W).to(device)
    UY_t = torch.from_numpy(UY).to(device)
    CY_t = torch.from_numpy(CY).to(device)
    lengths_t = torch.from_numpy(lengths.astype(np.int64)).to(device)
    order_t = torch.tensor([int(s) for s in order], dtype=torch.int64, device=device)

    se_chunks, g_chunks, tg_chunks, sid_chunks = [], [], [], []
    for t in tqdm.tqdm(range(maxL), desc="AR-roll(base+stats)"):
        raw = model(window)                                   # [S, 10]
        active = t < lengths_t
        if active.any():
            feats = build_error_features_acc(
                raw, window[:, -1, :num_continuous], UY_t[:, t, :], t,
                step_norm_const, step_norm_scale=step_norm_scale)
            e_hat = error_mlp(feats)                           # [S, 10]
            diff = CY_t[:, t, :] - raw                         # [S, 10]
            step_err = diff.abs().mean(dim=1)                  # [S]  per-step MAE
            g = torch.linalg.vector_norm(e_hat, ord=2, dim=1)  # [S]
            true_g = torch.linalg.vector_norm(diff, ord=2, dim=1)  # [S]
            se_chunks.append(step_err[active].detach().cpu().numpy().astype(np.float32))
            g_chunks.append(g[active].detach().cpu().numpy().astype(np.float32))
            tg_chunks.append(true_g[active].detach().cpu().numpy().astype(np.float32))
            sid_chunks.append(order_t[active].detach().cpu().numpy().astype(np.int64))
        # beta==0: RAW is fed back (exact uncorrected baseline trajectory).
        next_row = torch.cat([raw, UY_t[:, t, :]], dim=1)     # [S, input]
        window = torch.cat([window[:, 1:, :], next_row[:, None, :]], dim=1)

    cat = lambda cs, dt: (np.concatenate(cs, axis=0) if cs else np.zeros((0,), dt))
    return {
        "step_err": cat(se_chunks, np.float32),
        "g": cat(g_chunks, np.float32),
        "true_g": cat(tg_chunks, np.float32),
        "sid": cat(sid_chunks, np.int64),
    }


@torch.inference_mode()
def gated_corrected_rollout_acc(model, error_mlp, beta, tau, dataset, num_controls,
                                step_norm_const, device="cuda", gate_on="pred",
                                num_continuous=10, collect_batch=2048,
                                num_workers=4, restrict_scenarios=None,
                                step_norm_scale=1.0, collect_dagger=False):
    """GATED (selective) corrected AR. Lockstep rollout where at each step the
    correction `raw + beta*e_hat` is applied ONLY where the gate fires; elsewhere
    the value is `raw`. The chosen value (corr where gated, raw otherwise) is BOTH
    fed back AND reported.

    Gate:
        gate_on='pred' -> fire where g      = ||e_hat||_2      >  tau
        gate_on='true' -> fire where true_g = ||CY - raw||_2   >  tau   (ORACLE)

    Null-op contract: beta == 0.0 or tau == +inf => gate never applies a nonzero
    correction => the reported/fed-back trajectory is byte-identical to the
    uncorrected baseline AR. (Asserted once before the loop.)

    Returns (predictions_dict, true_dict) keyed by scenario id, exactly like
    autoregressive_corrected_batched_acc, so compute_micro_macro / per-step tail
    metrics consume it unchanged.

    collect_dagger=True (ADDITIVE; used by Phase-2 DAgger, V1) ALSO returns
    (X_d, Y_d, SID_d, interv): per ACTIVE step of THIS gated corrected trajectory,
    X_d = error features, Y_d = residual (CY - raw) of the RAW backbone on the
    corrected window (the target the ErrorMLP should predict at the corrected
    state), SID_d = scenario id. `interv` accounts for the whole-step gate over
    ACTIVE steps: {n_active_steps, n_fired_steps, intervention_rate}. At beta==0
    (null-op) the gate never fires (intervention_rate == 0) and Y_d is the
    open-loop residual, so a DAgger round at beta==0 collapses to round-0 data."""
    assert gate_on in ("pred", "true"), f"gate_on must be 'pred' or 'true', got {gate_on}"
    model = model.to(device=device, dtype=torch.float32).eval()
    if error_mlp is not None:
        error_mlp = error_mlp.to(device=device, dtype=torch.float32).eval()

    # Null-op assertion: beta==0 or tau==inf must be an exact baseline.
    null_op = (float(beta) == 0.0) or (not np.isfinite(tau)) or (error_mlp is None)
    do_corr = not null_op
    need_feats = do_corr or collect_dagger

    predictions_dict, true_dict = defaultdict(list), defaultdict(list)
    in_dim = in_dim_for(num_controls)
    zero_interv = {"n_active_steps": 0, "n_fired_steps": 0, "intervention_rate": 0.0}

    order, init_win, cont, ctrl = _collect_pass(
        dataset, collect_batch, num_workers, restrict_scenarios)
    if len(order) == 0:
        if collect_dagger:
            return (predictions_dict, true_dict,
                    np.zeros((0, in_dim), np.float32),
                    np.zeros((0, num_continuous), np.float32),
                    np.zeros((0,), np.int64), zero_interv)
        return predictions_dict, true_dict

    S, lengths, maxL, W, CY, UY = _pack(order, init_win, cont, ctrl, num_continuous, num_controls)
    window = torch.from_numpy(W).to(device)
    UY_t = torch.from_numpy(UY).to(device)
    CY_t = torch.from_numpy(CY).to(device)
    preds = np.zeros((S, maxL, num_continuous), np.float32)
    tau_t = None if null_op else torch.as_tensor(float(tau), device=device, dtype=torch.float32)

    if collect_dagger:
        lengths_t = torch.from_numpy(lengths.astype(np.int64)).to(device)
        order_t = torch.tensor([int(s) for s in order], dtype=torch.int64, device=device)
        Xd_chunks, Yd_chunks, SIDd_chunks = [], [], []
        n_active_steps, n_fired_steps = 0, 0

    for t in tqdm.tqdm(range(maxL), desc="AR-roll(gated)"):
        raw = model(window)                                   # [S, 10]
        if need_feats:
            feats = build_error_features_acc(
                raw, window[:, -1, :num_continuous], UY_t[:, t, :], t,
                step_norm_const, step_norm_scale=step_norm_scale)
        if do_corr:
            e_hat = error_mlp(feats)                           # [S, 10]
            if gate_on == "pred":
                score = torch.linalg.vector_norm(e_hat, ord=2, dim=1)          # [S]
            else:  # 'true' = ORACLE gate on realized error
                score = torch.linalg.vector_norm(CY_t[:, t, :] - raw, ord=2, dim=1)
            fire = (score > tau_t)                            # [S] bool (whole-step gate)
            corr_full = raw + beta * e_hat                    # [S, 10]
            chosen = torch.where(fire.unsqueeze(1), corr_full, raw)  # gated select
        else:
            chosen = raw                                      # exact null-op
            fire = None
        preds[:, t, :] = chosen.detach().cpu().numpy()
        if collect_dagger:
            active = t < lengths_t                            # [S] bool
            if active.any():
                err = CY_t[:, t, :] - raw                     # [S, 10]
                Xd_chunks.append(feats[active].detach().cpu().numpy().astype(np.float32))
                Yd_chunks.append(err[active].detach().cpu().numpy().astype(np.float32))
                SIDd_chunks.append(order_t[active].detach().cpu().numpy().astype(np.int64))
            n_active_steps += int(active.sum().item())
            if fire is not None:
                n_fired_steps += int((fire & active).sum().item())
        next_row = torch.cat([chosen, UY_t[:, t, :]], dim=1)  # [S, input]
        window = torch.cat([window[:, 1:, :], next_row[:, None, :]], dim=1)

    for i, s in enumerate(order):
        L = lengths[i]
        for t in range(L):
            predictions_dict[s].append(preds[i, t])
            true_dict[s].append(CY[i, t])

    if collect_dagger:
        Xd = np.concatenate(Xd_chunks, axis=0) if Xd_chunks else np.zeros((0, in_dim), np.float32)
        Yd = np.concatenate(Yd_chunks, axis=0) if Yd_chunks else np.zeros((0, num_continuous), np.float32)
        SIDd = np.concatenate(SIDd_chunks, axis=0) if SIDd_chunks else np.zeros((0,), np.int64)
        interv = {
            "n_active_steps": n_active_steps, "n_fired_steps": n_fired_steps,
            "intervention_rate": (float(n_fired_steps) / float(n_active_steps)
                                  if n_active_steps > 0 else 0.0),
        }
        return predictions_dict, true_dict, Xd, Yd, SIDd, interv
    return predictions_dict, true_dict


# ---------------------------------------------------------------------------
# Per-(step, variable) gated tail-correction primitives (ADDITIVE; used only by
# experiments/variable_gating_acc.py). These do NOT touch any function above.
# They add (a) a baseline pass that collects per-variable predicted-error
# magnitudes for per-variable gate calibration, and (b) a rollout that gates the
# correction per CELL (step, variable) -- a strict superset of the step gate.
# ---------------------------------------------------------------------------


@torch.inference_mode()
def baseline_var_stats_acc(model, error_mlp, dataset, num_controls,
                           step_norm_const, device="cuda",
                           num_continuous=10, collect_batch=2048,
                           num_workers=4, restrict_scenarios=None,
                           step_norm_scale=1.0):
    """UNCORRECTED (beta=0) lockstep AR that collects PER-(active step, variable)
    stats for per-variable gate calibration. The fed-back value is always the RAW
    backbone prediction (exact beta=0 baseline), so these are the TRUE baseline
    distributions (computing e_hat here does not change the trajectory).

    Per active step t the following aligned rows are collected:
        abs_ehat = |error_mlp(feats)|          [.,10]  per-variable predicted-error
                   magnitude -> the gate score for the per-(step,variable) gate.
        abs_err  = |CY[:,t] - raw|             [.,10]  realized per-variable error.
        g        = ||error_mlp(feats)||_2      [.]     step-level pred-error L2 (the
                   step-gate score; == baseline_rollout_with_stats_acc's g).
        step_err = mean_k |CY[:,t,k]-raw[:,k]| [.]     per-step MAE (compute_micro_macro unit).
        sid      = scenario id                 [.]

    Returns {abs_ehat [N,10], abs_err [N,10], g [N], step_err [N], sid [N]} (float32,
    sid int64), all aligned and pooled over (scenario, step). error_mlp must be
    non-None (abs_ehat/g need it)."""
    assert error_mlp is not None, "baseline_var_stats_acc needs error_mlp for abs_ehat/g"
    model = model.to(device=device, dtype=torch.float32).eval()
    error_mlp = error_mlp.to(device=device, dtype=torch.float32).eval()

    order, init_win, cont, ctrl = _collect_pass(
        dataset, collect_batch, num_workers, restrict_scenarios)
    if len(order) == 0:
        z1 = np.zeros((0,), np.float32)
        z2 = np.zeros((0, num_continuous), np.float32)
        return {"abs_ehat": z2, "abs_err": z2.copy(), "g": z1, "step_err": z1.copy(),
                "sid": np.zeros((0,), np.int64)}

    S, lengths, maxL, W, CY, UY = _pack(order, init_win, cont, ctrl, num_continuous, num_controls)
    window = torch.from_numpy(W).to(device)
    UY_t = torch.from_numpy(UY).to(device)
    CY_t = torch.from_numpy(CY).to(device)
    lengths_t = torch.from_numpy(lengths.astype(np.int64)).to(device)
    order_t = torch.tensor([int(s) for s in order], dtype=torch.int64, device=device)

    ae_chunks, aerr_chunks, g_chunks, se_chunks, sid_chunks = [], [], [], [], []
    for t in tqdm.tqdm(range(maxL), desc="AR-roll(var-stats)"):
        raw = model(window)                                   # [S, 10]
        active = t < lengths_t
        if active.any():
            feats = build_error_features_acc(
                raw, window[:, -1, :num_continuous], UY_t[:, t, :], t,
                step_norm_const, step_norm_scale=step_norm_scale)
            e_hat = error_mlp(feats)                           # [S, 10]
            diff = CY_t[:, t, :] - raw                         # [S, 10]
            abs_ehat = e_hat.abs()                             # [S, 10]
            abs_err = diff.abs()                               # [S, 10]
            g = torch.linalg.vector_norm(e_hat, ord=2, dim=1)  # [S]
            step_err = abs_err.mean(dim=1)                     # [S]
            ae_chunks.append(abs_ehat[active].detach().cpu().numpy().astype(np.float32))
            aerr_chunks.append(abs_err[active].detach().cpu().numpy().astype(np.float32))
            g_chunks.append(g[active].detach().cpu().numpy().astype(np.float32))
            se_chunks.append(step_err[active].detach().cpu().numpy().astype(np.float32))
            sid_chunks.append(order_t[active].detach().cpu().numpy().astype(np.int64))
        # beta==0: RAW is fed back (exact uncorrected baseline trajectory).
        next_row = torch.cat([raw, UY_t[:, t, :]], dim=1)     # [S, input]
        window = torch.cat([window[:, 1:, :], next_row[:, None, :]], dim=1)

    catf = lambda cs, w: (np.concatenate(cs, axis=0) if cs else np.zeros((0, w), np.float32))
    cat1 = lambda cs, dt: (np.concatenate(cs, axis=0) if cs else np.zeros((0,), dt))
    return {
        "abs_ehat": catf(ae_chunks, num_continuous),
        "abs_err": catf(aerr_chunks, num_continuous),
        "g": cat1(g_chunks, np.float32),
        "step_err": cat1(se_chunks, np.float32),
        "sid": cat1(sid_chunks, np.int64),
    }


@torch.inference_mode()
def var_gated_corrected_rollout_acc(model, error_mlp, beta, dataset, num_controls,
                                    step_norm_const, gate="var", tau_vec=None,
                                    tau_step=None, device="cuda", num_continuous=10,
                                    collect_batch=2048, num_workers=4,
                                    restrict_scenarios=None, step_norm_scale=1.0):
    """PER-(step, variable) gated corrected AR (ADDITIVE; a superset of the step gate).

    At each rollout step t the correction `raw + beta*e_hat` is applied per CELL
    (step t, variable j) ONLY where the gate fires; elsewhere the value is `raw`.
    The MIXED row (corrected j's + raw others) is BOTH fed back AND reported.

    gate:
      'var'  -> fire cell (t,j) where |e_hat_{t,j}| > tau_vec[j]  (per-variable
                threshold; the per-(step,variable) DYNAMIC gate). A fixed-variable-set
                gate is the special case tau_vec[j] = -inf for the chosen j's (always
                corrected) and +inf elsewhere (never corrected).
      'step' -> fire the WHOLE step (all vars) where ||e_hat_t||_2 > tau_step
                (reproduces gated_corrected_rollout_acc's step gate byte-for-byte,
                but ALSO reports the (step,variable)-cell intervention rate so step-
                and variable-gating are directly comparable on "cells touched").

    Null-op: beta==0, or error_mlp is None, or ('step' with non-finite tau_step),
    or ('var' with every tau_vec entry +inf) => byte-identical uncorrected baseline
    AR (asserted before the loop; the reported/fed-back trajectory is exactly `raw`).

    Returns (predictions_dict, true_dict, interv). `interv` is the realized
    (corrected-trajectory) intervention accounting over ACTIVE (step,variable)
    cells: {n_active_cells, n_fired, intervention_rate, per_var_active[10],
    per_var_fired[10], per_var_rate[10]}."""
    assert gate in ("var", "step"), f"gate must be 'var' or 'step', got {gate}"
    model = model.to(device=device, dtype=torch.float32).eval()
    if error_mlp is not None:
        error_mlp = error_mlp.to(device=device, dtype=torch.float32).eval()

    if gate == "var":
        assert tau_vec is not None, "gate='var' needs tau_vec [num_continuous]"
        tv = np.asarray(tau_vec, dtype=np.float32).reshape(-1)
        assert tv.shape[0] == num_continuous, f"tau_vec must be length {num_continuous}"
        empty_gate = bool(np.all(np.isposinf(tv)))            # all +inf -> never fires
    else:
        empty_gate = (tau_step is None) or (not np.isfinite(tau_step))

    null_op = (float(beta) == 0.0) or (error_mlp is None) or empty_gate
    do_corr = not null_op

    predictions_dict, true_dict = defaultdict(list), defaultdict(list)
    zero_interv = {
        "n_active_cells": 0, "n_fired": 0, "intervention_rate": 0.0,
        "per_var_active": [0] * num_continuous, "per_var_fired": [0] * num_continuous,
        "per_var_rate": [0.0] * num_continuous,
    }

    order, init_win, cont, ctrl = _collect_pass(
        dataset, collect_batch, num_workers, restrict_scenarios)
    if len(order) == 0:
        return predictions_dict, true_dict, zero_interv

    S, lengths, maxL, W, CY, UY = _pack(order, init_win, cont, ctrl, num_continuous, num_controls)
    window = torch.from_numpy(W).to(device)
    UY_t = torch.from_numpy(UY).to(device)
    lengths_t = torch.from_numpy(lengths.astype(np.int64)).to(device)
    preds = np.zeros((S, maxL, num_continuous), np.float32)

    if do_corr and gate == "var":
        tau_t = torch.from_numpy(tv).to(device)               # [10]
    elif do_corr:
        tau_t = torch.as_tensor(float(tau_step), device=device, dtype=torch.float32)

    per_var_fired = torch.zeros(num_continuous, dtype=torch.int64, device=device)
    per_var_active = torch.zeros(num_continuous, dtype=torch.int64, device=device)

    for t in tqdm.tqdm(range(maxL), desc="AR-roll(var-gated)"):
        raw = model(window)                                   # [S, 10]
        active = t < lengths_t                                # [S] bool
        per_var_active += active.sum().to(torch.int64)        # scalar broadcast over [10]
        if do_corr:
            feats = build_error_features_acc(
                raw, window[:, -1, :num_continuous], UY_t[:, t, :], t,
                step_norm_const, step_norm_scale=step_norm_scale)
            e_hat = error_mlp(feats)                           # [S, 10]
            if gate == "var":
                fire = e_hat.abs() > tau_t                     # [S, 10] per-variable
            else:  # 'step': whole-step fire broadcast to all vars
                score = torch.linalg.vector_norm(e_hat, ord=2, dim=1)          # [S]
                fire = (score > tau_t).unsqueeze(1).expand(-1, num_continuous)  # [S, 10]
            corr_full = raw + beta * e_hat                    # [S, 10]
            chosen = torch.where(fire, corr_full, raw)        # mixed row (gated cells)
            per_var_fired += (fire & active.unsqueeze(1)).sum(dim=0).to(torch.int64)
        else:
            chosen = raw                                      # exact null-op
        preds[:, t, :] = chosen.detach().cpu().numpy()
        next_row = torch.cat([chosen, UY_t[:, t, :]], dim=1)  # [S, input]
        window = torch.cat([window[:, 1:, :], next_row[:, None, :]], dim=1)

    for i, s in enumerate(order):
        L = lengths[i]
        for t in range(L):
            predictions_dict[s].append(preds[i, t])
            true_dict[s].append(CY[i, t])

    pva = per_var_active.detach().cpu().numpy().astype(np.int64)
    pvf = per_var_fired.detach().cpu().numpy().astype(np.int64)
    n_active_cells = int(pva.sum())
    n_fired = int(pvf.sum())
    per_var_rate = [float(pvf[j]) / float(pva[j]) if pva[j] > 0 else 0.0
                    for j in range(num_continuous)]
    interv = {
        "n_active_cells": n_active_cells,
        "n_fired": n_fired,
        "intervention_rate": (float(n_fired) / float(n_active_cells)
                              if n_active_cells > 0 else 0.0),
        "per_var_active": [int(x) for x in pva],
        "per_var_fired": [int(x) for x in pvf],
        "per_var_rate": per_var_rate,
    }
    return predictions_dict, true_dict, interv


# ---------------------------------------------------------------------------
# OOD-AWARE (Mahalanobis-gated) selective tail-correction (ADDITIVE; used only by
# experiments/ood_aware_gate.py). A STRICT TIGHTENING of gated_corrected_rollout_acc:
# a step's correction fires only if the predicted-error gate fires AND the ErrorMLP
# INPUT feature z_t is in-distribution (Mahalanobis D_M(z_t) <= tau_ood). The OOD
# gate can only SUPPRESS corrections, never create one, so the beta=0 / tau=inf
# null-op is preserved byte-for-byte. Does NOT touch any function above.
# ---------------------------------------------------------------------------


@torch.inference_mode()
def ood_gated_corrected_rollout_acc(model, error_mlp, beta, tau, dataset, num_controls,
                                    step_norm_const, ood_mu, ood_std, ood_sigma_inv,
                                    tau_ood, device="cuda", gate_on="pred",
                                    num_continuous=10, collect_batch=2048, num_workers=4,
                                    restrict_scenarios=None, step_norm_scale=1.0):
    """OOD-AWARE gated corrected AR. At each rollout step the correction
    `raw + beta*e_hat` is applied ONLY where BOTH gates fire; elsewhere `raw`:
        (i)  predicted-error gate: ||e_hat||_2 > tau   (gate_on='pred'; the existing
             step gate) [or the ORACLE ||CY-raw||_2 > tau for gate_on='true'], AND
        (ii) in-distribution:      D_M(z_t) <= tau_ood, where z_t is the ErrorMLP
             INPUT feature (`feats`) and
                 D_M(z) = sqrt( u^T Sigma_inv u ),  u = (z - mu) / std
             (std=None -> no standardization). `ood_mu` [in_dim], `ood_std` [in_dim]
             or None, `ood_sigma_inv` [in_dim, in_dim] (inverse of the REGULARIZED
             covariance in the standardized space). Fit on the corrector's TRAIN
             features; tau_ood = high percentile of the in-distribution D_M.
    Where D_M(z_t) > tau_ood the step ABSTAINS (uses raw = beta-0), even if the
    predicted-error gate fired. The chosen row (corr where BOTH fire, raw otherwise)
    is BOTH fed back AND reported.

    Null-op contract: beta==0 or tau==+inf or error_mlp is None => the correction
    branch is skipped entirely => byte-identical uncorrected baseline AR. The OOD
    gate only SUPPRESSES corrections (fire is a subset of the predicted-error gate),
    so it can never break the null-op.

    Returns (predictions_dict, true_dict, abstain) keyed by scenario id (same dict
    path as gated_corrected_rollout_acc, so the shared per-step tail metric consumes
    (predictions_dict, true_dict) unchanged). `abstain` accounts over ACTIVE steps:
        n_active_steps    : active (scenario, step) count
        n_ood             : D_M > tau_ood (flagged OOD) -> abstained
        n_pred_fire       : predicted-error gate fired (BEFORE the OOD veto)
        n_fire            : BOTH gates fired (correction actually applied)
        n_suppressed      : pred_fire AND OOD (corrections vetoed by the OOD gate)
        ood_abstain_rate  : n_ood / n_active_steps  (fraction of steps flagged OOD)
        gate_fire_rate    : n_fire / n_active_steps (fraction actually corrected)
        vanilla_fire_rate : n_pred_fire / n_active_steps (fraction the vanilla gate
                            would have corrected)
        suppression_rate  : n_suppressed / n_pred_fire (fraction of vanilla-gate
                            corrections vetoed as OOD)
    """
    assert gate_on in ("pred", "true"), f"gate_on must be 'pred' or 'true', got {gate_on}"
    model = model.to(device=device, dtype=torch.float32).eval()
    if error_mlp is not None:
        error_mlp = error_mlp.to(device=device, dtype=torch.float32).eval()

    null_op = (float(beta) == 0.0) or (not np.isfinite(tau)) or (error_mlp is None)
    do_corr = not null_op

    mu_t = torch.as_tensor(np.asarray(ood_mu, np.float32), device=device)
    std_t = (torch.as_tensor(np.asarray(ood_std, np.float32), device=device)
             if ood_std is not None else None)
    sinv_t = torch.as_tensor(np.asarray(ood_sigma_inv, np.float32), device=device)
    tau_ood_f = float(tau_ood)

    predictions_dict, true_dict = defaultdict(list), defaultdict(list)
    zero_abstain = {"n_active_steps": 0, "n_ood": 0, "n_pred_fire": 0, "n_fire": 0,
                    "n_suppressed": 0, "ood_abstain_rate": 0.0, "gate_fire_rate": 0.0,
                    "vanilla_fire_rate": 0.0, "suppression_rate": 0.0}

    order, init_win, cont, ctrl = _collect_pass(
        dataset, collect_batch, num_workers, restrict_scenarios)
    if len(order) == 0:
        return predictions_dict, true_dict, zero_abstain

    S, lengths, maxL, W, CY, UY = _pack(order, init_win, cont, ctrl, num_continuous, num_controls)
    window = torch.from_numpy(W).to(device)
    UY_t = torch.from_numpy(UY).to(device)
    CY_t = torch.from_numpy(CY).to(device) if (do_corr and gate_on == "true") else None
    lengths_t = torch.from_numpy(lengths.astype(np.int64)).to(device)
    preds = np.zeros((S, maxL, num_continuous), np.float32)
    tau_t = None if null_op else torch.as_tensor(float(tau), device=device, dtype=torch.float32)

    n_active_steps = n_ood = n_pred_fire = n_fire = n_suppressed = 0

    for t in tqdm.tqdm(range(maxL), desc="AR-roll(ood-gated)"):
        raw = model(window)                                   # [S, 10]
        active = t < lengths_t                                # [S] bool
        if do_corr:
            feats = build_error_features_acc(
                raw, window[:, -1, :num_continuous], UY_t[:, t, :], t,
                step_norm_const, step_norm_scale=step_norm_scale)
            e_hat = error_mlp(feats)                           # [S, 10]
            if gate_on == "pred":
                score = torch.linalg.vector_norm(e_hat, ord=2, dim=1)          # [S]
            else:  # 'true' = ORACLE gate on realized error
                score = torch.linalg.vector_norm(CY_t[:, t, :] - raw, ord=2, dim=1)
            pred_fire = score > tau_t                          # [S] predicted-error gate
            u = feats - mu_t
            if std_t is not None:
                u = u / std_t
            dm = torch.sqrt(((u @ sinv_t) * u).sum(dim=1).clamp_min(0))  # [S] D_M(z_t)
            ood = dm > tau_ood_f                               # [S] out-of-distribution
            fire = pred_fire & (~ood)                          # BOTH gates
            corr_full = raw + beta * e_hat                     # [S, 10]
            chosen = torch.where(fire.unsqueeze(1), corr_full, raw)
            # accounting over ACTIVE steps
            n_active_steps += int(active.sum().item())
            n_ood += int((ood & active).sum().item())
            n_pred_fire += int((pred_fire & active).sum().item())
            n_fire += int((fire & active).sum().item())
            n_suppressed += int((pred_fire & ood & active).sum().item())
        else:
            chosen = raw                                       # exact null-op
            n_active_steps += int(active.sum().item())
        preds[:, t, :] = chosen.detach().cpu().numpy()
        next_row = torch.cat([chosen, UY_t[:, t, :]], dim=1)   # [S, input]
        window = torch.cat([window[:, 1:, :], next_row[:, None, :]], dim=1)

    for i, s in enumerate(order):
        L = lengths[i]
        for t in range(L):
            predictions_dict[s].append(preds[i, t])
            true_dict[s].append(CY[i, t])

    abstain = {
        "n_active_steps": n_active_steps, "n_ood": n_ood, "n_pred_fire": n_pred_fire,
        "n_fire": n_fire, "n_suppressed": n_suppressed,
        "ood_abstain_rate": (float(n_ood) / n_active_steps if n_active_steps > 0 else 0.0),
        "gate_fire_rate": (float(n_fire) / n_active_steps if n_active_steps > 0 else 0.0),
        "vanilla_fire_rate": (float(n_pred_fire) / n_active_steps if n_active_steps > 0 else 0.0),
        "suppression_rate": (float(n_suppressed) / n_pred_fire if n_pred_fire > 0 else 0.0),
    }
    return predictions_dict, true_dict, abstain


# ---------------------------------------------------------------------------
# PHYSICS-PLAUSIBILITY-CONSTRAINED selective tail-correction (ADDITIVE; used only
# by experiments/physics_constrained_gate.py). Roadmap step #2 / method B4. Before
# a GATED correction is fed back, its 10-dim continuous vector is PROJECTED onto a
# physically plausible set fit from TRAIN:
#     corr <- clip(clip(raw + beta*e_hat, lo, hi), prev - dmax, prev + dmax)
#   (i)  BOUNDS: clip each variable k to [lo_k, hi_k] (per-variable TRAIN range);
#   (ii) RATE LIMIT: cap the per-step change to |corr_k - prev_k| <= dmax_k, where
#        prev_k = the last fed continuous value (window[:, -1, :10]) and dmax_k is a
#        high percentile of the per-variable |y_t - y_{t-1}| seen in TRAIN.
# The projection is applied ONLY to the value that is actually fed back where the
# gate fires (raw elsewhere), so a step that does not fire -- and, in particular,
# beta==0 / tau==inf -- is the byte-identical uncorrected baseline (null-op). The
# existing predicted-error gate and, optionally, the A1 OOD (Mahalanobis) gate
# select WHERE to correct; physics constrains HOW LARGE the fed-back correction can
# be, capping the OOD blowups the gates miss. Does NOT touch any function above.
# ---------------------------------------------------------------------------


@torch.inference_mode()
def physics_gated_corrected_rollout_acc(model, error_mlp, beta, tau, dataset, num_controls,
                                        step_norm_const, phys_lo, phys_hi, phys_dmax,
                                        ood_mu=None, ood_std=None, ood_sigma_inv=None,
                                        tau_ood=None, device="cuda", gate_on="pred",
                                        num_continuous=10, collect_batch=2048, num_workers=4,
                                        restrict_scenarios=None, step_norm_scale=1.0):
    """PHYSICS-CONSTRAINED gated corrected AR. At each rollout step the correction
    `raw + beta*e_hat` is applied ONLY where the gate(s) fire; the fed-back/reported
    value there is the PROJECTED correction:
        corr_proj = clip(clip(raw + beta*e_hat, lo, hi), prev - dmax, prev + dmax)
    where prev = window[:, -1, :num_continuous] (the last fed continuous value),
    `phys_lo`/`phys_hi`/`phys_dmax` are per-variable [num_continuous] arrays (TRAIN-
    fit bounds + max step size). Elsewhere the value is `raw`.

    Gates (a step fires where BOTH hold):
        (i)  predicted-error gate: ||e_hat||_2 > tau (gate_on='pred'; the existing
             step gate) [or the ORACLE ||CY-raw||_2 > tau for gate_on='true'];
        (ii) OPTIONAL A1 OOD gate: if `ood_mu` is not None, additionally require the
             ErrorMLP INPUT feature z_t to be in-distribution, D_M(z_t) <= tau_ood,
             D_M(z) = sqrt(u^T Sigma_inv u), u = (z-mu)/std. `ood_mu` is None => no
             OOD gate (physics composes with the vanilla predicted-error gate only).

    Null-op contract: beta==0 or tau==+inf or error_mlp is None => the correction
    branch is skipped entirely => byte-identical uncorrected baseline AR. The
    projection only reshapes the fired correction (never raw), so it can never break
    the null-op. (Asserted by the caller via the beta=0 baseline == open-loop mean.)

    Returns (predictions_dict, true_dict, stats) keyed by scenario id (same dict path
    as gated_corrected_rollout_acc). `stats` accounts over ACTIVE steps:
        n_active_steps, n_pred_fire (predicted-error gate), n_ood (D_M > tau_ood; 0 if
        no OOD gate), n_fire (both gates -> correction applied), n_suppressed
        (pred_fire AND OOD), n_proj_clipped (fired steps whose projection changed at
        least one variable), plus ood_abstain_rate / gate_fire_rate / vanilla_fire_rate
        / suppression_rate / proj_clip_rate (n_proj_clipped / n_fire) and use_ood_gate.
    """
    assert gate_on in ("pred", "true"), f"gate_on must be 'pred' or 'true', got {gate_on}"
    model = model.to(device=device, dtype=torch.float32).eval()
    if error_mlp is not None:
        error_mlp = error_mlp.to(device=device, dtype=torch.float32).eval()

    null_op = (float(beta) == 0.0) or (not np.isfinite(tau)) or (error_mlp is None)
    do_corr = not null_op
    use_ood = ood_mu is not None

    lo_t = torch.as_tensor(np.asarray(phys_lo, np.float32), device=device)      # [10]
    hi_t = torch.as_tensor(np.asarray(phys_hi, np.float32), device=device)      # [10]
    dmax_t = torch.as_tensor(np.asarray(phys_dmax, np.float32), device=device)  # [10]

    if use_ood:
        mu_t = torch.as_tensor(np.asarray(ood_mu, np.float32), device=device)
        std_t = (torch.as_tensor(np.asarray(ood_std, np.float32), device=device)
                 if ood_std is not None else None)
        sinv_t = torch.as_tensor(np.asarray(ood_sigma_inv, np.float32), device=device)
        tau_ood_f = float(tau_ood)

    predictions_dict, true_dict = defaultdict(list), defaultdict(list)
    zero_stats = {"n_active_steps": 0, "n_pred_fire": 0, "n_ood": 0, "n_fire": 0,
                  "n_suppressed": 0, "n_proj_clipped": 0, "ood_abstain_rate": 0.0,
                  "gate_fire_rate": 0.0, "vanilla_fire_rate": 0.0, "suppression_rate": 0.0,
                  "proj_clip_rate": 0.0, "use_ood_gate": bool(use_ood)}

    order, init_win, cont, ctrl = _collect_pass(
        dataset, collect_batch, num_workers, restrict_scenarios)
    if len(order) == 0:
        return predictions_dict, true_dict, zero_stats

    S, lengths, maxL, W, CY, UY = _pack(order, init_win, cont, ctrl, num_continuous, num_controls)
    window = torch.from_numpy(W).to(device)
    UY_t = torch.from_numpy(UY).to(device)
    CY_t = torch.from_numpy(CY).to(device) if (do_corr and gate_on == "true") else None
    lengths_t = torch.from_numpy(lengths.astype(np.int64)).to(device)
    preds = np.zeros((S, maxL, num_continuous), np.float32)
    tau_t = None if null_op else torch.as_tensor(float(tau), device=device, dtype=torch.float32)

    n_active_steps = n_pred_fire = n_ood = n_fire = n_suppressed = n_proj_clipped = 0

    for t in tqdm.tqdm(range(maxL), desc="AR-roll(phys-gated)"):
        raw = model(window)                                    # [S, 10]
        active = t < lengths_t                                 # [S] bool
        if do_corr:
            feats = build_error_features_acc(
                raw, window[:, -1, :num_continuous], UY_t[:, t, :], t,
                step_norm_const, step_norm_scale=step_norm_scale)
            e_hat = error_mlp(feats)                            # [S, 10]
            if gate_on == "pred":
                score = torch.linalg.vector_norm(e_hat, ord=2, dim=1)          # [S]
            else:  # 'true' = ORACLE gate on realized error
                score = torch.linalg.vector_norm(CY_t[:, t, :] - raw, ord=2, dim=1)
            pred_fire = score > tau_t                          # [S] predicted-error gate
            if use_ood:
                u = feats - mu_t
                if std_t is not None:
                    u = u / std_t
                dm = torch.sqrt(((u @ sinv_t) * u).sum(dim=1).clamp_min(0))    # [S] D_M(z_t)
                ood = dm > tau_ood_f                           # [S] out-of-distribution
                fire = pred_fire & (~ood)                      # both gates
            else:
                ood = None
                fire = pred_fire
            corr_full = raw + beta * e_hat                     # [S, 10]
            # physics projection: bounds then rate limit (relative to the last fed value)
            prev = window[:, -1, :num_continuous]              # [S, 10] last fed continuous
            corr_proj = torch.minimum(torch.maximum(corr_full, lo_t), hi_t)      # clip [lo,hi]
            corr_proj = torch.minimum(torch.maximum(corr_proj, prev - dmax_t),
                                      prev + dmax_t)                            # rate limit
            chosen = torch.where(fire.unsqueeze(1), corr_proj, raw)
            # accounting over ACTIVE steps
            n_active_steps += int(active.sum().item())
            n_pred_fire += int((pred_fire & active).sum().item())
            if use_ood:
                n_ood += int((ood & active).sum().item())
                n_suppressed += int((pred_fire & ood & active).sum().item())
            n_fire += int((fire & active).sum().item())
            proj_changed = (corr_proj != corr_full).any(dim=1)  # [S] projection altered value
            n_proj_clipped += int((fire & active & proj_changed).sum().item())
        else:
            chosen = raw                                       # exact null-op
            n_active_steps += int(active.sum().item())
        preds[:, t, :] = chosen.detach().cpu().numpy()
        next_row = torch.cat([chosen, UY_t[:, t, :]], dim=1)   # [S, input]
        window = torch.cat([window[:, 1:, :], next_row[:, None, :]], dim=1)

    for i, s in enumerate(order):
        L = lengths[i]
        for t in range(L):
            predictions_dict[s].append(preds[i, t])
            true_dict[s].append(CY[i, t])

    stats = {
        "n_active_steps": n_active_steps, "n_pred_fire": n_pred_fire, "n_ood": n_ood,
        "n_fire": n_fire, "n_suppressed": n_suppressed, "n_proj_clipped": n_proj_clipped,
        "ood_abstain_rate": (float(n_ood) / n_active_steps if n_active_steps > 0 else 0.0),
        "gate_fire_rate": (float(n_fire) / n_active_steps if n_active_steps > 0 else 0.0),
        "vanilla_fire_rate": (float(n_pred_fire) / n_active_steps if n_active_steps > 0 else 0.0),
        "suppression_rate": (float(n_suppressed) / n_pred_fire if n_pred_fire > 0 else 0.0),
        "proj_clip_rate": (float(n_proj_clipped) / n_fire if n_fire > 0 else 0.0),
        "use_ood_gate": bool(use_ood),
    }
    return predictions_dict, true_dict, stats
