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
                                step_norm_scale=1.0):
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
    metrics consume it unchanged."""
    assert gate_on in ("pred", "true"), f"gate_on must be 'pred' or 'true', got {gate_on}"
    model = model.to(device=device, dtype=torch.float32).eval()
    if error_mlp is not None:
        error_mlp = error_mlp.to(device=device, dtype=torch.float32).eval()

    # Null-op assertion: beta==0 or tau==inf must be an exact baseline.
    null_op = (float(beta) == 0.0) or (not np.isfinite(tau)) or (error_mlp is None)
    do_corr = not null_op

    predictions_dict, true_dict = defaultdict(list), defaultdict(list)

    order, init_win, cont, ctrl = _collect_pass(
        dataset, collect_batch, num_workers, restrict_scenarios)
    if len(order) == 0:
        return predictions_dict, true_dict

    S, lengths, maxL, W, CY, UY = _pack(order, init_win, cont, ctrl, num_continuous, num_controls)
    window = torch.from_numpy(W).to(device)
    UY_t = torch.from_numpy(UY).to(device)
    CY_t = torch.from_numpy(CY).to(device)
    preds = np.zeros((S, maxL, num_continuous), np.float32)
    tau_t = None if null_op else torch.as_tensor(float(tau), device=device, dtype=torch.float32)

    for t in tqdm.tqdm(range(maxL), desc="AR-roll(gated)"):
        raw = model(window)                                   # [S, 10]
        if do_corr:
            feats = build_error_features_acc(
                raw, window[:, -1, :num_continuous], UY_t[:, t, :], t,
                step_norm_const, step_norm_scale=step_norm_scale)
            e_hat = error_mlp(feats)                           # [S, 10]
            if gate_on == "pred":
                score = torch.linalg.vector_norm(e_hat, ord=2, dim=1)          # [S]
            else:  # 'true' = ORACLE gate on realized error
                score = torch.linalg.vector_norm(CY_t[:, t, :] - raw, ord=2, dim=1)
            fire = (score > tau_t).unsqueeze(1)               # [S, 1] bool
            corr_full = raw + beta * e_hat                    # [S, 10]
            chosen = torch.where(fire, corr_full, raw)        # gated select
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
    return predictions_dict, true_dict
