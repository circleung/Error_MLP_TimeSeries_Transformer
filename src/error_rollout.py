"""Rollout-error dataset generation + faithful corrected AR rollout.

This module is ADDITIVE: it does not modify predict_batched.py. Both rollout
functions below mirror `predict_batched.autoregressive_predictions_absolute_batched`
Pass-1 (collect) and Pass-2 (lockstep) VERBATIM to guarantee identical scenario
ordering, initial windows, CY, BY, and float dtypes. The ONLY difference in
`autoregressive_corrected_batched` is the optional `beta * error_mlp(feats)`
correction term added to the backbone output before it is reported / fed back.
At beta == 0 that term vanishes, so the corrected rollout is byte-identical to
the baseline AR rollout (AC2 null-op gate).

Also provides `load_frozen_backbone`, which loads the frozen layer8 backbone
exactly like `sweep_eval.load_model` (backbone_kwargs read from the run_dir's
config_used.yaml -- single source of truth), then freezes every parameter.

Runs from src/ (relative imports: model_selector, predict_batched).
"""
import os
import re
import glob

import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import DataLoader
import tqdm
import yaml

from model_selector import ModelSelector


# feature layout constant, single source of truth
NUM_CONTINUOUS = 10
IN_DIM = 31
STEP_NORM_CONST = 300.0   # fixed constant normalizer (see plan R6). NOT batch maxL, NOT per-scenario length.


def build_error_features(backbone_pred, last_obs_cont, cur_binary, step_idx,
                         step_norm_scale=1.0):
    """All tensors [S,10]. step_norm = step_idx / STEP_NORM_CONST (same meaning for
    every scenario, every run). step_idx is the ABSOLUTE rollout step (0-based) at
    this row. `step_norm_scale` multiplies the step feature (set to 0.0 for the
    step-index ablation, which zeroes only the last feature dim). Returns [S, 31]."""
    step_norm = torch.full((backbone_pred.shape[0], 1),
                           (step_idx / STEP_NORM_CONST) * step_norm_scale,
                           device=backbone_pred.device, dtype=backbone_pred.dtype)
    return torch.cat([backbone_pred, last_obs_cont, cur_binary, step_norm], dim=1)


def find_best_ckpt(run_dir):
    """Same selection rule as sweep_eval.find_best_ckpt."""
    ckpt_dir = os.path.join(run_dir, "checkpoints")
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


def load_frozen_backbone(run_dir, device, ckpt=None):
    """Load the frozen backbone identically to sweep_eval.load_model.

    SINGLE SOURCE OF TRUTH: backbone_kwargs / lightning_kwargs are read from
    run_dir/config_used.yaml (NOT from error_mlp.yaml). Every backbone param is
    frozen (requires_grad_(False)). Returns the eval-mode backbone module."""
    os.environ.setdefault("NONINTERACTIVE", "1")   # belt-and-suspenders (gating fix already applied)
    with open(os.path.join(run_dir, "config_used.yaml")) as f:
        cfg = yaml.safe_load(f)
    backbone_kwargs = cfg["model"]["backbone_kwargs"]
    lightning_kwargs = cfg["model"].get("lightning_kwargs", {})
    _, lit = ModelSelector(
        "transformer_decoder", backbone_kwargs=backbone_kwargs,
        lightning_kwargs=lightning_kwargs,
    )
    ckpt_path = os.path.join(run_dir, ckpt) if ckpt else find_best_ckpt(run_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint in {run_dir}")
    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    lit.load_state_dict(state)                    # strict=True (default): fail loudly on mismatch
    model = lit.backbone.to(device=device, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _collect_pass(dataset, collect_batch, num_workers, restrict_scenarios=None):
    """Pass-1 collect, VERBATIM from autoregressive_predictions_absolute_batched,
    with an optional scenario restriction (used for held-out beta selection)."""
    if restrict_scenarios is not None:
        restrict_scenarios = set(int(s) for s in restrict_scenarios)
    dl = DataLoader(dataset, batch_size=collect_batch, shuffle=False,
                    num_workers=num_workers)
    order, init_win = [], {}
    cont, binr = defaultdict(list), defaultdict(list)
    for batch in tqdm.tqdm(dl, desc="AR-collect"):
        pv = batch["past_values"].numpy()                # [B, seq, F]
        cy = batch["continuous_y"].numpy()               # [B, nc]
        by = batch["binary_y"].numpy()                   # [B, nc]
        scn = batch["y_metadata"][:, 0].numpy().astype(np.int64)
        for b in range(pv.shape[0]):
            s = int(scn[b])
            if restrict_scenarios is not None and s not in restrict_scenarios:
                continue
            if s not in init_win:
                init_win[s] = pv[b]
                order.append(s)
            cont[s].append(cy[b])
            binr[s].append(by[b])
    return order, init_win, cont, binr


def _pack(order, init_win, cont, binr, num_continuous):
    """Pack collected per-scenario lists into dense [S, maxL, nc] arrays, VERBATIM
    from autoregressive_predictions_absolute_batched."""
    S = len(order)
    nc = num_continuous
    lengths = np.array([len(cont[s]) for s in order])
    maxL = int(lengths.max())
    W = np.stack([init_win[s] for s in order]).astype(np.float32)   # [S, seq, F]
    CY = np.zeros((S, maxL, nc), np.float32)
    BY = np.zeros((S, maxL, nc), np.float32)
    for i, s in enumerate(order):
        L = lengths[i]
        CY[i, :L] = np.vstack(cont[s])
        BY[i, :L] = np.vstack(binr[s])
    return S, nc, lengths, maxL, W, CY, BY


@torch.inference_mode()
def generate_rollout_error_dataset(model, dataset, device="cuda", num_continuous=10,
                                   collect_batch=4096, num_workers=4,
                                   restrict_scenarios=None):
    """OPEN-LOOP (beta=0) ROLLOUT round-0 collection (Option A1).

    Reuses the exact Pass-1 collect from autoregressive_predictions_absolute_batched
    to get order/init_win/CY/BY, then lockstep rolls with PREDICTED (uncorrected)
    continuous fed back and binary=truth. At each step t (absolute rollout index)
    per active scenario:
        feats_t = build_error_features(out, window[:, -1, :10], BY[:, t, :], t)
        err_t   = CY[:, t, :] - out             # 10-dim error of the RAW backbone

    Returns (X, Y, SID) as numpy: X=[N,31] float32, Y=[N,10] float32,
    SID=[N] int64 scenario id per row (N = sum of scenario lengths). Only rows with
    t < scenario_length are emitted (done scenarios masked), matching baseline
    collection. This is the open-loop distribution; it aligns with eval only at
    beta=0 (see plan Principle 3)."""
    model = model.to(device=device, dtype=torch.float32).eval()
    order, init_win, cont, binr = _collect_pass(
        dataset, collect_batch, num_workers, restrict_scenarios)
    if len(order) == 0:
        return (np.zeros((0, IN_DIM), np.float32),
                np.zeros((0, num_continuous), np.float32),
                np.zeros((0,), np.int64))

    S, nc, lengths, maxL, W, CY, BY = _pack(order, init_win, cont, binr, num_continuous)
    window = torch.from_numpy(W).to(device)
    BY_t = torch.from_numpy(BY).to(device)
    CY_t = torch.from_numpy(CY).to(device)
    lengths_t = torch.from_numpy(lengths.astype(np.int64)).to(device)
    order_t = torch.tensor([int(s) for s in order], dtype=torch.int64, device=device)

    X_chunks, Y_chunks, SID_chunks = [], [], []
    for t in tqdm.tqdm(range(maxL), desc="AR-roll(gen)"):
        out = model(window)                                  # [S, nc]
        active = t < lengths_t                               # [S] bool
        if active.any():
            feats = build_error_features(out, window[:, -1, :nc], BY_t[:, t, :], t)
            err = CY_t[:, t, :] - out                        # [S, nc]
            X_chunks.append(feats[active].detach().cpu().numpy().astype(np.float32))
            Y_chunks.append(err[active].detach().cpu().numpy().astype(np.float32))
            SID_chunks.append(order_t[active].detach().cpu().numpy().astype(np.int64))
        next_row = torch.cat([out, BY_t[:, t, :]], dim=1)    # [S, F]
        window = torch.cat([window[:, 1:, :], next_row[:, None, :]], dim=1)

    X = np.concatenate(X_chunks, axis=0) if X_chunks else np.zeros((0, IN_DIM), np.float32)
    Y = np.concatenate(Y_chunks, axis=0) if Y_chunks else np.zeros((0, num_continuous), np.float32)
    SID = np.concatenate(SID_chunks, axis=0) if SID_chunks else np.zeros((0,), np.int64)
    return X, Y, SID


@torch.inference_mode()
def autoregressive_corrected_batched(model, error_mlp, beta, dataset, device="cuda",
                                     num_continuous=10, collect_batch=4096, num_workers=4,
                                     restrict_scenarios=None, collect_dagger=False,
                                     step_norm_scale=1.0):
    """Corrected AR eval. Byte-identical to
    predict_batched.autoregressive_predictions_absolute_batched EXCEPT the
    fed-back / reported value is:
        raw   = model(window)                                   # [S,10]
        feats = build_error_features(raw, window[:,-1,:10], BY[:,t,:], t)
        corr  = raw + beta * error_mlp(feats)                   # [S,10]
        preds[:, t] = corr ; next_row = cat([corr, BY[:,t]])    # CORRECTED value FED BACK

    beta == 0 -> the correction term is skipped entirely -> corr is `raw` and the
    rollout is byte-identical to the baseline AR rollout. This null-op is a property
    of THIS wrapper and is INDEPENDENT of error_mlp weights / DAgger rounds.

    restrict_scenarios: optional set/list of scenario ids -> run rollout over only
      those scenarios (used for the held-out beta-selection set; a FULL corrected
      rollout, NOT a replay of stored open-loop CY/BY).
    step_norm_scale: multiplies the step feature (0.0 for the step-index ablation).
    collect_dagger=False (default): returns (predictions_dict, true_dict).
    collect_dagger=True: ALSO returns (X_d[N,31], Y_d[N,10], SID_d[N]) where X_d =
      feats on the CORRECTED trajectory and Y_d = CY[:,t] - raw (error of RAW
      backbone measured at corrected-trajectory states). Enables Phase-2 DAgger."""
    model = model.to(device=device, dtype=torch.float32).eval()
    if error_mlp is not None:
        error_mlp = error_mlp.to(device=device, dtype=torch.float32).eval()

    predictions_dict, true_dict = defaultdict(list), defaultdict(list)
    empty_ret = (predictions_dict, true_dict)

    order, init_win, cont, binr = _collect_pass(
        dataset, collect_batch, num_workers, restrict_scenarios)
    if len(order) == 0:
        if collect_dagger:
            return (predictions_dict, true_dict,
                    np.zeros((0, IN_DIM), np.float32),
                    np.zeros((0, num_continuous), np.float32),
                    np.zeros((0,), np.int64))
        return empty_ret

    S, nc, lengths, maxL, W, CY, BY = _pack(order, init_win, cont, binr, num_continuous)
    window = torch.from_numpy(W).to(device)
    BY_t = torch.from_numpy(BY).to(device)
    preds = np.zeros((S, maxL, nc), np.float32)

    do_corr = (beta != 0.0) and (error_mlp is not None)

    if collect_dagger:
        CY_t = torch.from_numpy(CY).to(device)
        lengths_t = torch.from_numpy(lengths.astype(np.int64)).to(device)
        order_t = torch.tensor([int(s) for s in order], dtype=torch.int64, device=device)
        Xd_chunks, Yd_chunks, SIDd_chunks = [], [], []

    # Pass 2: lockstep rollout. Done scenarios keep rolling on garbage but are
    # never read (we only collect preds[:, t] for t < length below).
    for t in tqdm.tqdm(range(maxL), desc="AR-roll(corr)"):
        raw = model(window)                                  # [S, nc]
        if do_corr or collect_dagger:
            feats = build_error_features(raw, window[:, -1, :nc], BY_t[:, t, :], t,
                                         step_norm_scale=step_norm_scale)
        if do_corr:
            corr = raw + beta * error_mlp(feats)             # [S, nc]
        else:
            corr = raw                                       # exact null-op at beta==0
        preds[:, t, :] = corr.detach().cpu().numpy()
        if collect_dagger:
            active = t < lengths_t
            if active.any():
                err = CY_t[:, t, :] - raw                    # error of RAW backbone at corrected state
                Xd_chunks.append(feats[active].detach().cpu().numpy().astype(np.float32))
                Yd_chunks.append(err[active].detach().cpu().numpy().astype(np.float32))
                SIDd_chunks.append(order_t[active].detach().cpu().numpy().astype(np.int64))
        next_row = torch.cat([corr, BY_t[:, t, :]], dim=1)   # [S, F]
        window = torch.cat([window[:, 1:, :], next_row[:, None, :]], dim=1)

    for i, s in enumerate(order):
        L = lengths[i]
        for t in range(L):
            predictions_dict[s].append(preds[i, t])
            true_dict[s].append(CY[i, t])

    if collect_dagger:
        Xd = np.concatenate(Xd_chunks, axis=0) if Xd_chunks else np.zeros((0, IN_DIM), np.float32)
        Yd = np.concatenate(Yd_chunks, axis=0) if Yd_chunks else np.zeros((0, num_continuous), np.float32)
        SIDd = np.concatenate(SIDd_chunks, axis=0) if SIDd_chunks else np.zeros((0,), np.int64)
        return predictions_dict, true_dict, Xd, Yd, SIDd
    return predictions_dict, true_dict
