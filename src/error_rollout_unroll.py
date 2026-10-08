"""Unrolled multi-step (truncated-BPTT) training for the per-accident-type ErrorMLP (V2).

ADDITIVE and self-contained. It does NOT modify error_rollout_acc.py, the one-shot
training/eval path, or any backbone. It REUSES the byte-faithful collect/pack
primitives (`_collect_pass`, `_pack`), the feature builder (`build_error_features_acc`),
and the per-cell input width (`in_dim_for`) from error_rollout_acc, so the features,
the fed-back-correction convention, and the step-norm are IDENTICAL to the existing
open-loop / gated rollouts (the eval stays comparable).

Idea (fixing the covariate shift of the one-shot ErrorMLP)
----------------------------------------------------------
The one-shot ErrorMLP is trained on the OPEN-LOOP (beta=0) trajectory with a
one-step target, but at inference it runs on the CORRECTED trajectory it itself
produces. Here we instead UNROLL the corrected AR rollout for H steps and
backpropagate the MULTI-STEP rollout error into the (weight-tied) ErrorMLP:

    window <- init
    for h in 0..H-1:
        raw   = backbone(window)                 # FROZEN, forward-only
        e_hat = error_mlp(features(raw, window, controls_t, t))
        corr  = raw + beta * e_hat               # UNGATED during training (see below)
        loss += SmoothL1(corr[active], GT_t[active])
        window <- roll(window, cat([corr, controls_t]))   # CORRECTED value fed back
    loss = loss / H ; loss.backward()            # grads flow through the fed-back corr

The backbone is frozen (`requires_grad_(False)`) but still differentiable as a
forward map, so gradients flow through the fed-back CORRECTED continuous values
back into the ErrorMLP at every earlier unrolled step (genuine BPTT). Only the
ErrorMLP weights accumulate gradient. We use TRUNCATED BPTT: the window is
detached at every H-step segment boundary, so memory is bounded by H.

Gate handling during training: TRAIN UNGATED, apply the hard top-q gate ONLY at
inference. Rationale: the hard top-q gate is non-differentiable; a soft sigmoid
surrogate would add a temperature hyperparameter AND a soft(train)/hard(test)
mismatch that muddies the comparison. Training ungated directly optimizes the
corrected trajectory the backbone follows when the correction is applied, while
INFERENCE stays byte-identical to the existing hard-gated corrected rollout
(`gated_corrected_rollout_acc`) -- so the shared tail eval is apples-to-apples and
the beta=0 null-op remains exact.

Controls are fed as GROUND TRUTH each step (same known-future convention as the
existing rollouts). Runs from src/.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import tqdm

from error_rollout_acc import (
    _collect_pass,
    _pack,
    build_error_features_acc,
)

__all__ = ["collect_scenario_segments", "train_unroll_epoch"]


def collect_scenario_segments(dataset, num_controls, device="cuda",
                              num_continuous=10, collect_batch=2048, num_workers=4,
                              restrict_scenarios=None):
    """Pass-1 collect + pack the per-scenario dense arrays needed for unrolled
    training. Returns a dict of torch tensors on `device`:
        order   [S] int64 scenario ids
        W       [S, seq, input] float32 init windows
        CY      [S, maxL, num_continuous] float32 continuous-Y (GT targets)
        UY      [S, maxL, num_controls]   float32 control-Y (known-future, GT)
        lengths [S] int64 valid rollout length per scenario
    (Identical collect/pack path as the open-loop/gated rollouts.)"""
    order, init_win, cont, ctrl = _collect_pass(
        dataset, collect_batch, num_workers, restrict_scenarios)
    if len(order) == 0:
        raise ValueError("collect_scenario_segments: no scenarios collected")
    S, lengths, maxL, W, CY, UY = _pack(
        order, init_win, cont, ctrl, num_continuous, num_controls)
    return {
        "order": torch.tensor([int(s) for s in order], dtype=torch.int64),
        "W": torch.from_numpy(W).to(device),
        "CY": torch.from_numpy(CY).to(device),
        "UY": torch.from_numpy(UY).to(device),
        "lengths": torch.from_numpy(lengths.astype(np.int64)).to(device),
        "S": int(S), "maxL": int(maxL),
    }


def train_unroll_epoch(model, error_mlp, opt, criterion, packed, beta,
                       unroll_len, step_norm_const, device="cuda",
                       num_continuous=10, scenario_batch=64, grad_clip=0.0,
                       max_segments=None, seed=0, desc="unroll"):
    """One epoch of truncated-BPTT unrolled training over scenario mini-batches.

    Backbone FROZEN (forward-only); only `error_mlp` params are optimized. The
    optimizer steps once per H-step segment (truncated BPTT). Returns
    (mean_step_loss, n_segments, n_steps) for the epoch.

    packed: dict from collect_scenario_segments (W/CY/UY/lengths on `device`).
    max_segments: cap segments PER mini-batch (SMOKE only; None = full length)."""
    assert all(not p.requires_grad for p in model.parameters()), "backbone must be frozen"
    model = model.to(device=device, dtype=torch.float32).eval()
    error_mlp = error_mlp.to(device=device, dtype=torch.float32).train()

    W, CY, UY = packed["W"], packed["CY"], packed["UY"]
    lengths, maxL, S = packed["lengths"], packed["maxL"], packed["S"]
    nc = num_continuous

    rng = np.random.default_rng(seed)
    perm = rng.permutation(S)
    total_loss, total_steps, total_segs = 0.0, 0, 0

    n_batches = (S + scenario_batch - 1) // scenario_batch
    for bstart in tqdm.tqdm(range(0, S, scenario_batch), total=n_batches, desc=desc):
        idx = perm[bstart:bstart + scenario_batch]
        idx_t = torch.as_tensor(idx, dtype=torch.int64, device=device)
        window = W.index_select(0, idx_t).clone()          # [b, seq, input]
        CY_b = CY.index_select(0, idx_t)                    # [b, maxL, nc]
        UY_b = UY.index_select(0, idx_t)                    # [b, maxL, num_controls]
        len_b = lengths.index_select(0, idx_t)             # [b]
        b_maxL = int(len_b.max().item())

        seg_i = 0
        for seg_start in range(0, b_maxL, unroll_len):
            if max_segments is not None and seg_i >= max_segments:
                break
            seg_i += 1
            window = window.detach()                       # truncate BPTT here
            opt.zero_grad()
            seg_loss = window.new_zeros(())
            seg_steps = 0
            for h in range(unroll_len):
                t = seg_start + h
                if t >= b_maxL:
                    break
                raw = model(window)                        # [b, nc], grads via window
                feats = build_error_features_acc(
                    raw, window[:, -1, :nc], UY_b[:, t, :], t, step_norm_const)
                e_hat = error_mlp(feats)                    # [b, nc]
                corr = raw + beta * e_hat                   # UNGATED (train)
                active = t < len_b                          # [b] bool
                if active.any():
                    seg_loss = seg_loss + criterion(corr[active], CY_b[active, t, :])
                    seg_steps += 1
                next_row = torch.cat([corr, UY_b[:, t, :]], dim=1)   # [b, input]
                window = torch.cat([window[:, 1:, :], next_row[:, None, :]], dim=1)
            if seg_steps == 0:
                continue
            seg_loss = seg_loss / seg_steps                 # mean over H unrolled steps
            seg_loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(error_mlp.parameters(), grad_clip)
            opt.step()
            total_loss += float(seg_loss.detach().item()) * seg_steps
            total_steps += seg_steps
            total_segs += 1

    mean_loss = total_loss / max(1, total_steps)
    return mean_loss, total_segs, total_steps
