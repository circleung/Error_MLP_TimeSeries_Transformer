"""Differentiable, AR-aware rollout for warm-start backbone fine-tuning (Plan #4).

ADDITIVE. Does NOT modify error_rollout_acc.py / predict_batched.py / any committed
path. It provides a *trainable* mirror of the byte-faithful AR convention in
`error_rollout_acc.autoregressive_corrected_batched_acc` (continuous preds fed back,
controls UY injected as truth each step, window slide via torch.cat) but WITHOUT
`@torch.inference_mode` and WITHOUT `.detach()` on the modeled path (in mode='bptt'),
so gradients flow into the backbone.

Two arms (Plan O1 / O2):
  * mode='detached_ss' (DEFAULT): per-step scheduled sampling. At each step the
    fed-back continuous value is the model's own prediction w.p. `sampling_prob`
    (else the ground-truth continuous), and the fed-back value is `.detach()`'d, so
    NO BPTT graph spans steps (bounded memory even on LLOCA). The loss is the
    per-step SmoothL1/MSE(pred_t, CY_t) reduced over the horizon (dense supervision).
  * mode='bptt' (FALLBACK): fully free-running (sampling_prob forced to 1), the
    modeled path keeps its graph across a short horizon H (truncated-BPTT); loss is
    mean_t loss(pred_t, CY_t).

Controls (UY) are DATA tensors -> never require grad (fine as-is).

PARITY CONTRACT (correctness foundation, asserted by tests_arbb_smoke.py):
  The free-running path (sampling_prob=1, eval mode, torch.no_grad) reproduces
  `autoregressive_corrected_batched_acc(model, error_mlp=None, beta=0, ...)`
  trajectories byte-for-byte (both feed the RAW continuous prediction back and inject
  UY=truth; beta=0 there is an exact null-op corr=raw).

Scope-addition knobs (ALL DEFAULT-OFF -> reduce exactly to the core above; the
parity/null-op tests are unaffected):
  * feedback_noise_sigma (train-only, mode='detached_ss'): add eps~N(0,sigma^2) to
    the fed-back continuous state before it re-enters the window, widening the
    trajectory tube so the model learns to recover from drifted/low-density states.
    sigma=0.0 (default) -> no noise. Never applied under free-running parity (the
    branch is only entered in detached_ss with sigma>0).
  * loss_reduction in {'mean'(default),'cvar','tail_weighted'} for the trajectory
    loss (the lambda_tf teacher-forced anchor is always plain 'mean' -- crown-jewel
    guard -- and lives in the trainer, not here):
      - 'cvar': mean of the worst `cvar_alpha` fraction of per-(scenario,step) losses
        (CVaR_alpha). cvar_alpha=1.0 (default) == plain mean.
      - 'tail_weighted': per-(scenario,step) loss weighted by
        (scenario_base_err)^`tail_weight_pow` (normalized). tail_weight_pow=0.0
        (default) == plain mean.

Run from src/ (relative import: error_rollout_acc, model_selector).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

# Reuse the committed byte-faithful collect/pack so seeding matches the null-op path.
from error_rollout_acc import _collect_pass, _pack


NUM_CONTINUOUS = 10


def make_traj_loss_fn(name: str = "smoothl1", huber_beta: float = 0.01):
    """Elementwise (reduction='none') per-variable loss used by the trajectory term.

    reduction='none' so the caller can reduce per-(scenario,step) then over the batch
    with the configurable tail/CVaR reductions. Choices mirror train_error_mlp_acc."""
    name = str(name).lower()
    if name == "smoothl1":
        return nn.SmoothL1Loss(beta=float(huber_beta), reduction="none")
    if name == "l1":
        return nn.L1Loss(reduction="none")
    if name == "mse":
        return nn.MSELoss(reduction="none")
    raise ValueError(f"Unknown traj loss '{name}' (choose smoothl1|l1|mse)")


def _reduce_traj_loss(per_losses, sids, reduction, cvar_alpha, tail_weight_pow,
                      scenario_base_err):
    """Reduce a 1-D tensor of per-(scenario,step) scalar losses to a scalar.

    `per_losses` [N] (graph-carrying), `sids` [N] long (scenario ROW index into
    `scenario_base_err`). Each mode reduces EXACTLY to the plain mean at its neutral
    parameter (cvar_alpha=1.0 / tail_weight_pow=0.0) -> the additions are zero-cost
    null-ops by default (asserted by tests_arbb_smoke.py)."""
    n = per_losses.numel()
    if n == 0:
        return per_losses.sum()  # 0.0 with grad-safe dtype/device
    if reduction == "mean":
        return per_losses.mean()
    if reduction == "cvar":
        if float(cvar_alpha) >= 1.0:
            return per_losses.mean()                      # exact null-op
        k = max(1, int(math.ceil(float(cvar_alpha) * n)))
        top = torch.topk(per_losses, k, largest=True).values
        return top.mean()
    if reduction == "tail_weighted":
        if float(tail_weight_pow) == 0.0 or scenario_base_err is None:
            return per_losses.mean()                      # exact null-op
        w = scenario_base_err[sids].clamp_min(0.0) ** float(tail_weight_pow)
        denom = w.sum().clamp_min(1e-12)
        return (w * per_losses).sum() / denom
    raise ValueError(f"Unknown loss_reduction '{reduction}' "
                     f"(choose mean|cvar|tail_weighted)")


def differentiable_rollout(model, init_win, CY, UY, horizon,
                           sampling_prob=1.0, mode="detached_ss",
                           lengths=None, loss_fn=None,
                           loss_reduction="mean", cvar_alpha=1.0,
                           tail_weight_pow=0.0, scenario_base_err=None,
                           feedback_noise_sigma=0.0, num_continuous=NUM_CONTINUOUS,
                           generator=None, return_preds=True):
    """AR-aware differentiable rollout over a packed scenario batch.

    Args (packed, all on the SAME device as `model`):
      model      : the backbone (SimpleDecoderOnlyTransformer). Caller sets .eval()/
                   .train() (dropout) and float32. NOT moved/re-cast here.
      init_win   : [S, seq, input]   seed window per scenario (first GT window).
      CY         : [S, maxL, nc]      ground-truth continuous targets per step.
      UY         : [S, maxL, nctrl]   ground-truth controls (known-future) per step.
      horizon    : int rollout length (<= CY.shape[1]).
      sampling_prob : per-step P(feed model's own prediction) [detached_ss]. >=1.0 =>
                   always pred (free-running); <=0.0 => always GT (full teacher force).
                   Forced to 1.0 in mode='bptt'.
      mode       : 'detached_ss' (default) | 'bptt'.
      lengths    : [S] long active length per scenario (steps t>=length excluded from
                   the loss). None => all `horizon` steps active.
      loss_fn    : elementwise (reduction='none') loss; None => no loss computed
                   (pure prediction pass, e.g. parity).
      loss_reduction/cvar_alpha/tail_weight_pow/scenario_base_err : see module docstring.
      feedback_noise_sigma : train-only Gaussian feedback-noise std (detached_ss only).
      generator  : optional torch.Generator for reproducible sampling/noise.
      return_preds : also return stacked per-step predictions [S, horizon, nc] (detached).

    Returns dict {"loss": scalar tensor or None, "preds": [S,horizon,nc] or None,
                  "n_active": int}.
    """
    assert mode in ("detached_ss", "bptt"), f"mode must be detached_ss|bptt, got {mode}"
    device = init_win.device
    S = init_win.shape[0]
    horizon = int(horizon)
    assert horizon >= 1 and horizon <= CY.shape[1], (
        f"horizon {horizon} out of range [1, {CY.shape[1]}]")

    if mode == "bptt":
        sampling_prob = 1.0                    # bptt is free-running by definition

    if lengths is None:
        lengths = torch.full((S,), horizon, dtype=torch.long, device=device)
    else:
        lengths = lengths.to(device=device, dtype=torch.long)

    window = init_win                          # do NOT mutate caller's tensor in place
    preds_out = [] if return_preds else None
    loss_terms, loss_sids = [], []
    n_active = 0

    for t in range(horizon):
        raw = model(window)                                    # [S, nc]
        if return_preds:
            preds_out.append(raw.detach())
        tgt = CY[:, t, :]                                      # [S, nc]
        active = t < lengths                                  # [S] bool

        if loss_fn is not None and bool(active.any()):
            elem = loss_fn(raw, tgt)                           # [S, nc] elementwise
            per = elem.mean(dim=1)                             # [S] per-(scen,step)
            idx = torch.nonzero(active, as_tuple=False).squeeze(1)
            loss_terms.append(per[active])
            loss_sids.append(idx)
            n_active += int(idx.numel())

        # ---- decide the fed-back continuous value ----
        if sampling_prob >= 1.0:
            fed = raw                                         # free-running
        elif sampling_prob <= 0.0:
            fed = tgt                                         # full teacher forcing
        else:
            use_pred = torch.rand((S, 1), device=device, generator=generator) < sampling_prob
            fed = torch.where(use_pred, raw, tgt)

        if mode == "detached_ss":
            fed = fed.detach()                                # NO BPTT graph across steps
            if feedback_noise_sigma > 0.0:                    # train-only, default off
                fed = fed + feedback_noise_sigma * torch.randn(
                    fed.shape, device=device, generator=generator, dtype=fed.dtype)
        # mode='bptt': keep graph, no detach, no feedback noise (per spec).

        next_row = torch.cat([fed, UY[:, t, :]], dim=1)       # [S, input]
        window = torch.cat([window[:, 1:, :], next_row[:, None, :]], dim=1)

    loss = None
    if loss_fn is not None:
        if loss_terms:
            all_l = torch.cat(loss_terms)
            all_s = torch.cat(loss_sids)
        else:
            all_l = torch.zeros((0,), device=device)
            all_s = torch.zeros((0,), dtype=torch.long, device=device)
        loss = _reduce_traj_loss(all_l, all_s, loss_reduction, cvar_alpha,
                                 tail_weight_pow, scenario_base_err)

    preds = torch.stack(preds_out, dim=1) if return_preds else None
    return {"loss": loss, "preds": preds, "n_active": n_active}


def pack_scenarios(dataset, num_controls, num_continuous=NUM_CONTINUOUS,
                   collect_batch=2048, num_workers=4, restrict_scenarios=None):
    """Pass-1 collect + pack a dataset (or Subset) into dense per-scenario tensors,
    using the COMMITTED byte-faithful `_collect_pass`/`_pack` so seeding (first GT
    window per scenario) matches the null-op AR path exactly.

    Returns None if empty, else dict of CPU tensors + metadata:
      {"order":[sid...], "S", "maxL", "lengths":[S], "W":[S,seq,input],
       "CY":[S,maxL,nc], "UY":[S,maxL,nctrl]}."""
    order, init_win, cont, ctrl = _collect_pass(
        dataset, collect_batch, num_workers, restrict_scenarios)
    if len(order) == 0:
        return None
    S, lengths, maxL, W, CY, UY = _pack(
        order, init_win, cont, ctrl, num_continuous, num_controls)
    return {
        "order": [int(s) for s in order],
        "S": int(S),
        "maxL": int(maxL),
        "lengths": torch.from_numpy(lengths.astype(np.int64)),
        "W": torch.from_numpy(W),
        "CY": torch.from_numpy(CY),
        "UY": torch.from_numpy(UY),
    }
