"""Batched, faithful re-implementations of the predict.py evaluation passes.

These produce the SAME predictions_dict / true_dict structure as
  - predict.regressive_predictions_absolute      (teacher-forcing, Table 7)
  - predict.autoregressive_predictions_absolute   (rollout,        Table 8)
driven by the same TransformerDataset, but batched so the 60-run sweep is
tractable (the originals loop one sample at a time -> hours per 5min run).

Metrics use compute_micro_macro(), which is mathematically identical to
scenario_wise_metrics():
    per-step error = mean over the 10 continuous variables
        MAE_step  = mean(|pred - true|)
        RMSE_step = sqrt(mean((pred - true)**2))
    per-scenario   = mean over that scenario's steps
    micro = mean over all (scenario, step) pairs   (pooled, = all_df.MAE.mean())
    macro = mean of per-scenario means             (= Table 7/8 "averaged over scenarios")

predict.py exposes these via `from predict_batched import ...`.
"""
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import DataLoader
import tqdm


def compute_micro_macro(predictions_dict, true_dict):
    """Identical definition to scenario_wise_metrics' micro/macro MAE & RMSE."""
    per_mae, per_rmse, all_mae, all_rmse = [], [], [], []
    for sc in predictions_dict:
        if not predictions_dict[sc]:
            continue
        P = np.vstack([np.ravel(p) for p in predictions_dict[sc]]).astype(np.float64)
        Y = np.vstack([np.ravel(y) for y in true_dict[sc]]).astype(np.float64)
        diff = P - Y
        mae_steps = np.mean(np.abs(diff), axis=1)
        rmse_steps = np.sqrt(np.mean(diff ** 2, axis=1))
        all_mae.append(mae_steps)
        all_rmse.append(rmse_steps)
        per_mae.append(float(mae_steps.mean()))
        per_rmse.append(float(rmse_steps.mean()))
    if not per_mae:
        nan = float("nan")
        return dict(micro_mae=nan, micro_rmse=nan, macro_mae=nan, macro_rmse=nan, n_scen=0)
    am = np.concatenate(all_mae)
    ar = np.concatenate(all_rmse)
    return dict(
        micro_mae=float(am.mean()), micro_rmse=float(ar.mean()),
        macro_mae=float(np.mean(per_mae)), macro_rmse=float(np.mean(per_rmse)),
        n_scen=len(per_mae),
    )


@torch.inference_mode()
def regressive_predictions_absolute_batched(model, dataset, device="cuda",
                                            batch_size=4096, num_workers=4):
    """Teacher-forcing: every step uses the true past window (no feedback).
    Batched equivalent of predict.regressive_predictions_absolute.
    Returns (predictions_dict, true_dict) keyed by scenario id."""
    model = model.to(device=device, dtype=torch.float32).eval()
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers)
    predictions_dict, true_dict = defaultdict(list), defaultdict(list)
    for batch in tqdm.tqdm(dl, desc="TF"):
        pv = batch["past_values"].to(device, torch.float32)
        out = model(pv).detach().cpu().numpy()           # [B, num_continuous]
        cy = batch["continuous_y"].numpy()               # [B, num_continuous]
        scn = batch["y_metadata"][:, 0].numpy().astype(np.int64)
        for b in range(out.shape[0]):
            s = int(scn[b])
            predictions_dict[s].append(out[b])
            true_dict[s].append(cy[b])
    return predictions_dict, true_dict


@torch.inference_mode()
def autoregressive_predictions_absolute_batched(model, dataset, device="cuda",
                                                num_continuous=10, collect_batch=4096,
                                                num_workers=4):
    """Rollout: feed predicted continuous + true binary back in, roll the window,
    reset per scenario. Scenario-batched equivalent of
    predict.autoregressive_predictions_absolute.
    Returns (predictions_dict, true_dict) keyed by scenario id."""
    model = model.to(device=device, dtype=torch.float32).eval()

    # Pass 1: collect each scenario's ordered targets + its initial window.
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
            if s not in init_win:
                init_win[s] = pv[b]
                order.append(s)
            cont[s].append(cy[b])
            binr[s].append(by[b])

    predictions_dict, true_dict = defaultdict(list), defaultdict(list)
    S = len(order)
    if S == 0:
        return predictions_dict, true_dict

    nc = num_continuous
    F = init_win[order[0]].shape[1]
    lengths = np.array([len(cont[s]) for s in order])
    maxL = int(lengths.max())
    W = np.stack([init_win[s] for s in order]).astype(np.float32)   # [S, seq, F]
    CY = np.zeros((S, maxL, nc), np.float32)
    BY = np.zeros((S, maxL, nc), np.float32)
    for i, s in enumerate(order):
        L = lengths[i]
        CY[i, :L] = np.vstack(cont[s])
        BY[i, :L] = np.vstack(binr[s])

    window = torch.from_numpy(W).to(device)
    BY_t = torch.from_numpy(BY).to(device)
    preds = np.zeros((S, maxL, nc), np.float32)

    # Pass 2: lockstep rollout. Done scenarios keep rolling on garbage but are
    # never read (we only collect preds[:, t] for t < length below).
    for t in tqdm.tqdm(range(maxL), desc="AR-roll"):
        out = model(window)                              # [S, nc]
        preds[:, t, :] = out.detach().cpu().numpy()
        next_row = torch.cat([out, BY_t[:, t, :]], dim=1)  # [S, F]
        window = torch.cat([window[:, 1:, :], next_row[:, None, :]], dim=1)

    for i, s in enumerate(order):
        L = lengths[i]
        for t in range(L):
            predictions_dict[s].append(preds[i, t])
            true_dict[s].append(CY[i, t])
    return predictions_dict, true_dict
