"""Train the ErrorMLP on the frozen backbone's OPEN-LOOP (beta=0) rollout errors.

Round-0: roll the frozen backbone over the TRAIN scenarios (predicted continuous
fed back, binary=truth), emit (features, error) pairs, split scenarios disjointly
into train / held-out, and fit the ErrorMLP with the configured loss. Persists the
held-out scenario id list (the ONLY beta-selection set, consumed by eval).

Phase-2 DAgger (only if training.dagger_rounds > 0; default 0 = skipped): after
round-0, pick the current best beta on the held-out set, collect on-corrected-
trajectory data on the TRAIN scenarios, aggregate with round-0 data, and
FINE-TUNE the existing round-0 weights (not retrain from scratch).

Usage (from src/):
    NONINTERACTIVE=1 python experiments/train_error_mlp.py --interval 60min --k 3 --seed 42
"""
import os
import sys
import json
import argparse

# make src/ importable
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

os.environ.setdefault("NONINTERACTIVE", "1")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import seed_everything

import utils
from models.error_mlp import ErrorMLP
from error_rollout import (
    load_frozen_backbone,
    generate_rollout_error_dataset,
    autoregressive_corrected_batched,
)
from predict_batched import compute_micro_macro


def make_loss(cfg_tr):
    name = cfg_tr.get("loss", "smoothl1")
    if name == "smoothl1":
        return nn.SmoothL1Loss(beta=float(cfg_tr.get("huber_beta", 0.01)))
    if name == "l1":
        return nn.L1Loss()
    if name == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unknown loss '{name}' (choose smoothl1|l1|mse)")


def scenario_disjoint_split(sid, val_frac, seed):
    """Split UNIQUE scenario ids into (train_ids, heldout_ids), seeded. Returns
    boolean row masks + the id lists."""
    uniq = np.unique(sid)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    heldout_ids = np.sort(perm[:n_val])
    train_ids = np.sort(perm[n_val:])
    heldout_set = set(int(s) for s in heldout_ids)
    is_heldout = np.array([int(s) in heldout_set for s in sid])
    return train_ids, heldout_ids, ~is_heldout, is_heldout


def train_mlp(mlp, X_tr, Y_tr, X_val, Y_val, cfg_tr, device, init_state=None):
    """Train (or fine-tune from init_state) the ErrorMLP; early-stop on held-out
    loss. Returns the best state_dict and the (train_loss, val_loss) tail."""
    if init_state is not None:
        mlp.load_state_dict(init_state)
    mlp = mlp.to(device)
    criterion = make_loss(cfg_tr)
    opt = torch.optim.AdamW(mlp.parameters(), lr=float(cfg_tr["lr"]),
                            weight_decay=float(cfg_tr["weight_decay"]))
    bs = int(cfg_tr["batch_size"])
    gen = torch.Generator()
    gen.manual_seed(int(cfg_tr.get("dataloader_seed", 42)))
    tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr))
    tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=bool(cfg_tr.get("dataloader_shuffle", True)),
                       generator=gen, num_workers=0)
    Xv = torch.from_numpy(X_val).to(device)
    Yv = torch.from_numpy(Y_val).to(device)
    grad_clip = float(cfg_tr.get("grad_clip_norm", 0.0))

    best_val, best_state, history = float("inf"), None, []
    patience, bad = 5, 0
    for epoch in range(int(cfg_tr["epochs"])):
        mlp.train()
        tr_loss_sum, tr_n = 0.0, 0
        for xb, yb in tr_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            loss = criterion(mlp(xb), yb)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(mlp.parameters(), grad_clip)
            opt.step()
            tr_loss_sum += loss.item() * xb.shape[0]
            tr_n += xb.shape[0]
        tr_loss = tr_loss_sum / max(1, tr_n)
        mlp.eval()
        with torch.no_grad():
            val_loss = criterion(mlp(Xv), Yv).item()
        history.append((epoch, tr_loss, val_loss))
        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in mlp.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[train] early stop at epoch {epoch} (best val {best_val:.6f})")
                break
    if best_state is not None:
        mlp.load_state_dict(best_state)
    return best_state, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-scenarios", type=int, default=None,
                    help="SMOKE ONLY: cap the number of TRAIN scenarios rolled out.")
    args = ap.parse_args()

    cfg = utils.load_config("error_mlp")
    cfg_tr = cfg["training"]
    seed = args.seed if args.seed is not None else int(cfg_tr["seed"])
    seed_everything(seed, workers=True)

    device = torch.device(cfg_tr["device"] if torch.cuda.is_available() else "cpu")
    run_dir = cfg["backbone"]["run_dir"]
    ckpt = cfg["backbone"].get("ckpt")
    model = load_frozen_backbone(run_dir, device, ckpt=ckpt)
    assert all(not p.requires_grad for p in model.parameters()), "backbone must be frozen"

    train_csv = cfg["data"]["train_csv"]
    seq_len = int(cfg["data"]["seq_len"])
    pred_len = int(cfg["data"]["pred_len"])
    ptype = cfg["data"]["prediction_type"]
    train_ds = utils.get_dataset(train_csv, seq_len, pred_len, ptype)

    restrict = None
    if args.max_scenarios is not None:
        # SMOKE: take the first N distinct scenario ids in dataset order.
        seen, ordered = set(), []
        for i in range(len(train_ds)):
            s = int(train_ds[i]["y_metadata"][0].item())
            if s not in seen:
                seen.add(s)
                ordered.append(s)
            if len(ordered) >= args.max_scenarios:
                break
        restrict = ordered
        print(f"[smoke] restricting to {len(restrict)} train scenarios: {restrict}")

    print("[gen] open-loop rollout error dataset (round-0)...")
    X, Y, SID = generate_rollout_error_dataset(
        model, train_ds, device=device, num_continuous=10,
        num_workers=int(cfg_tr["num_workers"]), restrict_scenarios=restrict)
    assert X.shape[0] > 0 and X.shape[1] == 31 and Y.shape[1] == 10
    assert np.isfinite(X).all() and np.isfinite(Y).all()
    print(f"[gen] X={X.shape} Y={Y.shape} scenarios={len(np.unique(SID))}")

    train_ids, heldout_ids, tr_mask, val_mask = scenario_disjoint_split(
        SID, float(cfg_tr["val_split_by_scenario"]), seed)
    X_tr, Y_tr = X[tr_mask], Y[tr_mask]
    X_val, Y_val = X[val_mask], Y[val_mask]
    print(f"[split] train scen={len(train_ids)} rows={X_tr.shape[0]} | "
          f"heldout scen={len(heldout_ids)} rows={X_val.shape[0]}")

    out_dir = cfg_tr["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "heldout_scenarios.json"), "w") as f:
        json.dump({"heldout_scenarios": [int(s) for s in heldout_ids],
                   "train_scenarios": [int(s) for s in train_ids],
                   "seed": seed, "val_split_by_scenario": float(cfg_tr["val_split_by_scenario"])},
                  f, indent=2)

    mlp = ErrorMLP(**cfg["error_mlp"])
    print("[train] round-0 ErrorMLP...")
    best_state, history = train_mlp(mlp, X_tr, Y_tr, X_val, Y_val, cfg_tr, device)
    for ep, tl, vl in history[-5:]:
        print(f"  epoch {ep}: train={tl:.6f} val={vl:.6f}")

    # ---- Phase-2 DAgger (gated; default 0 = skipped) ----
    dagger_rounds = int(cfg_tr.get("dagger_rounds", 0))
    betas = [float(b) for b in cfg["eval"]["betas"]]
    select_metric = cfg["eval"]["select_metric"]
    for r in range(dagger_rounds):
        # pick current best beta on the held-out set (full corrected AR)
        best_beta, best_val = 0.0, float("inf")
        for beta in betas:
            pd_, td_ = autoregressive_corrected_batched(
                model, mlp, beta, train_ds, device=device,
                num_workers=int(cfg_tr["num_workers"]),
                restrict_scenarios=[int(s) for s in heldout_ids])
            m = compute_micro_macro(pd_, td_)[select_metric]
            if m < best_val:
                best_val, best_beta = m, beta
        print(f"[dagger r{r}] best beta on held-out = {best_beta} ({select_metric}={best_val:.6f})")
        _, _, Xd, Yd, SIDd = autoregressive_corrected_batched(
            model, mlp, best_beta, train_ds, device=device,
            num_workers=int(cfg_tr["num_workers"]),
            restrict_scenarios=[int(s) for s in train_ids], collect_dagger=True)
        X_tr = np.concatenate([X_tr, Xd], axis=0)
        Y_tr = np.concatenate([Y_tr, Yd], axis=0)
        print(f"[dagger r{r}] fine-tune on aggregated rows={X_tr.shape[0]}")
        best_state, history = train_mlp(
            mlp, X_tr, Y_tr, X_val, Y_val, cfg_tr, device, init_state=best_state)

    torch.save({"state_dict": best_state if best_state is not None else mlp.state_dict(),
                "config": cfg, "seed": seed,
                "heldout_scenarios": [int(s) for s in heldout_ids]},
               os.path.join(out_dir, "error_mlp.pt"))
    print(f"[done] saved error_mlp.pt + heldout_scenarios.json to {out_dir}")


if __name__ == "__main__":
    main()
