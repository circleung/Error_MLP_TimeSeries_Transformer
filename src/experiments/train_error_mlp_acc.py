"""Train the per-accident-type ErrorMLP on a frozen per-cell backbone's OPEN-LOOP
(beta=0) rollout errors.

ADDITIVE mirror of train_error_mlp.py for the seq50 / variable-control cells.
Round-0: roll the frozen cell backbone over the cell's TRAIN scenarios (predicted
continuous fed back, controls=truth), emit (features, error) pairs, split scenarios
disjointly into train / held-out, fit the ErrorMLP. Persists error_mlp.pt +
heldout_scenarios.json to <out_root>/<cell>/. Phase-2 DAgger gated (default 0).

Usage (from src/):
    NONINTERACTIVE=1 python experiments/train_error_mlp_acc.py --cell SBO
"""
import os
import sys
import json
import argparse

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
from accident_dataset import AccidentWindowDataset
from models.error_mlp import ErrorMLP
from error_rollout_acc import (
    load_frozen_backbone_acc,
    generate_rollout_error_dataset_acc,
    autoregressive_corrected_batched_acc,
    in_dim_for,
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
    id lists + boolean row masks."""
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
    tr_dl = DataLoader(tr_ds, batch_size=bs,
                       shuffle=bool(cfg_tr.get("dataloader_shuffle", True)),
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
    ap.add_argument("--cell", required=True, help="one of the cells: map key (e.g. SBO)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-scenarios", type=int, default=None,
                    help="SMOKE ONLY: cap the number of TRAIN scenarios rolled out.")
    args = ap.parse_args()

    cfg = utils.load_config("error_mlp_accident")
    cells = cfg["cells"]
    assert args.cell in cells, f"--cell {args.cell} not in {list(cells)}"
    cell = cells[args.cell]
    cfg_tr = cfg["training"]
    cfg_data = cfg["data"]
    seed = args.seed if args.seed is not None else int(cfg_tr["seed"])
    seed_everything(seed, workers=True)

    device = torch.device(cfg_tr["device"] if torch.cuda.is_available() else "cpu")
    run_dir = cell["run_dir"]
    model, input_size = load_frozen_backbone_acc(run_dir, device)
    assert all(not p.requires_grad for p in model.parameters()), "backbone must be frozen"

    seq_len = int(cfg_data["seq_len"])
    pred_len = int(cfg_data["pred_len"])
    cache_dir = cfg_data.get("cache_dir")
    step_norm_const = float(cell["step_norm_const"])

    train_ds = AccidentWindowDataset(
        cell["train_csv"], seq_len=seq_len, pred_len=pred_len, cache_dir=cache_dir,
        max_scenarios=args.max_scenarios)
    num_controls = train_ds.num_controls
    in_dim = in_dim_for(num_controls)
    assert train_ds.input_size == input_size, (
        f"csv input_size {train_ds.input_size} != backbone input_size {input_size}")
    print(f"[{args.cell}] input_size={input_size} num_controls={num_controls} in_dim={in_dim} "
          f"step_norm_const={step_norm_const} train_windows={len(train_ds)}")

    print(f"[{args.cell}][gen] open-loop rollout error dataset (round-0)...")
    X, Y, SID = generate_rollout_error_dataset_acc(
        model, train_ds, num_controls, step_norm_const, device=device,
        num_continuous=int(cfg_data["num_continuous"]),
        collect_batch=int(cfg_tr.get("collect_batch", 2048)),
        num_workers=int(cfg_tr["num_workers"]))
    assert X.shape[0] > 0 and X.shape[1] == in_dim and Y.shape[1] == 10
    assert np.isfinite(X).all() and np.isfinite(Y).all()
    print(f"[{args.cell}][gen] X={X.shape} Y={Y.shape} scenarios={len(np.unique(SID))}")

    train_ids, heldout_ids, tr_mask, val_mask = scenario_disjoint_split(
        SID, float(cfg_tr["val_split_by_scenario"]), seed)
    X_tr, Y_tr = X[tr_mask], Y[tr_mask]
    X_val, Y_val = X[val_mask], Y[val_mask]
    print(f"[{args.cell}][split] train scen={len(train_ids)} rows={X_tr.shape[0]} | "
          f"heldout scen={len(heldout_ids)} rows={X_val.shape[0]}")

    out_dir = os.path.join(cfg["out_root"], args.cell)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "heldout_scenarios.json"), "w") as f:
        json.dump({"cell": args.cell,
                   "heldout_scenarios": [int(s) for s in heldout_ids],
                   "train_scenarios": [int(s) for s in train_ids],
                   "seed": seed, "num_controls": num_controls, "in_dim": in_dim,
                   "step_norm_const": step_norm_const,
                   "val_split_by_scenario": float(cfg_tr["val_split_by_scenario"])},
                  f, indent=2)

    mlp = ErrorMLP(in_dim=in_dim, **cfg["error_mlp"])
    print(f"[{args.cell}][train] round-0 ErrorMLP...")
    best_state, history = train_mlp(mlp, X_tr, Y_tr, X_val, Y_val, cfg_tr, device)
    for ep, tl, vl in history[-5:]:
        print(f"  epoch {ep}: train={tl:.6f} val={vl:.6f}")

    # ---- Phase-2 DAgger (gated; default 0 = skipped) ----
    dagger_rounds = int(cfg_tr.get("dagger_rounds", 0))
    betas = [float(b) for b in cfg["eval"]["betas"]]
    select_metric = cfg["eval"]["select_metric"]
    for r in range(dagger_rounds):
        best_beta, best_val = 0.0, float("inf")
        for beta in betas:
            pd_, td_ = autoregressive_corrected_batched_acc(
                model, mlp, beta, train_ds, num_controls, step_norm_const,
                device=device, num_workers=int(cfg_tr["num_workers"]),
                collect_batch=int(cfg_tr.get("collect_batch", 2048)),
                restrict_scenarios=[int(s) for s in heldout_ids])
            m = compute_micro_macro(pd_, td_)[select_metric]
            if m < best_val:
                best_val, best_beta = m, beta
        print(f"[{args.cell}][dagger r{r}] best beta on held-out = {best_beta} "
              f"({select_metric}={best_val:.6f})")
        _, _, Xd, Yd, SIDd = autoregressive_corrected_batched_acc(
            model, mlp, best_beta, train_ds, num_controls, step_norm_const,
            device=device, num_workers=int(cfg_tr["num_workers"]),
            collect_batch=int(cfg_tr.get("collect_batch", 2048)),
            restrict_scenarios=[int(s) for s in train_ids], collect_dagger=True)
        X_tr = np.concatenate([X_tr, Xd], axis=0)
        Y_tr = np.concatenate([Y_tr, Yd], axis=0)
        print(f"[{args.cell}][dagger r{r}] fine-tune on aggregated rows={X_tr.shape[0]}")
        best_state, history = train_mlp(
            mlp, X_tr, Y_tr, X_val, Y_val, cfg_tr, device, init_state=best_state)

    torch.save({"state_dict": best_state if best_state is not None else mlp.state_dict(),
                "cell": args.cell, "in_dim": in_dim, "num_controls": num_controls,
                "step_norm_const": step_norm_const, "seed": seed,
                "heldout_scenarios": [int(s) for s in heldout_ids]},
               os.path.join(out_dir, "error_mlp.pt"))
    print(f"[{args.cell}][done] saved error_mlp.pt + heldout_scenarios.json to {out_dir}")


if __name__ == "__main__":
    main()
