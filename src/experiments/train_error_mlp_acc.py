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
from accident_dataset import AccidentWindowDataset, CONTINUOUS_COLS
from models.error_mlp import ErrorMLP
from error_rollout_acc import (
    load_frozen_backbone_acc,
    generate_rollout_error_dataset_acc,
    baseline_rollout_with_stats_acc,
    gated_corrected_rollout_acc,
    in_dim_for,
)
from tail_analysis_acc import step_err_from_dicts, tail_metrics


def dim_weight_vector(dim_weights_cfg, num_continuous=10):
    """{dim_key: weight} -> a length-num_continuous float32 tensor (default 1.0 for
    unlisted dims). dim_key may be an int index OR a CONTINUOUS_COLS name (e.g.
    "TGRCS(10)"), so config can use either the plan's names or raw indices (as the
    unit test does)."""
    w = np.ones(num_continuous, dtype=np.float32)
    for key, val in (dim_weights_cfg or {}).items():
        if isinstance(key, str) and key in CONTINUOUS_COLS:
            idx = CONTINUOUS_COLS.index(key)
        else:
            idx = int(key)
        w[idx] = float(val)
    return torch.from_numpy(w)


def tail_sample_weights(Y, tail_q, tail_lambda):
    """Per-sample tail weight wsamp[n] = 1 + tail_lambda * 1[||Y_n||_2 > tau], where
    tau = quantile(||Y_n||_2, tail_q) over the given (training) Y. tau is computed
    ONCE from this Y (the caller passes the actual training targets). Returns
    (weights[N] float32, tau: float)."""
    norms = np.linalg.norm(Y, axis=1)
    tau = float(np.quantile(norms, tail_q))
    w = 1.0 + float(tail_lambda) * (norms > tau).astype(np.float32)
    return w.astype(np.float32), tau


def make_loss(cfg_tr):
    """Returns a callable criterion(pred, target, weight=None). `weight` is an
    optional per-sample weight vector; smoothl1/l1/mse ignore it (so the default
    smoothl1 path is byte-identical to before this option existed).

    'tail_weighted': SmoothL1(beta=huber_beta, reduction='none') per-element loss
    L[n,d], up-weighted per-dim by cfg `dim_weights` (wdim) and per-sample by
    `weight` (wsamp, precomputed by the caller from tail_sample_weights):
        loss = mean_n( wsamp[n] * mean_d( wdim[d] * L[n,d] ) )
    With wdim all 1.0 and weight all 1.0 (i.e. tail_lambda=0.0), this reduces
    exactly to SmoothL1Loss(reduction='mean').
    """
    name = cfg_tr.get("loss", "smoothl1")
    if name == "smoothl1":
        base = nn.SmoothL1Loss(beta=float(cfg_tr.get("huber_beta", 0.01)))
        return lambda pred, target, weight=None: base(pred, target)
    if name == "l1":
        base = nn.L1Loss()
        return lambda pred, target, weight=None: base(pred, target)
    if name == "mse":
        base = nn.MSELoss()
        return lambda pred, target, weight=None: base(pred, target)
    if name == "tail_weighted":
        base = nn.SmoothL1Loss(beta=float(cfg_tr.get("huber_beta", 0.01)), reduction="none")
        wdim = dim_weight_vector(cfg_tr.get("dim_weights", {}))

        def _tail_weighted(pred, target, weight=None):
            L = base(pred, target)                          # [N, D] elementwise
            d = wdim.to(device=L.device, dtype=L.dtype)
            per_sample = (L * d.unsqueeze(0)).mean(dim=1)    # mean_d(wdim*L) -> [N]
            if weight is not None:
                w = weight.to(device=L.device, dtype=L.dtype)
                return (w * per_sample).mean()
            return per_sample.mean()
        return _tail_weighted
    raise ValueError(f"Unknown loss '{name}' (choose smoothl1|l1|mse|tail_weighted)")


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
    loss_name = cfg_tr.get("loss", "smoothl1")
    criterion = make_loss(cfg_tr)
    # Val loss ALWAYS plain/unweighted (SmoothL1 mean for tail_weighted, else the
    # identical criterion) so early-stopping / model selection stays comparable
    # across loss choices (tail_weighted's train loss is not directly comparable
    # to other losses' scale).
    if loss_name == "tail_weighted":
        val_criterion = nn.SmoothL1Loss(beta=float(cfg_tr.get("huber_beta", 0.01)))
    else:
        val_criterion = criterion
    opt = torch.optim.AdamW(mlp.parameters(), lr=float(cfg_tr["lr"]),
                            weight_decay=float(cfg_tr["weight_decay"]))
    bs = int(cfg_tr["batch_size"])
    gen = torch.Generator()
    gen.manual_seed(int(cfg_tr.get("dataloader_seed", 42)))

    sample_w = None
    if loss_name == "tail_weighted":
        tail_q = float(cfg_tr.get("tail_q", 0.99))
        tail_lambda = float(cfg_tr.get("tail_lambda", 4.0))
        sample_w, tau = tail_sample_weights(Y_tr, tail_q, tail_lambda)
        n_tail = int((sample_w > 1.0).sum())
        print(f"[train] tail_weighted: tau={tau:.6f} (q={tail_q}) lambda={tail_lambda} "
              f"tail_rows={n_tail}/{len(sample_w)}")
        tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr),
                              torch.from_numpy(sample_w))
    else:
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
        for batch in tr_dl:
            if sample_w is not None:
                xb, yb, wb = batch
                wb = wb.to(device)
            else:
                xb, yb = batch
                wb = None
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            loss = criterion(mlp(xb), yb, wb)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(mlp.parameters(), grad_clip)
            opt.step()
            tr_loss_sum += loss.item() * xb.shape[0]
            tr_n += xb.shape[0]
        tr_loss = tr_loss_sum / max(1, tr_n)
        mlp.eval()
        with torch.no_grad():
            val_loss = val_criterion(mlp(Xv), Yv).item()
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


# ---------------------------------------------------------------------------
# Phase-2 DAgger (V1): closed-loop iterative retraining on the GATED corrected
# trajectory the model actually runs at inference, with a held-out rollback guard.
#
# Design (documented so the eval is reproducible / comparable to the one-shot
# baseline and to V2 unrolled BPTT):
#   * Operating point (beta*, q*) is FIXED across rounds. Resolved from CLI, else
#     the cell's committed tail_analysis.json `operating_point_strict`, else a
#     sensible default (beta=0.5, q=0.1).
#   * The gate threshold tau is RECALIBRATED per-model on the held-out scenarios
#     each round so the target gate fraction q* is held fixed (deployment-realistic:
#     tau is a deployment knob calibrated on held-out and applied to train/test).
#   * Each round: (1) run the GATED corrected rollout at (beta*, tau) over TRAIN
#     collecting (features, CY-raw) on that CORRECTED trajectory; (2) fine-tune the
#     ErrorMLP on round-0 data UNION all corrected data so far, starting FROM
#     round-0 weights; (3) evaluate p99/mean on held-out and ROLL BACK (discard the
#     candidate, keep the previous best model for the next round's collection) if
#     held-out p99 regresses or mean regresses beyond the rollback tolerance
#     (--dagger-rollback-tol / cfg `dagger_rollback_tol`, default 0.0 = any
#     held-out mean regression rolls back).
#   * INFERENCE is identical to the existing hard-gated corrected rollout
#     (gated_corrected_rollout_acc), so beta=0 null-op stays exact and the shared
#     tail eval (tail_analysis_acc.py --model-file error_mlp_dagger.pt) is
#     apples-to-apples with the one-shot baseline.
#   * Committed one-shot artifacts are NEVER touched in DAgger mode: only
#     error_mlp_dagger.pt + dagger_rounds.json are written.
# ---------------------------------------------------------------------------

DAGGER_P99_TOL = 1e-6    # candidate held-out p99 must not regress (float-noise guard)


def resolve_rollback_tol(args, cfg_tr):
    """DAgger acceptance mean-regression tolerance. CLI > cfg `dagger_rollback_tol`
    > default 0.0 (any held-out mean regression rolls back)."""
    if args.dagger_rollback_tol is not None:
        return float(args.dagger_rollback_tol)
    return float(cfg_tr.get("dagger_rollback_tol", 0.0))


def resolve_operating_point(args, cfg, cell_name):
    """(beta*, q*) fixed across DAgger rounds. CLI > committed tail_analysis.json
    operating_point_strict > default (0.5, 0.1)."""
    beta, q = args.dagger_beta, args.dagger_q
    if beta is None or q is None:
        ta_path = os.path.join(cfg["out_root"], cell_name, "tail_analysis.json")
        op = None
        if os.path.exists(ta_path):
            with open(ta_path) as f:
                op = json.load(f).get("operating_point_strict")
        if op:
            beta = beta if beta is not None else float(op["beta"])
            q = q if q is not None else float(op["q"])
    beta = 0.5 if beta is None else float(beta)
    q = 0.1 if q is None else float(q)
    return beta, q


def calibrate_tau(model, mlp, q, ds, num_controls, step_norm_const, device,
                  num_workers, collect_batch, nc, restrict):
    """tau = quantile(||e_hat||_2, 1-q) of the CURRENT model on the (restricted)
    baseline trajectory. q<=0 -> +inf (never fire = null-op)."""
    if q <= 0.0:
        return float("inf")
    stats = baseline_rollout_with_stats_acc(
        model, mlp, ds, num_controls, step_norm_const, device=device,
        num_continuous=nc, collect_batch=collect_batch, num_workers=num_workers,
        restrict_scenarios=restrict)
    g = stats["g"]
    if g.size == 0:
        return float("inf")
    return float(np.quantile(g, 1.0 - q))


def eval_gated_metrics(model, mlp, beta, tau, ds, num_controls, step_norm_const,
                       device, num_workers, collect_batch, nc, restrict):
    """Gated corrected rollout at (beta, tau) -> per-step tail metrics + realized
    intervention rate. Uses the SAME dict path as the shared tail eval."""
    pd_, td_, _Xd, _Yd, _SIDd, interv = gated_corrected_rollout_acc(
        model, mlp, beta, tau, ds, num_controls, step_norm_const, device=device,
        gate_on="pred", num_continuous=nc, collect_batch=collect_batch,
        num_workers=num_workers, restrict_scenarios=restrict, collect_dagger=True)
    se, ps = step_err_from_dicts(pd_, td_)
    m = tail_metrics(se, ps)
    return {"mean": m["mean"], "p95": m["p95"], "p99": m["p99"],
            "p999": m["p999"], "max": m["max"],
            "worst10_scen_mean": m["worst10_scen_mean"],
            "n_steps": m["n_steps"], "n_scen": m["n_scen"],
            "intervention_rate": interv["intervention_rate"]}


def run_dagger(args, cfg, model, train_ds, round0_state, X_tr, Y_tr, X_val, Y_val,
               train_ids, heldout_ids, in_dim, num_controls, step_norm_const,
               seed, out_dir, dagger_rounds):
    """Finalized V1 DAgger loop. Returns (best_state, records, beta*, q*)."""
    cfg_tr, cfg_data = cfg["training"], cfg["data"]
    device = torch.device(cfg_tr["device"] if torch.cuda.is_available() else "cpu")
    nw = int(cfg_tr["num_workers"])
    cb = int(cfg_tr.get("collect_batch", 2048))
    nc = int(cfg_data["num_continuous"])
    seq_len, pred_len = int(cfg_data["seq_len"]), int(cfg_data["pred_len"])
    cache_dir = cfg_data.get("cache_dir")

    beta_star, q_star = resolve_operating_point(args, cfg, args.cell)
    rollback_tol = resolve_rollback_tol(args, cfg_tr)
    print(f"[{args.cell}][dagger] operating point beta*={beta_star} q*={q_star} "
          f"(tau recalibrated per-model on held-out; R={dagger_rounds}; "
          f"rollback_tol={rollback_tol})")

    test_ds = AccidentWindowDataset(
        cfg["cells"][args.cell]["test_csv"], seq_len=seq_len, pred_len=pred_len,
        cache_dir=cache_dir, max_scenarios=args.max_scenarios)

    train_ids = [int(s) for s in train_ids]
    heldout_ids = [int(s) for s in heldout_ids]

    def make_mlp(state):
        m = ErrorMLP(in_dim=in_dim, **cfg["error_mlp"])
        m.load_state_dict(state)
        return m.to(device).eval()

    current_state = {k: v.detach().cpu().clone() for k, v in round0_state.items()}
    current_mlp = make_mlp(current_state)

    # ---- round 0 = current one-shot baseline at (beta*, q*) ----
    tau0 = calibrate_tau(model, current_mlp, q_star, train_ds, num_controls,
                         step_norm_const, device, nw, cb, nc, heldout_ids)
    ho0 = eval_gated_metrics(model, current_mlp, beta_star, tau0, train_ds,
                             num_controls, step_norm_const, device, nw, cb, nc, heldout_ids)
    te0 = eval_gated_metrics(model, current_mlp, beta_star, tau0, test_ds,
                             num_controls, step_norm_const, device, nw, cb, nc, None)
    print(f"[{args.cell}][dagger r0] heldout mean={ho0['mean']:.6f} p99={ho0['p99']:.6f} "
          f"interv={ho0['intervention_rate']:.4f} | test mean={te0['mean']:.6f} "
          f"p99={te0['p99']:.6f}")
    records = [{"round": 0, "operating_point": {"beta": beta_star, "q": q_star,
               "tau_heldout": tau0}, "heldout": ho0, "test": te0,
               "n_train_rows_aggregated": int(X_tr.shape[0]),
               "accepted": True, "rolled_back": False}]
    best_ho, best_state = ho0, current_state

    def _save_round(state, r):
        if getattr(args, "save_round_models", False):
            torch.save({"state_dict": {k: v.detach().cpu().clone() for k, v in state.items()},
                        "cell": args.cell, "in_dim": in_dim, "num_controls": num_controls,
                        "step_norm_const": step_norm_const, "round": r,
                        "beta_star": beta_star, "q_star": q_star},
                       os.path.join(out_dir, f"error_mlp_dagger_r{r}.pt"))
    _save_round(best_state, 0)                # R0 = one-shot start

    X_agg, Y_agg = X_tr.copy(), Y_tr.copy()   # round-0 open-loop data (D0)

    for r in range(1, dagger_rounds + 1):
        # (1) tau for the CURRENT best model, calibrated on held-out (fixed q*).
        tau_c = calibrate_tau(model, current_mlp, q_star, train_ds, num_controls,
                              step_norm_const, device, nw, cb, nc, heldout_ids)
        # collect on TRAIN over the GATED corrected trajectory.
        _pd, _td, Xd, Yd, _sid, interv_tr = gated_corrected_rollout_acc(
            model, current_mlp, beta_star, tau_c, train_ds, num_controls,
            step_norm_const, device=device, gate_on="pred", num_continuous=nc,
            collect_batch=cb, num_workers=nw, restrict_scenarios=train_ids,
            collect_dagger=True)
        X_agg = np.concatenate([X_agg, Xd], axis=0)
        Y_agg = np.concatenate([Y_agg, Yd], axis=0)
        print(f"[{args.cell}][dagger r{r}] collected {Xd.shape[0]} corrected rows "
              f"(train interv={interv_tr['intervention_rate']:.4f}); "
              f"aggregated rows={X_agg.shape[0]}")

        # (2) fine-tune FROM round-0 weights on D0 UNION corrected-so-far.
        cand = ErrorMLP(in_dim=in_dim, **cfg["error_mlp"])
        cand_state, _hist = train_mlp(cand, X_agg, Y_agg, X_val, Y_val, cfg_tr,
                                      device, init_state=round0_state)
        cand_mlp = make_mlp(cand_state)

        # (3) evaluate candidate at its own held-out-calibrated tau; rollback guard.
        tau_cand = calibrate_tau(model, cand_mlp, q_star, train_ds, num_controls,
                                 step_norm_const, device, nw, cb, nc, heldout_ids)
        ho = eval_gated_metrics(model, cand_mlp, beta_star, tau_cand, train_ds,
                                num_controls, step_norm_const, device, nw, cb, nc, heldout_ids)
        te = eval_gated_metrics(model, cand_mlp, beta_star, tau_cand, test_ds,
                                num_controls, step_norm_const, device, nw, cb, nc, None)
        accept = (ho["p99"] <= best_ho["p99"] * (1.0 + DAGGER_P99_TOL) and
                  ho["mean"] <= best_ho["mean"] * (1.0 + rollback_tol))
        print(f"[{args.cell}][dagger r{r}] cand heldout mean={ho['mean']:.6f} "
              f"p99={ho['p99']:.6f} (best p99={best_ho['p99']:.6f}) -> "
              f"{'ACCEPT' if accept else 'ROLLBACK'} | test p99={te['p99']:.6f}")
        if accept:
            best_ho, best_state = ho, cand_state
            current_mlp, current_state = cand_mlp, cand_state
        records.append({"round": r, "operating_point": {"beta": beta_star,
                        "q": q_star, "tau_heldout": tau_cand},
                        "train_intervention_rate": interv_tr["intervention_rate"],
                        "heldout": ho, "test": te,
                        "n_train_rows_aggregated": int(X_agg.shape[0]),
                        "accepted": bool(accept), "rolled_back": bool(not accept)})
        _save_round(best_state, r)            # adopted (best-through-r) model

    return best_state, records, beta_star, q_star


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, help="one of the cells: map key (e.g. SBO)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-scenarios", type=int, default=None,
                    help="SMOKE ONLY: cap the number of TRAIN scenarios rolled out.")
    ap.add_argument("--dagger-rounds", type=int, default=None,
                    help="V1 DAgger rounds R (overrides config; default config value). "
                         "R>0 -> DAgger mode: writes error_mlp_dagger.pt + "
                         "dagger_rounds.json, NEVER touches error_mlp.pt.")
    ap.add_argument("--dagger-beta", type=float, default=None,
                    help="V1 fixed operating-point beta* (default: tail_analysis.json).")
    ap.add_argument("--dagger-q", type=float, default=None,
                    help="V1 fixed operating-point gate fraction q* (default: tail_analysis.json).")
    ap.add_argument("--dagger-rollback-tol", type=float, default=None,
                    help="DAgger held-out mean-regression rollback tolerance, e.g. 0.02 "
                         "= +2%% (overrides config `dagger_rollback_tol`; default 0.0 = "
                         "any regression rolls back).")
    ap.add_argument("--loss", type=str, default=None,
                    choices=["smoothl1", "l1", "mse", "tail_weighted"],
                    help="ErrorMLP training loss (overrides config `training.loss`; "
                         "default config value, normally smoothl1).")
    ap.add_argument("--tail-lambda", type=float, default=None,
                    help="tail_weighted: per-sample tail up-weight lambda "
                         "(overrides config `training.tail_lambda`).")
    ap.add_argument("--tail-q", type=float, default=None,
                    help="tail_weighted: tail quantile threshold on train ||Y||_2 "
                         "(overrides config `training.tail_q`).")
    ap.add_argument("--out-root", type=str, default=None,
                    help="Override out_root (SMOKE: keep the real out_root pristine).")
    ap.add_argument("--run-dir", type=str, default=None,
                    help="Override the cell's backbone run_dir (e.g. a NEW retrained "
                         "backbone); default = config cells.<cell>.run_dir.")
    ap.add_argument("--save-round-models", action="store_true",
                    help="DAgger: also save the adopted (best-through-r) ErrorMLP after "
                         "each round to error_mlp_dagger_r{r}.pt (for per-round analysis).")
    args = ap.parse_args()

    cfg = utils.load_config("error_mlp_accident")
    cells = cfg["cells"]
    assert args.cell in cells, f"--cell {args.cell} not in {list(cells)}"
    cell = cells[args.cell]
    cfg_tr = cfg["training"]
    cfg_data = cfg["data"]
    if args.loss is not None:
        cfg_tr["loss"] = args.loss
    if args.tail_lambda is not None:
        cfg_tr["tail_lambda"] = args.tail_lambda
    if args.tail_q is not None:
        cfg_tr["tail_q"] = args.tail_q
    seed = args.seed if args.seed is not None else int(cfg_tr["seed"])
    seed_everything(seed, workers=True)

    device = torch.device(cfg_tr["device"] if torch.cuda.is_available() else "cpu")
    run_dir = args.run_dir if args.run_dir is not None else cell["run_dir"]
    model, input_size = load_frozen_backbone_acc(run_dir, device)
    print(f"[{args.cell}] backbone run_dir={run_dir}")
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

    dagger_rounds = (args.dagger_rounds if args.dagger_rounds is not None
                     else int(cfg_tr.get("dagger_rounds", 0)))
    out_root = args.out_root if args.out_root is not None else cfg["out_root"]
    out_dir = os.path.join(out_root, args.cell)
    os.makedirs(out_dir, exist_ok=True)

    # heldout_scenarios.json (a committed one-shot artifact) is written ONLY in
    # one-shot mode; DAgger runs (R>0) reuse the committed split and never rewrite it.
    if dagger_rounds <= 0:
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

    if dagger_rounds <= 0:
        # ---- one-shot mode (existing, unchanged behavior): save error_mlp.pt ----
        torch.save({"state_dict": best_state if best_state is not None else mlp.state_dict(),
                    "cell": args.cell, "in_dim": in_dim, "num_controls": num_controls,
                    "step_norm_const": step_norm_const, "seed": seed,
                    "heldout_scenarios": [int(s) for s in heldout_ids]},
                   os.path.join(out_dir, "error_mlp.pt"))
        print(f"[{args.cell}][done] saved error_mlp.pt + heldout_scenarios.json to {out_dir}")
        return

    # ---- V1 DAgger mode (R>0): closed-loop iterative retraining ----
    round0_state = best_state if best_state is not None else mlp.state_dict()
    round0_state = {k: v.detach().cpu().clone() for k, v in round0_state.items()}
    best_dagger_state, records, beta_star, q_star = run_dagger(
        args, cfg, model, train_ds, round0_state, X_tr, Y_tr, X_val, Y_val,
        train_ids, heldout_ids, in_dim, num_controls, step_norm_const, seed,
        out_dir, dagger_rounds)

    torch.save({"state_dict": best_dagger_state, "cell": args.cell, "in_dim": in_dim,
                "num_controls": num_controls, "step_norm_const": step_norm_const,
                "seed": seed, "heldout_scenarios": [int(s) for s in heldout_ids],
                "dagger_rounds": dagger_rounds, "beta_star": beta_star, "q_star": q_star},
               os.path.join(out_dir, "error_mlp_dagger.pt"))
    final_round = max((rec["round"] for rec in records if rec.get("accepted")), default=0)
    rollback_tol = resolve_rollback_tol(args, cfg_tr)
    with open(os.path.join(out_dir, "dagger_rounds.json"), "w") as f:
        json.dump({"cell": args.cell, "dagger_rounds": dagger_rounds,
                   "beta_star": beta_star, "q_star": q_star,
                   "tau_calibration": "per-model quantile(||e_hat||_2, 1-q*) on held-out",
                   "rollback_rule": (f"accept iff heldout p99 non-regressed AND "
                                     f"mean <= best*(1+{rollback_tol})"),
                   "in_dim": in_dim, "num_controls": num_controls,
                   "step_norm_const": step_norm_const, "seed": seed,
                   "max_scenarios": args.max_scenarios,
                   "final_accepted_round": final_round, "rounds": records}, f, indent=2)
    print(f"[{args.cell}][done] saved error_mlp_dagger.pt + dagger_rounds.json to {out_dir} "
          f"(final accepted round={final_round})")


if __name__ == "__main__":
    main()
