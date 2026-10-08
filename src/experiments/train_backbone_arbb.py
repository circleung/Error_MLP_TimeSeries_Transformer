"""Warm-start AR-aware fine-tune trainer for a per-cell backbone (Plan #4, Phase 0/1).

Loads the committed frozen backbone (SAME path as load_frozen_backbone_acc:
ModelSelector from <run_dir>/config_used.yaml + strict-load ckpt state_dict), then
UNFREEZES it and fine-tunes with a composite loss:
    L = lambda_tf * single-step-TF-MSE  +  lambda_ar * traj-loss(differentiable_rollout)
The traj-loss uses detached scheduled sampling (Plan O1 default) or short-H truncated
BPTT (O2 fallback), mirroring the byte-faithful AR convention.

CROWN-JEWEL SAFETY:
  * OWN optimizer -- AdamW(lr from config, default 1e-5) + cosine-to-zero. NEVER calls
    LitRNNBaseModel.configure_optimizers (which hardcodes lr=1e-3 + StepLR).
  * NONINTERACTIVE=1 set in-process before instantiating LitRNNBaseModel.
  * fine-tune forward in model.train() (dropout ON); ALL metrics/gates in model.eval().
  * Ckpt saved as {"state_dict": {"backbone.*": ...}} into a NEW out_root run_dir with
    config_used.yaml copied in, so load_frozen_backbone_acc strict-loads it. Original
    weight dirs are NEVER written.

Data: train_csv carved by a pre-registered 3-role scenario-disjoint split
(RandomState(42)) into fine-tune / gate / report. Fine-tune ONLY on the fine-tune role;
per-epoch readouts on the gate role.

Usage (from src/, NONINTERACTIVE=1):
  python experiments/train_backbone_arbb.py --cell SBO --smoke --max-scenarios 40 --epochs 2
"""
import os
import sys
import csv
import json
import time
import shutil
import argparse

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

os.environ.setdefault("NONINTERACTIVE", "1")   # R8: gate LitRNNBaseModel.__init__ input()

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset
from pytorch_lightning import seed_everything

import utils
from accident_dataset import AccidentWindowDataset
from model_selector import ModelSelector
from error_rollout_acc import (
    find_best_ckpt_acc, autoregressive_corrected_batched_acc, in_dim_for, CKPT_SUBDIR,
)
from error_rollout_arbb import differentiable_rollout, make_traj_loss_fn, pack_scenarios
from predict_batched import compute_micro_macro
from tail_analysis_acc import step_err_from_dicts, tail_metrics


def load_backbone_for_finetune(run_dir, device):
    """Same load contract as load_frozen_backbone_acc, but returns the model UNFROZEN
    (trainable) plus the lit wrapper (for backbone.* state_dict saving) and cfg."""
    os.environ.setdefault("NONINTERACTIVE", "1")
    with open(os.path.join(run_dir, "config_used.yaml")) as f:
        cfg = yaml.safe_load(f)
    model_name = cfg["model"].get("name", "transformer_decoder")
    backbone_kwargs = cfg["model"]["backbone_kwargs"]
    lightning_kwargs = cfg["model"].get("lightning_kwargs", {})
    _, lit = ModelSelector(model_name, backbone_kwargs=backbone_kwargs,
                           lightning_kwargs=lightning_kwargs)
    ckpt_path = find_best_ckpt_acc(run_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint under {os.path.join(run_dir, CKPT_SUBDIR)}")
    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    lit.load_state_dict(state)                                 # strict=True
    model = lit.backbone.to(device=device, dtype=torch.float32)
    for p in model.parameters():
        p.requires_grad_(True)                                # UNFREEZE for fine-tune
    return model, lit, int(backbone_kwargs["input_size"]), cfg


def three_role_split(dataset, finetune_frac, gate_frac, report_frac, seed=42):
    """Pre-registered 3-role scenario-disjoint split (mirrors accident_dataset's
    RandomState(seed) shuffle-then-partition; leak-safe by unique scenario)."""
    window_scen = dataset.window_scenarios()
    unique = np.unique(window_scen)
    rng = np.random.RandomState(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_ft = max(1, int(round(finetune_frac * n)))
    n_gate = max(1, int(round(gate_frac * n)))
    if n >= 3:                                                # keep all 3 roles non-empty
        n_ft = min(n_ft, n - 2)
        n_gate = min(n_gate, n - n_ft - 1)
    ft = set(int(s) for s in shuffled[:n_ft])
    gate = set(int(s) for s in shuffled[n_ft:n_ft + n_gate])
    report = set(int(s) for s in shuffled[n_ft + n_gate:])
    assert ft.isdisjoint(gate) and ft.isdisjoint(report) and gate.isdisjoint(report), \
        "scenario leakage across roles"
    return ft, gate, report


def role_subset(dataset, scen_set):
    ws = dataset.window_scenarios()
    idx = [i for i, s in enumerate(ws) if int(s) in scen_set]
    return Subset(dataset, idx)


@torch.inference_mode()
def single_step_metrics(model, subset, device, batch_size, num_workers):
    """Single-step teacher-forced MSE + MAE over a window subset (.eval())."""
    model.eval()
    dl = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    se_sum, ae_sum, n = 0.0, 0.0, 0
    for batch in dl:
        pv = batch["past_values"].to(device, torch.float32)
        cy = batch["continuous_y"].to(device, torch.float32)
        diff = model(pv) - cy
        se_sum += float((diff * diff).sum().item())
        ae_sum += float(diff.abs().sum().item())
        n += int(diff.numel())
    if n == 0:
        return float("nan"), float("nan")
    return se_sum / n, ae_sum / n


def ar_beta0(model, subset, num_controls, step_norm_const, device, nc,
             collect_batch, num_workers):
    """AR beta=0 (null-op) micro-MAE + p99 on a subset (.eval() via inference_mode)."""
    pd_, td_ = autoregressive_corrected_batched_acc(
        model, None, 0.0, subset, num_controls, step_norm_const, device=device,
        num_continuous=nc, collect_batch=collect_batch, num_workers=num_workers)
    mm = compute_micro_macro(pd_, td_)
    se, ps = step_err_from_dicts(pd_, td_)
    tm = tail_metrics(se, ps)
    return mm["micro_mae"], tm["p99"]


@torch.inference_mode()
def scenario_base_err_from_pack(model, W, CY, UY, lengths, horizon, device, nc):
    """Per-scenario open-loop (free-running) mean-abs error over the horizon, for the
    tail_weighted reduction. Computed ONCE from the (warm-start=B0) backbone."""
    model.eval()
    out = differentiable_rollout(model, W, CY, UY, horizon, sampling_prob=1.0,
                                 mode="detached_ss", lengths=lengths, loss_fn=None,
                                 num_continuous=nc, return_preds=True)
    preds = out["preds"]                                       # [S, horizon, nc]
    tgt = CY[:, :horizon, :]
    err = (preds - tgt).abs().mean(dim=2)                      # [S, horizon]
    t = torch.arange(horizon, device=device)[None, :]
    mask = (t < lengths[:, None]).float()
    per = (err * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)   # [S]
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--config", default="backbone_arbb")
    ap.add_argument("--cells-config", default="error_mlp_accident",
                    help="Config providing cells/data (run_dir/csv/step_norm_const).")
    ap.add_argument("--run_dir", default=None, help="Override cell run_dir.")
    ap.add_argument("--smoke", action="store_true", help="Plumbing smoke (small run).")
    ap.add_argument("--max-scenarios", type=int, default=None, help="Cap TRAIN scenarios.")
    ap.add_argument("--epochs", type=int, default=None, help="Override epoch cap.")
    ap.add_argument("--arm", choices=["detached_ss", "bptt"], default=None)
    ap.add_argument("--unroll-len", type=int, default=None, help="Rollout horizon H.")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--ss-p", type=float, default=None,
                    help="Pin the scheduled-sampling prob to a CONSTANT (p_start=p_end); "
                         "overrides the curriculum. Useful for fixed-difficulty smoke "
                         "and Phase-1 operating points.")
    ap.add_argument("--scenario-batch", type=int, default=None)
    ap.add_argument("--out-root", type=str, default=None, help="Override out_root (NEW tree).")
    ap.add_argument("--fixed-tail-json", type=str, default=None,
                    help="Optional frozen B0-ranked TEST-tail json -> per-epoch CR_tail.")
    # --- OOD-max sweep overrides (CLI > config; default None -> config value) ---
    ap.add_argument("--feedback-noise-sigma", type=float, default=None,
                    help="Override feedback_noise_sigma (train-only augmentation).")
    ap.add_argument("--feedback-noise-sigma-end", type=float, default=None,
                    help="Override feedback_noise_sigma_end (curriculum endpoint).")
    ap.add_argument("--loss-reduction", choices=["mean", "cvar", "tail_weighted"],
                    default=None, help="Override trajectory-loss reduction.")
    ap.add_argument("--cvar-alpha", type=float, default=None,
                    help="Override cvar_alpha (worst-fraction; 1.0==mean).")
    ap.add_argument("--tail-weight-pow", type=float, default=None,
                    help="Override tail_weight_pow (per-scenario B0-error weighting; 0==uniform).")
    args = ap.parse_args()

    acfg = utils.load_config(args.config)
    ccfg = utils.load_config(args.cells_config)
    cells = ccfg["cells"]
    assert args.cell in cells, f"--cell {args.cell} not in {list(cells)}"
    cell = cells[args.cell]
    cfg_tr, cfg_data = ccfg["training"], ccfg["data"]

    seed = int(acfg.get("split", {}).get("seed", 42))
    seed_everything(seed, workers=True)
    device = torch.device(cfg_tr["device"] if torch.cuda.is_available() else "cpu")

    # ---- resolve hyperparameters (CLI > config) ----
    run_dir = args.run_dir or cell["run_dir"]
    epochs = args.epochs if args.epochs is not None else int(acfg["epochs"])
    arm = args.arm or acfg["arm"]
    unroll_len = args.unroll_len if args.unroll_len is not None else int(acfg["unroll_len"])
    lr = args.lr if args.lr is not None else float(acfg["optim"]["lr"])
    scenario_batch = (args.scenario_batch if args.scenario_batch is not None
                      else int(acfg["scenario_batch"]))
    wd = float(acfg["optim"]["weight_decay"])
    grad_clip = float(acfg["optim"].get("grad_clip_norm", 0.0))
    lambda_tf = float(acfg["loss"]["lambda_tf"])
    lambda_ar = float(acfg["loss"]["lambda_ar"])
    traj_loss_fn = make_traj_loss_fn(acfg["loss"].get("traj_loss", "smoothl1"),
                                     acfg["loss"].get("huber_beta", 0.01))
    p_start = float(acfg["scheduled_sampling"]["p_start"])
    p_end = float(acfg["scheduled_sampling"]["p_end"])
    if args.ss_p is not None:                                   # pin constant difficulty
        p_start = p_end = float(args.ss_p)
    sigma0 = (args.feedback_noise_sigma if args.feedback_noise_sigma is not None
              else float(acfg.get("feedback_noise_sigma", 0.0)))
    sigma1 = (args.feedback_noise_sigma_end if args.feedback_noise_sigma_end is not None
              else acfg.get("feedback_noise_sigma_end", None))
    sigma1 = sigma0 if sigma1 is None else float(sigma1)
    loss_reduction = args.loss_reduction or str(acfg.get("loss_reduction", "mean"))
    cvar_alpha = (args.cvar_alpha if args.cvar_alpha is not None
                  else float(acfg.get("cvar_alpha", 1.0)))
    tail_weight_pow = (args.tail_weight_pow if args.tail_weight_pow is not None
                       else float(acfg.get("tail_weight_pow", 0.0)))

    nc = int(cfg_data["num_continuous"])
    seq_len, pred_len = int(cfg_data["seq_len"]), int(cfg_data["pred_len"])
    cache_dir = cfg_data.get("cache_dir")
    num_workers = int(cfg_tr["num_workers"])
    collect_batch = int(cfg_tr.get("collect_batch", 2048))
    tf_batch_size = int(cfg_tr.get("batch_size", 4096))
    step_norm_const = float(cell["step_norm_const"])

    # ---- warm-start load + UNFREEZE ----
    model, lit, input_size, used_cfg = load_backbone_for_finetune(run_dir, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.cell}] warm-start from {run_dir} | input_size={input_size} "
          f"params={n_params} | arm={arm} H={unroll_len} lr={lr:g} "
          f"scenario_batch={scenario_batch} epochs={epochs}")

    # ---- data + 3-role split ----
    train_ds = AccidentWindowDataset(cell["train_csv"], seq_len=seq_len, pred_len=pred_len,
                                     cache_dir=cache_dir, max_scenarios=args.max_scenarios)
    num_controls = train_ds.num_controls
    in_dim = in_dim_for(num_controls)
    assert train_ds.input_size == input_size, (
        f"csv input_size {train_ds.input_size} != backbone input_size {input_size}")
    sp = acfg["split"]
    ft_set, gate_set, report_set = three_role_split(
        train_ds, float(sp["finetune_frac"]), float(sp["gate_frac"]),
        float(sp["report_frac"]), seed=seed)
    ft_subset = role_subset(train_ds, ft_set)
    gate_subset = role_subset(train_ds, gate_set)
    print(f"[{args.cell}] split scen: fine-tune={len(ft_set)} gate={len(gate_set)} "
          f"report={len(report_set)} | ft_windows={len(ft_subset)} gate_windows={len(gate_subset)}")

    # ---- pre-pack fine-tune scenarios once (seed=first GT window per scenario) ----
    ft_pack = pack_scenarios(ft_subset, num_controls, num_continuous=nc,
                             collect_batch=collect_batch, num_workers=num_workers)
    assert ft_pack is not None and ft_pack["S"] > 0, "empty fine-tune pack"
    horizon = min(ft_pack["maxL"], unroll_len)
    W = ft_pack["W"].to(device)
    CY = ft_pack["CY"][:, :horizon].contiguous().to(device)
    UY = ft_pack["UY"][:, :horizon].contiguous().to(device)
    lengths = ft_pack["lengths"].to(device)
    Sf = ft_pack["S"]
    print(f"[{args.cell}] fine-tune pack: S={Sf} maxL={ft_pack['maxL']} horizon(H)={horizon}")

    base_err = None
    if tail_weight_pow > 0.0:
        base_err = scenario_base_err_from_pack(model, W, CY, UY, lengths, horizon, device, nc)

    # ---- optional fixed test-tail (per-epoch CR_tail) ----
    tail_subset, test_ds = None, None
    if args.fixed_tail_json is not None and os.path.exists(args.fixed_tail_json):
        with open(args.fixed_tail_json) as f:
            obj = json.load(f)
        tail_ids = set(int(s) for s in (obj["scenarios"] if isinstance(obj, dict) else obj))
        test_ds = AccidentWindowDataset(cell["test_csv"], seq_len=seq_len, pred_len=pred_len,
                                        cache_dir=cache_dir, max_scenarios=args.max_scenarios)
        tail_subset = role_subset(test_ds, tail_ids)
        print(f"[{args.cell}] fixed test-tail: {len(tail_ids)} scenarios "
              f"({len(tail_subset)} windows) for CR_tail")

    # ---- OWN optimizer (NEVER configure_optimizers) ----
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
             if acfg["optim"].get("cosine_to_zero", True) else None)
    mse = nn.MSELoss()

    def cycle(dl):
        while True:
            for b in dl:
                yield b
    tf_dl = DataLoader(ft_subset, batch_size=tf_batch_size, shuffle=True,
                       num_workers=num_workers, drop_last=False)
    tf_iter = cycle(tf_dl)

    # ---- output dirs ----
    out_root = args.out_root or acfg["out_root"]
    new_run_dir = os.path.join(out_root, f"{args.cell}_arbb_seq50_pred1")
    ckpt_dir = os.path.join(new_run_dir, CKPT_SUBDIR)
    os.makedirs(ckpt_dir, exist_ok=True)
    shutil.copy(os.path.join(run_dir, "config_used.yaml"),
                os.path.join(new_run_dir, "config_used.yaml"))
    results_dir = os.path.join(SRC_DIR, acfg.get("results_root", "results/arbb"), args.cell)
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "train_log.csv")
    log_cols = ["epoch", "train_loss", "tf_loss", "traj_loss", "lr", "ss_p", "noise_sigma",
                "gate_ss_mse", "gate_ss_mae", "gate_ar_micro_mae", "gate_ar_p99",
                "cr_indist", "cr_tail", "peak_mem_gb", "epoch_time_s"]
    logf = open(log_path, "w", newline="")
    writer = csv.DictWriter(logf, fieldnames=log_cols)
    writer.writeheader()

    best_val, best_state, best_epoch = float("inf"), None, -1
    train_loss_hist = []

    for epoch in range(epochs):
        p = p_start if epochs <= 1 else p_start + (p_end - p_start) * epoch / (epochs - 1)
        sigma = sigma0 if epochs <= 1 else sigma0 + (sigma1 - sigma0) * epoch / (epochs - 1)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.time()

        model.train()
        perm = torch.randperm(Sf)
        ep_loss, ep_tf, ep_traj, nb = 0.0, 0.0, 0.0, 0
        for s0 in range(0, Sf, scenario_batch):
            bidx = perm[s0:s0 + scenario_batch]
            w = W[bidx]
            cy = CY[bidx]
            uy = UY[bidx]
            ln = lengths[bidx]
            h = min(horizon, int(ln.max().item()))
            be = base_err[bidx] if base_err is not None else None
            out = differentiable_rollout(
                model, w, cy, uy, h, sampling_prob=p, mode=arm, lengths=ln,
                loss_fn=traj_loss_fn, loss_reduction=loss_reduction, cvar_alpha=cvar_alpha,
                tail_weight_pow=tail_weight_pow, scenario_base_err=be,
                feedback_noise_sigma=sigma, num_continuous=nc, return_preds=False)
            traj_loss = out["loss"]

            tf_batch = next(tf_iter)
            pv = tf_batch["past_values"].to(device, torch.float32)
            tfy = tf_batch["continuous_y"].to(device, torch.float32)
            tf_loss = mse(model(pv), tfy)

            loss = lambda_tf * tf_loss + lambda_ar * traj_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            ep_loss += float(loss.item())
            ep_tf += float(tf_loss.item())
            ep_traj += float(traj_loss.item())
            nb += 1
        if sched is not None:
            sched.step()
        ep_loss /= max(1, nb); ep_tf /= max(1, nb); ep_traj /= max(1, nb)
        train_loss_hist.append(ep_loss)

        # ---- per-epoch readouts (.eval()) ----
        gate_mse, gate_mae = single_step_metrics(model, gate_subset, device,
                                                 tf_batch_size, num_workers)
        gate_ar_mae, gate_ar_p99 = ar_beta0(model, gate_subset, num_controls,
                                            step_norm_const, device, nc, collect_batch,
                                            num_workers)
        cr_indist = (gate_ar_mae / gate_mae) if gate_mae and gate_mae > 0 else float("nan")
        cr_tail = float("nan")
        if tail_subset is not None:
            t_mse, t_mae = single_step_metrics(model, tail_subset, device, tf_batch_size,
                                               num_workers)
            t_ar_mae, _ = ar_beta0(model, tail_subset, num_controls, step_norm_const,
                                   device, nc, collect_batch, num_workers)
            cr_tail = (t_ar_mae / t_mae) if t_mae and t_mae > 0 else float("nan")
        peak_gb = (torch.cuda.max_memory_allocated(device) / 1e9
                   if device.type == "cuda" else 0.0)
        dt = time.time() - t0
        cur_lr = opt.param_groups[0]["lr"]

        writer.writerow({"epoch": epoch, "train_loss": ep_loss, "tf_loss": ep_tf,
                         "traj_loss": ep_traj, "lr": cur_lr, "ss_p": p, "noise_sigma": sigma,
                         "gate_ss_mse": gate_mse, "gate_ss_mae": gate_mae,
                         "gate_ar_micro_mae": gate_ar_mae, "gate_ar_p99": gate_ar_p99,
                         "cr_indist": cr_indist, "cr_tail": cr_tail,
                         "peak_mem_gb": peak_gb, "epoch_time_s": dt})
        logf.flush()
        print(f"[{args.cell}] ep{epoch}: loss={ep_loss:.6e} (tf={ep_tf:.6e} traj={ep_traj:.6e}) "
              f"p={p:.2f} sigma={sigma:g} | gate ssMSE={gate_mse:.3e} ssMAE={gate_mae:.6f} "
              f"ARmMAE={gate_ar_mae:.6f} ARp99={gate_ar_p99:.6f} CR={cr_indist:.2f} "
              f"| peak={peak_gb:.2f}GB {dt:.1f}s")

        if gate_mse < best_val:                                # crown-jewel proxy
            best_val = gate_mse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in lit.state_dict().items()}
    logf.close()

    # ---- save best ckpt in load_frozen_backbone_acc format + copy config ----
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in lit.state_dict().items()}
        best_epoch, best_val = epochs - 1, gate_mse
    # Clear any prior run's ckpts in this cell's dir so find_best_ckpt_acc picks THIS
    # run's best unambiguously (exactly one ckpt per run).
    for old in os.listdir(ckpt_dir):
        if old.endswith(".ckpt"):
            os.remove(os.path.join(ckpt_dir, old))
    # Trailing -step=0 mirrors the committed ckpt convention so find_best_ckpt_acc's
    # greedy regex val_loss=([0-9.]+) terminates the number at '-' (a bare
    # val_loss=X.XXXXXXXX.ckpt would let the regex swallow the extension dot).
    ckpt_name = f"epoch={best_epoch:02d}-val_loss={best_val:.8f}-step=0.ckpt"
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    torch.save({"state_dict": best_state}, ckpt_path)
    meta = {"cell": args.cell, "src_run_dir": run_dir, "new_run_dir": new_run_dir,
            "arm": arm, "unroll_len": horizon, "lr": lr, "epochs": epochs,
            "scenario_batch": scenario_batch, "seed": seed,
            "lambda_tf": lambda_tf, "lambda_ar": lambda_ar,
            "loss_reduction": loss_reduction, "cvar_alpha": cvar_alpha,
            "tail_weight_pow": tail_weight_pow, "feedback_noise_sigma": [sigma0, sigma1],
            "best_epoch": best_epoch, "best_gate_ss_mse": best_val,
            "train_loss_hist": train_loss_hist, "smoke": bool(args.smoke),
            "max_scenarios": args.max_scenarios,
            "split": {"finetune": sorted(ft_set), "gate": sorted(gate_set),
                      "report": sorted(report_set)}}
    with open(os.path.join(new_run_dir, "arbb_train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{args.cell}][done] saved {ckpt_path}")
    print(f"[{args.cell}][done] copied config_used.yaml + meta to {new_run_dir}")
    print(f"[{args.cell}][done] log -> {log_path}")
    if len(train_loss_hist) >= 2:
        print(f"[{args.cell}] train loss trend: {train_loss_hist[0]:.6e} -> "
              f"{train_loss_hist[-1]:.6e} "
              f"({'DOWN' if train_loss_hist[-1] < train_loss_hist[0] else 'UP/FLAT'})")


if __name__ == "__main__":
    main()
