"""From-scratch scheduled-sampling (SS) training of a per-cell backbone.

ADDITIVE. Trains a NEW backbone (random init) with the SAME architecture, batch size,
optimizer and LR schedule as the cell's original backbone; the ONLY intended
difference is detached scheduled sampling over a short unroll horizon H.

Recipe parity with the original backbone (src/train.py + <run_dir>/config_used.yaml):
  * architecture: ModelSelector(model.name, backbone_kwargs, lightning_kwargs) from
    the ORIGINAL run_dir's config_used.yaml (SBO: transformer_decoder d64/L8/h4,
    dropout 0, input 14).
  * optimizer/scheduler: taken from LitRNNBaseModel.configure_optimizers() itself
    (AdamW lr=1e-3 + StepLR(step_size=2 epochs, gamma=0.1)) -- this is what the
    original Lightning run actually used (the yaml lr/scheduler keys are unused).
  * batch size 128 (data.batch_size), MSE loss, FP32 (TF32 off), seed 42.
  * epoch budget: original max_epochs=150 + EarlyStopping(patience=10); the original
    run stopped after 18 epochs. Here: patience 10 on the validation metric with a
    hard cap (--max-epochs, default 30 >= 18) so the run stays within budget.

Scheduled sampling (the only change):
  * Each training window target is visited exactly once per epoch, but windows are
    grouped into per-scenario CHUNKS of H consecutive windows (random per-scenario
    phase offset each epoch). A chunk starts from the ground-truth window and is
    unrolled H steps; after each step the fed-back continuous row is the model's own
    prediction (detached) w.p. p, else ground truth, per (chunk, step). Controls are
    always ground truth (same convention as the AR rollout).
  * One optimizer update per (128-chunk minibatch, unroll step), loss = MSE of that
    step's prediction -> the number of updates / batch size / targets per epoch match
    teacher forcing (up to the short partial chunks at scenario edges). To keep
    minibatches i.i.d.-like, chunks are processed in groups of 128*H: at unroll step
    k all H minibatches of the group take one update each, then the group advances.
  * p ramps linearly per optimizer update from 0 to --p-max over the first
    --ramp-epochs epochs (default 2 = the lr=1e-3 phase of StepLR), then stays.

Validation (scenario-disjoint 10% of train_csv scenarios, RandomState(42)):
  * val_ar_mae : beta=0 free-running AR micro-MAE (10-var mean |err| per step,
    pooled), byte-faithful to autoregressive_corrected_batched_acc (checked at the
    end with --parity-check, on by default). CHECKPOINT SELECTION METRIC.
  * val_tf_mse : single-step teacher-forced MSE (the original val_loss), logged.
TEST data is never touched.

Output (NEW dir, default <out_root>/ss_scratch/<cell>_backbone/):
  config_used.yaml (original model block + ss block), <CKPT_SUBDIR>/epoch=..ckpt
  ({"state_dict": {"backbone.*"}} -> load_frozen_backbone_acc strict-loads it),
  last.ckpt, train_log.json, split.json.

Usage (from src/, NONINTERACTIVE=1):
  python experiments/train_backbone_ss_scratch.py --cell SBO --smoke
  python experiments/train_backbone_ss_scratch.py --cell SBO
"""
import os
import sys
import json
import time
import argparse

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

os.environ.setdefault("NONINTERACTIVE", "1")

import numpy as np
import torch
import yaml
from pytorch_lightning import seed_everything

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

import utils
from accident_dataset import AccidentWindowDataset
from model_selector import ModelSelector
from error_rollout_acc import (
    CKPT_SUBDIR, load_frozen_backbone_acc, autoregressive_corrected_batched_acc,
)
from predict_batched import compute_micro_macro


def scenario_ranges(ds):
    """{scenario: (first_window_start_row, n_windows)}; windows of one scenario are
    contiguous rows, so window j of scenario s starts at row first+j."""
    starts = np.asarray(ds.series_index, dtype=np.int64)
    scen = ds._scenario[starts]
    cut = np.concatenate([[0], np.nonzero(scen[1:] != scen[:-1])[0] + 1, [scen.size]])
    out = {}
    for a, b in zip(cut[:-1], cut[1:]):
        s, f, n = int(scen[a]), int(starts[a]), int(b - a)
        assert s not in out, f"scenario {s} appears in two separate blocks"
        assert int(starts[b - 1]) == f + n - 1, f"non-contiguous windows in scenario {s}"
        out[s] = (f, n)
    return out


def split_scenarios(scens, val_frac, seed):
    """Scenario-disjoint split, same RandomState(seed) shuffle-then-partition as
    accident_dataset.scenario_disjoint_split."""
    shuffled = np.array(sorted(scens))
    np.random.RandomState(seed).shuffle(shuffled)
    n_train = int(round((1.0 - val_frac) * len(shuffled)))
    n_train = max(1, min(n_train, len(shuffled) - 1))
    return sorted(int(s) for s in shuffled[:n_train]), sorted(int(s) for s in shuffled[n_train:])


def build_chunks(ranges, scen_list, H, rng):
    """Per-scenario chunks of <=H consecutive windows with a random phase offset,
    covering every window exactly once. Returns (start_rows, lengths) shuffled."""
    starts, lens = [], []
    for s in scen_list:
        f, n = ranges[s]
        o = int(rng.randint(H))
        cuts = np.arange(o, n, H) if o == 0 else np.concatenate([[0], np.arange(o, n, H)])
        ln = np.diff(np.append(cuts, n))
        starts.append(f + cuts)
        lens.append(ln)
    starts = np.concatenate(starts)
    lens = np.concatenate(lens)
    perm = rng.permutation(starts.size)
    return starts[perm], lens[perm]


@torch.no_grad()
def val_tf_mse(model, feats, ranges, scen_list, seq, nc, batch=4096):
    model.eval()
    starts = np.concatenate([ranges[s][0] + np.arange(ranges[s][1]) for s in scen_list])
    ar = torch.arange(seq, device=feats.device)
    se, n = 0.0, 0
    for i in range(0, starts.size, batch):
        st = torch.from_numpy(starts[i:i + batch]).to(feats.device)
        x = feats[st[:, None] + ar[None, :]]
        y = feats[st + seq, :nc]
        d = model(x) - y
        se += float((d * d).sum())
        n += d.numel()
    return se / n


def pack_val(feats, ranges, scen_list, seq, nc):
    """Dense [S, maxL] packing equal to error_rollout_acc._collect_pass/_pack for
    pred_len=1: init window = rows [f, f+seq), step t target/control = row f+seq+t."""
    scen_list = sorted(scen_list, key=lambda s: ranges[s][0])   # dataset (csv) order
    S = len(scen_list)
    lengths = np.array([ranges[s][1] for s in scen_list], dtype=np.int64)
    maxL = int(lengths.max())
    first = torch.tensor([ranges[s][0] for s in scen_list], device=feats.device)
    W = feats[first[:, None] + torch.arange(seq, device=feats.device)[None, :]]
    t = torch.arange(maxL, device=feats.device)
    L_t = torch.from_numpy(lengths).to(feats.device)
    rows = first[:, None] + seq + torch.minimum(t[None, :], L_t[:, None] - 1)
    return {"W": W, "CY": feats[rows, :nc], "UY": feats[rows, nc:], "lengths": L_t,
            "maxL": maxL, "S": S}


@torch.no_grad()
def val_ar(model, pk, nc):
    """beta=0 free-running AR on the packed val set -> (micro_mae, p99 step err)."""
    model.eval()
    window = pk["W"]
    errs = []
    for t in range(pk["maxL"]):
        raw = model(window)
        errs.append((raw - pk["CY"][:, t, :]).abs().mean(dim=1))
        next_row = torch.cat([raw, pk["UY"][:, t, :]], dim=1)
        window = torch.cat([window[:, 1:, :], next_row[:, None, :]], dim=1)
    E = torch.stack(errs, dim=1)                                    # [S, maxL]
    mask = torch.arange(pk["maxL"], device=E.device)[None, :] < pk["lengths"][:, None]
    flat = E[mask].double().cpu().numpy()
    return float(flat.mean()), float(np.quantile(flat, 0.99))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--cells-config", default="error_mlp_accident")
    ap.add_argument("--out-dir", default=None,
                    help="NEW run dir (default <out_root>/ss_scratch/<cell>_backbone).")
    ap.add_argument("--unroll-len", type=int, default=10, help="SS unroll horizon H.")
    ap.add_argument("--p-max", type=float, default=0.5, help="final SS feed-back probability.")
    ap.add_argument("--ramp-epochs", type=float, default=2.0,
                    help="epochs over which p ramps linearly 0 -> p_max.")
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-scenarios", type=int, default=None, help="SMOKE: cap scenarios.")
    ap.add_argument("--max-steps-per-epoch", type=int, default=None, help="SMOKE: cap updates.")
    ap.add_argument("--smoke", action="store_true",
                    help="60 scenarios, 2 epochs, 300 updates/epoch, out-dir suffix _smoke.")
    ap.add_argument("--no-parity-check", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.max_scenarios = args.max_scenarios or 60
        args.max_epochs = min(args.max_epochs, 2)
        args.max_steps_per_epoch = args.max_steps_per_epoch or 300

    ccfg = utils.load_config(args.cells_config)
    cell = ccfg["cells"][args.cell]
    cfg_data, cfg_tr = ccfg["data"], ccfg["training"]
    seq, nc = int(cfg_data["seq_len"]), int(cfg_data["num_continuous"])
    H, p_max = int(args.unroll_len), float(args.p_max)
    out_dir = args.out_dir or os.path.join(
        ccfg["out_root"], "ss_scratch", f"{args.cell}_backbone" + ("_smoke" if args.smoke else ""))
    ckpt_dir = os.path.join(out_dir, CKPT_SUBDIR)
    if os.path.isdir(ckpt_dir) and any(f.endswith(".ckpt") for f in os.listdir(ckpt_dir)) \
            and not args.smoke:
        raise FileExistsError(f"{ckpt_dir} already has checkpoints; refusing to overwrite")
    os.makedirs(ckpt_dir, exist_ok=True)

    seed_everything(args.seed, workers=True)
    device = torch.device(cfg_tr["device"] if torch.cuda.is_available() else "cpu")

    with open(os.path.join(cell["run_dir"], "config_used.yaml")) as f:
        orig_cfg = yaml.safe_load(f)
    _, lit = ModelSelector(orig_cfg["model"]["name"],
                           backbone_kwargs=orig_cfg["model"]["backbone_kwargs"],
                           lightning_kwargs=orig_cfg["model"].get("lightning_kwargs", {}))
    model = lit.backbone.to(device=device, dtype=torch.float32)
    (opt,), (sched,) = lit.configure_optimizers()
    batch = int(orig_cfg["data"]["batch_size"])

    ds = AccidentWindowDataset(cell["train_csv"], seq_len=seq, pred_len=1,
                               cache_dir=cfg_data.get("cache_dir"),
                               max_scenarios=args.max_scenarios)
    assert ds.input_size == int(orig_cfg["model"]["backbone_kwargs"]["input_size"])
    feats = torch.from_numpy(ds._feats).to(device)
    ranges = scenario_ranges(ds)
    tr_scen, va_scen = split_scenarios(list(ranges), args.val_frac, args.seed)
    n_tr_win = sum(ranges[s][1] for s in tr_scen)
    n_va_win = sum(ranges[s][1] for s in va_scen)
    print(f"[{args.cell}] windows={len(ds)} train scen={len(tr_scen)} win={n_tr_win} | "
          f"val scen={len(va_scen)} win={n_va_win} | H={H} p_max={p_max} batch={batch}", flush=True)
    with open(os.path.join(out_dir, "split.json"), "w") as f:
        json.dump({"cell": args.cell, "seed": args.seed, "val_frac": args.val_frac,
                   "train_scenarios": tr_scen, "val_scenarios": va_scen}, f)

    pk = pack_val(feats, ranges, va_scen, seq, nc)
    est_steps = int(np.ceil(n_tr_win / batch))
    ramp_steps = max(1, int(args.ramp_epochs * est_steps))
    if args.max_steps_per_epoch:
        ramp_steps = max(1, int(args.ramp_epochs * args.max_steps_per_epoch))
    G = batch * H
    ar_seq = torch.arange(seq, device=device)
    fb_gen = torch.Generator(device=device).manual_seed(args.seed)

    log, best, bad, gstep = [], (float("inf"), -1), 0, 0
    best_path = None
    for epoch in range(args.max_epochs):
        t0 = time.time()
        model.train()
        lr = opt.param_groups[0]["lr"]
        rng = np.random.RandomState(args.seed + epoch)
        c_start, c_len = build_chunks(ranges, tr_scen, H, rng)
        loss_sum, n_upd, n_tgt, n_fed, n_fed_pred = 0.0, 0, 0, 0, 0
        stop_epoch = False
        for g0 in range(0, c_start.size, G):
            S0 = torch.from_numpy(c_start[g0:g0 + G]).to(device)
            Lc = torch.from_numpy(c_len[g0:g0 + G]).to(device)
            g = S0.numel()
            W = feats[S0[:, None] + ar_seq[None, :]]                     # [g, seq, in]
            for k in range(int(Lc.max())):
                active = k < Lc
                tgt_rows = torch.where(active, S0 + seq + k, S0 + seq)
                Y = feats[tgt_rows, :nc]
                P = torch.empty_like(Y)
                for b0 in range(0, g, batch):
                    sl = slice(b0, b0 + batch)
                    act = active[sl]
                    if not bool(act.any()):
                        P[sl] = Y[sl]
                        continue
                    pred = model(W[sl])
                    loss = ((pred - Y[sl]) ** 2)[act].mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    P[sl] = pred.detach()
                    loss_sum += float(loss.detach()) * int(act.sum())
                    n_tgt += int(act.sum())
                    n_upd += 1
                    gstep += 1
                p = p_max * min(1.0, gstep / ramp_steps)
                use_pred = torch.rand((g, 1), device=device, generator=fb_gen) < p
                fed = torch.where(use_pred, P, Y)
                n_fed += int(active.sum()); n_fed_pred += int((use_pred[:, 0] & active).sum())
                next_row = torch.cat([fed, feats[tgt_rows, nc:]], dim=1)
                W = torch.cat([W[:, 1:, :], next_row[:, None, :]], dim=1)
                if args.max_steps_per_epoch and n_upd >= args.max_steps_per_epoch:
                    stop_epoch = True
                    break
            if stop_epoch:
                break
        sched.step()
        t_train = time.time() - t0
        vmse = val_tf_mse(model, feats, ranges, va_scen, seq, nc)
        vmae, vp99 = val_ar(model, pk, nc)
        rec = {"epoch": epoch, "lr": lr, "p_end": p_max * min(1.0, gstep / ramp_steps),
               "realized_feed_pred_frac": n_fed_pred / max(1, n_fed),
               "train_mse": loss_sum / max(1, n_tgt), "n_updates": n_upd, "n_targets": n_tgt,
               "val_tf_mse": vmse, "val_ar_mae": vmae, "val_ar_p99": vp99,
               "sec_train": t_train, "sec_total": time.time() - t0}
        log.append(rec)
        print(f"[{args.cell}][ep {epoch}] lr={lr:.1e} p={rec['p_end']:.3f} "
              f"train_mse={rec['train_mse']:.3e} upd={n_upd} | val_tf_mse={vmse:.3e} "
              f"val_ar_mae={vmae:.6f} val_ar_p99={vp99:.6f} | {rec['sec_total']:.0f}s", flush=True)

        state = {"backbone." + k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save({"state_dict": state, "epoch": epoch},
                   os.path.join(ckpt_dir, "last.ckpt"))
        if vmae < best[0]:
            best, bad = (vmae, epoch), 0
            if best_path and os.path.exists(best_path):
                os.remove(best_path)
            # val_loss slot holds the SELECTION metric (val AR micro-MAE); trailing
            # -step= keeps find_best_ckpt_acc's regex off the extension dot.
            best_path = os.path.join(
                ckpt_dir, f"epoch={epoch:02d}-val_loss={vmae:.8f}-step={gstep}.ckpt")
            torch.save({"state_dict": state, "epoch": epoch, "val_ar_mae": vmae,
                        "val_tf_mse": vmse}, best_path)
        else:
            bad += 1
        cfg_out = dict(orig_cfg)
        cfg_out["scheduled_sampling"] = {
            "from_scratch": True, "unroll_len": H, "p_max": p_max,
            "ramp_epochs": args.ramp_epochs, "ramp": "linear per update", "feed": "detached",
            "selection_metric": "val_ar_mae (beta=0 AR micro-MAE, scenario-disjoint val)",
            "patience": args.patience, "max_epochs": args.max_epochs,
            "val_frac": args.val_frac, "seed": args.seed, "source_run_dir": cell["run_dir"],
            "optimizer": "LitRNNBaseModel.configure_optimizers (AdamW 1e-3, StepLR 2/0.1)",
            "max_scenarios": args.max_scenarios, "max_steps_per_epoch": args.max_steps_per_epoch}
        with open(os.path.join(out_dir, "config_used.yaml"), "w") as f:
            yaml.safe_dump(cfg_out, f)
        with open(os.path.join(out_dir, "train_log.json"), "w") as f:
            json.dump({"best_epoch": best[1], "best_val_ar_mae": best[0],
                       "epochs": log}, f, indent=2)
        if bad >= args.patience:
            print(f"[{args.cell}] early stop at epoch {epoch} (best epoch {best[1]})")
            break

    print(f"[{args.cell}][done] best epoch={best[1]} val_ar_mae={best[0]:.6f} -> {best_path}", flush=True)

    if not args.no_parity_check:
        m, _ = load_frozen_backbone_acc(out_dir, device)
        fast_mae, _ = val_ar(m, pk, nc)
        pd_, td_ = autoregressive_corrected_batched_acc(
            m, None, 0.0, ds, ds.num_controls, float(cell["step_norm_const"]),
            device=device, num_continuous=nc, collect_batch=int(cfg_tr.get("collect_batch", 2048)),
            num_workers=int(cfg_tr["num_workers"]), restrict_scenarios=va_scen)
        ref = compute_micro_macro(pd_, td_)["micro_mae"]
        print(f"[{args.cell}][parity] fast val_ar_mae={fast_mae:.10f} "
              f"canonical={ref:.10f} |diff|={abs(fast_mae - ref):.3e} best={best[0]:.10f}")
        assert abs(fast_mae - ref) <= 1e-7 * max(1.0, abs(ref)), "val AR parity FAILED"
        with open(os.path.join(out_dir, "train_log.json")) as f:
            tl = json.load(f)
        tl["parity_check"] = {"fast": fast_mae, "canonical": ref, "abs_diff": abs(fast_mae - ref)}
        with open(os.path.join(out_dir, "train_log.json"), "w") as f:
            json.dump(tl, f, indent=2)


if __name__ == "__main__":
    main()
