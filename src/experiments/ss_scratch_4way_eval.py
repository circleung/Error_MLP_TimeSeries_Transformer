"""4-way TEST comparison: original vs from-scratch scheduled-sampling (SS) backbone,
each with and without its own DAgger ErrorMLP corrector.

  arm1  base backbone, beta=0
  arm2  base backbone + adopted error_mlp_dagger.pt  (op selected on VALIDATION)
  arm3  SS backbone, beta=0
  arm4  SS backbone + its own DAgger ErrorMLP        (op selected on VALIDATION)

Corrector operating points follow op_val_select_test_eval.py exactly: grid
GATED_BETAS x GATE_FRACTIONS on the corrector's held-out VALIDATION scenarios
(train_csv), tau = (1-q) quantile of ||e_hat||_2 on the validation baseline rollout,
pick min val p99 s.t. val mean <= val baseline mean, freeze (beta*, tau*), then ONE
test pass. All rollouts go through gated_corrected_rollout_acc (beta=0/tau=inf is
the exact uncorrected AR null-op).

Run from src/ (NONINTERACTIVE=1):
  python experiments/ss_scratch_4way_eval.py --cell SBO \
      --ss-run-dir <out_root>/ss_scratch/SBO_backbone --ss-mlp <out_root>/ss_scratch/SBO/error_mlp_dagger.pt
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
from torch.utils.data import DataLoader, Subset

import utils
from accident_dataset import AccidentWindowDataset, CONTINUOUS_COLS
from models.error_mlp import ErrorMLP
from error_rollout_acc import (
    load_frozen_backbone_acc, baseline_rollout_with_stats_acc, gated_corrected_rollout_acc,
)
from experiments.tail_analysis_acc import (
    step_err_from_dicts, tail_metrics, GATE_FRACTIONS, GATED_BETAS,
)

PEX = CONTINUOUS_COLS.index("PEX0(17)")
ANCHORS = {"arm1": {"step_p99": 0.08953, "pex_mae": 0.02402, "pex_p99": 0.2058},
           "arm2": {"step_p99": 0.05701, "pex_mae": 0.01758, "pex_p99": 0.0978}}


def load_mlp(path, cfg, device):
    pt = torch.load(path, map_location="cpu")
    mlp = ErrorMLP(in_dim=int(pt["in_dim"]), **cfg["error_mlp"])
    mlp.load_state_dict(pt["state_dict"])
    return mlp.to(device).eval(), pt


@torch.inference_mode()
def tf_mse(model, ds, device, nc, batch=4096, num_workers=4):
    """Single-step teacher-forced MSE (overall + per variable) over all windows."""
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=num_workers)
    se, n = torch.zeros(nc, dtype=torch.float64), 0
    for b in dl:
        d = model(b["past_values"].to(device)) - b["continuous_y"].to(device)
        se += (d.double() ** 2).sum(dim=0).cpu()
        n += d.shape[0]
    per = (se / n).numpy()
    return float(per.mean()), per.tolist()


def rollout(model, mlp, beta, tau, ds, ncontrols, snc, device, nw, cb, nc,
            restrict=None, with_fire=False):
    out = gated_corrected_rollout_acc(
        model, mlp, beta, tau, ds, ncontrols, snc, device=device, gate_on="pred",
        num_continuous=nc, collect_batch=cb, num_workers=nw, restrict_scenarios=restrict,
        collect_dagger=with_fire)
    if with_fire:
        return out[0], out[1], float(out[5]["intervention_rate"])
    return out[0], out[1], None


def full_metrics(pdict, tdict):
    """Step-level (10-var mean) + per-variable metrics, plus per-scenario arrays."""
    scen = sorted(pdict)
    E = {s: np.abs(np.vstack(pdict[s]).astype(np.float64) - np.vstack(tdict[s]).astype(np.float64))
         for s in scen}                                              # [L, 10] abs err
    se, ps = step_err_from_dicts(pdict, tdict)
    step = tail_metrics(se, ps)
    # ABC-Transformer definitions (plot_cell_metrics.metrics): macro over scenarios,
    # RMSE with the root inside the time average.
    raw = {s: np.vstack(pdict[s]).astype(np.float64) - np.vstack(tdict[s]).astype(np.float64)
           for s in scen}
    step["macro_mae"] = float(np.mean([np.abs(raw[s]).mean() for s in scen]))
    step["macro_rmse"] = float(np.mean([np.sqrt((raw[s] ** 2).mean(axis=1)).mean() for s in scen]))
    allE = np.concatenate([E[s] for s in scen], axis=0)             # [N, 10]
    scen_var = np.stack([E[s].mean(axis=0) for s in scen])          # [S, 10]
    per_var = {}
    for v, name in enumerate(CONTINUOUS_COLS):
        per_var[name] = {
            "mae": float(allE[:, v].mean()),
            "p99": float(np.quantile(allE[:, v], 0.99)),
            "max": float(allE[:, v].max()),
            "worst10_scen_mean": float(np.sort(scen_var[:, v])[::-1][:10].mean()),
        }
    return {"step": step, "per_var": per_var,
            "scen": scen, "scen_step_mean": np.array([E[s].mean() for s in scen]),
            "scen_var": scen_var,
            "pex_by_step": [E[s][:, PEX] for s in scen],
            "pex_pred": [np.vstack(pdict[s])[:, PEX] for s in scen],
            "pex_true": [np.vstack(tdict[s])[:, PEX] for s in scen]}


def compare_scen(m, ref):
    assert m["scen"] == ref["scen"]
    out = {}
    for key, a, b in (("step", m["scen_step_mean"], ref["scen_step_mean"]),
                      ("PEX0(17)", m["scen_var"][:, PEX], ref["scen_var"][:, PEX])):
        out[key] = {"improved": int((a < b).sum()), "worsened": int((a > b).sum()),
                    "tied": int((a == b).sum())}
    return out


def select_op(model, mlp, train_ds, heldout, ncontrols, snc, device, nw, cb, nc, tag):
    """op_val_select_test_eval.run_cell selection, verbatim logic."""
    vstats = baseline_rollout_with_stats_acc(
        model, mlp, train_ds, ncontrols, snc, device=device, num_continuous=nc,
        collect_batch=cb, num_workers=nw, restrict_scenarios=heldout)
    pdv, tdv, _ = rollout(model, mlp, 0.0, np.inf, train_ds, ncontrols, snc, device,
                          nw, cb, nc, heldout)
    vbase = tail_metrics(*step_err_from_dicts(pdv, tdv))
    tau_of = {q: float(np.quantile(vstats["g"], 1.0 - q)) for q in GATE_FRACTIONS}
    grid = {}
    for beta in GATED_BETAS:
        for q in GATE_FRACTIONS:
            pd_, td_, _ = rollout(model, mlp, beta, tau_of[q], train_ds, ncontrols, snc,
                                  device, nw, cb, nc, heldout)
            grid[(beta, q)] = tail_metrics(*step_err_from_dicts(pd_, td_))
            print(f"[{tag}][val] beta={beta} q={q} mean={grid[(beta, q)]['mean']:.6f} "
                  f"p99={grid[(beta, q)]['p99']:.6f}", flush=True)
    cands = [(k, m) for k, m in grid.items() if m["mean"] <= vbase["mean"]]
    relaxed = False
    if not cands:
        cands = [(k, m) for k, m in grid.items() if m["mean"] <= vbase["mean"] * 1.02]
        relaxed = True
    (beta_s, q_s), vsel = min(cands, key=lambda kv: kv[1]["p99"])
    print(f"[{tag}][VAL-PICK] beta*={beta_s} q*={q_s} tau*={tau_of[q_s]:.6f} relaxed={relaxed}")
    return {"beta_star": beta_s, "q_star": q_s, "tau_star": tau_of[q_s], "relaxed": relaxed,
            "val_baseline": vbase, "val_op": vsel, "n_heldout": len(heldout),
            "val_grid": [{"beta": k[0], "q": k[1], "mean": m["mean"], "p99": m["p99"]}
                         for k, m in grid.items()]}


def strip(m):
    return {"step": m["step"], "per_var": m["per_var"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="SBO")
    ap.add_argument("--ss-run-dir", required=True)
    ap.add_argument("--ss-mlp", required=True)
    ap.add_argument("--base-mlp", default=None, help="default <out_root>/<cell>/error_mlp_dagger.pt")
    ap.add_argument("--arms", default="1234", help="subset of arms to run, e.g. 12")
    ap.add_argument("--max-scenarios", type=int, default=None, help="SMOKE: cap scenarios.")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-npz", default=None,
                    help="per-step PEX0(17) |err|, prediction and GT per arm (figures).")
    ap.add_argument("--extra-ss-ckpts", nargs="*", default=[],
                    help="sensitivity: other SS ckpt files (relative to --ss-run-dir) evaluated "
                         "on TEST at beta=0 only.")
    args = ap.parse_args()

    cfg = utils.load_config("error_mlp_accident")
    cell = cfg["cells"][args.cell]
    cfg_tr, cfg_data = cfg["training"], cfg["data"]
    nw = int(cfg_tr["num_workers"]); cb = int(cfg_tr.get("collect_batch", 2048))
    seq, nc = int(cfg_data["seq_len"]), int(cfg_data["num_continuous"])
    snc = float(cell["step_norm_const"]); cache = cfg_data.get("cache_dir")
    device = torch.device(cfg_tr["device"] if torch.cuda.is_available() else "cpu")
    base_mlp_path = args.base_mlp or os.path.join(cfg["out_root"], args.cell, "error_mlp_dagger.pt")

    test_ds = AccidentWindowDataset(cell["test_csv"], seq_len=seq, pred_len=1, cache_dir=cache,
                                    max_scenarios=args.max_scenarios)
    train_full = AccidentWindowDataset(cell["train_csv"], seq_len=seq, pred_len=1,
                                       cache_dir=cache, max_scenarios=args.max_scenarios)
    ncontrols = test_ds.num_controls

    arms = {"arm1": ("base", None), "arm2": ("base", base_mlp_path),
            "arm3": ("ss", None), "arm4": ("ss", args.ss_mlp)}
    run_dirs = {"base": cell["run_dir"], "ss": args.ss_run_dir}
    res = {"cell": args.cell, "run_dirs": run_dirs, "base_mlp": base_mlp_path,
           "ss_mlp": args.ss_mlp, "n_test_scen": None, "tf_mse": {}, "arms": {},
           "vs_arm1": {}, "max_scenarios": args.max_scenarios}
    metrics, pex = {}, {}
    for bb in ("base", "ss"):
        todo = [a for a in arms if arms[a][0] == bb and a[-1] in args.arms]
        if not todo:
            continue
        model, _ = load_frozen_backbone_acc(run_dirs[bb], device)
        mse, mse_var = tf_mse(model, test_ds, device, nc)
        res["tf_mse"][bb] = {"mse": mse, "per_var": dict(zip(CONTINUOUS_COLS, mse_var))}
        print(f"[{bb}] test single-step TF MSE={mse:.4e}", flush=True)
        for a in todo:
            mlp_path = arms[a][1]
            entry = {"backbone": bb, "run_dir": run_dirs[bb], "mlp": mlp_path}
            if mlp_path is None:
                pd_, td_, _ = rollout(model, None, 0.0, np.inf, test_ds, ncontrols, snc,
                                      device, nw, cb, nc)
            else:
                mlp, pt = load_mlp(mlp_path, cfg, device)
                heldout = [int(s) for s in pt["heldout_scenarios"]]
                ws = train_full.window_scenarios()
                hs = set(heldout)
                val_ds = Subset(train_full, [i for i, s in enumerate(ws) if int(s) in hs])
                op = select_op(model, mlp, val_ds, heldout, ncontrols, snc, device, nw, cb,
                               nc, a)
                entry["op"] = op
                pd_, td_, fire = rollout(model, mlp, op["beta_star"], op["tau_star"], test_ds,
                                         ncontrols, snc, device, nw, cb, nc, with_fire=True)
                entry["fire_rate"] = fire
            m = full_metrics(pd_, td_)
            metrics[a] = m
            entry.update(strip(m))
            res["arms"][a] = entry
            res["n_test_scen"] = len(m["scen"])
            pv = m["per_var"]["PEX0(17)"]
            print(f"[{a}] step mean={m['step']['mean']:.6f} p99={m['step']['p99']:.6f} "
                  f"max={m['step']['max']:.6f} w10={m['step']['worst10_scen_mean']:.6f} | "
                  f"PEX mae={pv['mae']:.6f} p99={pv['p99']:.6f} max={pv['max']:.6f}"
                  + (f" | fire={entry['fire_rate']*100:.3f}%" if mlp_path else ""), flush=True)
            pex[a] = m["pex_by_step"]
        del model
        torch.cuda.empty_cache()

    res["ss_ckpt_sensitivity"] = {}
    for ck in args.extra_ss_ckpts:
        model, _ = load_frozen_backbone_acc(args.ss_run_dir, device, ckpt=ck)
        pd_, td_, _ = rollout(model, None, 0.0, np.inf, test_ds, ncontrols, snc, device, nw, cb, nc)
        m = full_metrics(pd_, td_)
        res["ss_ckpt_sensitivity"][ck] = {"step": m["step"],
                                          "PEX0(17)": m["per_var"]["PEX0(17)"]}
        print(f"[ss ckpt {ck}] step mean={m['step']['mean']:.6f} p99={m['step']['p99']:.6f} "
              f"max={m['step']['max']:.6f}", flush=True)
        del model
        torch.cuda.empty_cache()

    if "arm1" in metrics:
        for a in metrics:
            if a != "arm1":
                res["vs_arm1"][a] = compare_scen(metrics[a], metrics["arm1"])
    if args.max_scenarios is None:
        checks = {}
        for a, anc in ANCHORS.items():
            if a not in metrics:
                continue
            got = {"step_p99": metrics[a]["step"]["p99"],
                   "pex_mae": metrics[a]["per_var"]["PEX0(17)"]["mae"],
                   "pex_p99": metrics[a]["per_var"]["PEX0(17)"]["p99"]}
            ok = all(abs(got[k] - v) <= 0.5 * 10 ** -(len(repr(v).split(".")[1]))
                     for k, v in anc.items())
            checks[a] = {"expected": anc, "got": got, "ok": bool(ok)}
            print(f"[anchor] {a} ok={ok} got={got} expected={anc}")
        res["anchor_check"] = checks

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(res, f, indent=2)
    if args.out_npz:
        maxL = max(len(e) for lst in pex.values() for e in lst)

        def pad(lst):
            arr = np.full((len(lst), maxL), np.nan)
            for i, e in enumerate(lst):
                arr[i, :len(e)] = e
            return arr                                               # [S, maxL], NaN-padded
        packed = {a: pad(lst) for a, lst in pex.items()}
        packed.update({f"{a}_pred": pad(metrics[a]["pex_pred"]) for a in metrics})
        packed["true"] = pad(metrics[next(iter(metrics))]["pex_true"])
        np.savez_compressed(args.out_npz, scen=np.array(metrics[next(iter(metrics))]["scen"]),
                            **packed)
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
