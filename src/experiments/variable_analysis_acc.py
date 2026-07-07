"""Per-accident-type QUANTITATIVE variable analysis (NO retraining; frozen backbone).

ADDITIVE, read-only-on-the-backbone. REUSES the per-cell frozen backbone
(load_frozen_backbone_acc), the AccidentWindowDataset windowing, and the same
compute-error conventions as the rest of the acc/ pipeline. It does NOT modify any
existing acc/60min file and does NOT retrain anything (the backbone is asserted
frozen after load).

Produces, per cell on the TEST set, the evidence base for a variable-removal /
extreme-scenario follow-up:

  1. VARIABLE IMPORTANCE for PREDICTION -- PERMUTATION importance in TEACHER-FORCING
     next-step. For each INPUT feature channel (the 10 continuous AND the per-cell
     controls), shuffle that channel's values ACROSS the test windows (keep all
     other channels intact), run the frozen backbone once (single forward -- no
     rollout, no feedback), and measure the increase in next-step prediction error:
        dMAE_overall = MAE(permuted) - MAE(baseline)   (mean over the 10 outputs)
        dMAE_out[k]  = per-output-variable delta
     Averaged over 3 shuffles (fixed seed) for stability. Ranked table by dMAE.
     The SAME shuffle permutation is applied to every timestep of the chosen
     channel within a window (a channel is permuted as a whole across samples).

  2. Per-OUTPUT-variable error from the BASELINE uncorrected AR rollout (beta=0,
     reusing autoregressive_corrected_batched_acc) -- per each of the 10 continuous
     OUTPUTS: MAE and p99 (pooled over all scenario x step). Ranked table.

  3. WORST-scenario error attribution -- top-20 worst scenarios by per-scenario mean
     AR error; within those, decompose the pooled error by output variable: the
     fraction of the worst-scenario error carried by each variable (share), the top
     contributors, and the cumulative share of the top-3 variables.

  4. LINK importance <-> error -- is the most IMPORTANT input variable (#1, over the
     10 continuous, which are ALSO outputs) among the hardest OUTPUTS (#2) and the
     top worst-scenario contributors (#3)? Quantified with Spearman rank correlation
     between per-continuous-variable import rank and per-continuous-variable output-
     error rank (and vs worst-scenario share), plus a short per-cell verdict.

Notes:
  * Importance is over ALL input channels (10 continuous + controls); the link
    stats (#4) are over the 10 continuous only, since inputs<->outputs share those
    names (controls are inputs-only known-future covariates, not outputs).
  * The permutation-importance error is MAE of the RAW backbone next-step prediction
    vs continuous_y (teacher forcing) -- the SAME per-output |pred-true| unit as
    compute_micro_macro, just decomposed per output channel.

Writes <out_root>/<cell>/variable_analysis.json.

Usage (from src/):
    NONINTERACTIVE=1 python experiments/variable_analysis_acc.py --cell SBO
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
import pandas as pd
import torch
from torch.utils.data import DataLoader

import utils
from accident_dataset import (
    AccidentWindowDataset,
    infer_schema_from_csv,
    CONTINUOUS_COLS,
    NUM_CONTINUOUS,
)
from error_rollout_acc import (
    load_frozen_backbone_acc,
    autoregressive_corrected_batched_acc,
)

N_SHUFFLES = 3          # permutation-importance shuffles averaged for stability
PERM_SEED = 1234        # fixed base seed for the permutations
WORST_N = 20            # top-N worst scenarios for the attribution
TOPK_CONTRIB = 3        # cumulative-share horizon reported for the attribution


def assert_frozen(model):
    """Hard-assert the backbone carries no trainable parameters (NO retraining)."""
    n_trainable = sum(int(p.requires_grad) for p in model.parameters())
    assert n_trainable == 0, f"backbone is NOT frozen: {n_trainable} trainable params"
    assert not model.training, "backbone must be in eval() mode"


# ---------------------------------------------------------------------------
# 1. Permutation importance (teacher-forcing next-step, single forward).
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _collect_teacher_forcing(model, dataset, device, batch_size, num_workers,
                             num_continuous):
    """One TF pass over TEST: stack past windows [N, seq, input], continuous targets
    [N, 10], and baseline raw next-step predictions [N, 10]. Returns
    (past[N,seq,input] float32, cont_y[N,10] float32, base_pred[N,10] float32)."""
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers)
    past_chunks, y_chunks, pred_chunks = [], [], []
    for batch in dl:
        pv = batch["past_values"].to(device, torch.float32)          # [B, seq, input]
        out = model(pv)                                              # [B, 10]
        past_chunks.append(pv.detach().cpu().numpy().astype(np.float32))
        y_chunks.append(batch["continuous_y"].numpy().astype(np.float32))
        pred_chunks.append(out.detach().cpu().numpy().astype(np.float32))
    past = np.concatenate(past_chunks, axis=0)
    cont_y = np.concatenate(y_chunks, axis=0)
    base_pred = np.concatenate(pred_chunks, axis=0)
    return past, cont_y, base_pred


@torch.inference_mode()
def _forward_all(model, past, device, batch_size):
    """Forward a big [N, seq, input] float32 array through the frozen backbone in
    minibatches; return [N, 10] float32 predictions."""
    N = past.shape[0]
    outs = np.empty((N, NUM_CONTINUOUS), np.float32)
    for i in range(0, N, batch_size):
        xb = torch.from_numpy(past[i:i + batch_size]).to(device, torch.float32)
        outs[i:i + batch_size] = model(xb).detach().cpu().numpy()
    return outs


def permutation_importance(model, past, cont_y, base_pred, feature_cols, device,
                           batch_size, n_shuffles=N_SHUFFLES, seed=PERM_SEED):
    """Per-INPUT-channel permutation importance in teacher forcing.

    past [N, seq, input], base_pred [N,10] = model(past) (unpermuted). For each input
    channel c, shuffle past[:, :, c] across the N samples (same permutation applied to
    the whole channel across all seq positions of a sample), forward, and measure the
    increase in next-step MAE vs cont_y. Averaged over n_shuffles.

    Returns dict:
      baseline_mae (float, over 10 outputs),
      baseline_mae_per_out [10],
      per_feature: list of dicts (one per input channel) with dMAE_overall,
        dMAE_overall_std, dMAE_per_out [10], plus name/index/is_control.
    """
    N, _, input_size = past.shape
    assert input_size == len(feature_cols), (
        f"input_size {input_size} != len(feature_cols) {len(feature_cols)}")

    base_abs = np.abs(base_pred - cont_y)                    # [N, 10]
    baseline_mae_per_out = base_abs.mean(axis=0)             # [10]
    baseline_mae = float(baseline_mae_per_out.mean())

    rng = np.random.RandomState(seed)
    # Pre-draw one permutation per (channel, shuffle) for reproducibility.
    perms = {c: [rng.permutation(N) for _ in range(n_shuffles)]
             for c in range(input_size)}

    per_feature = []
    for c in range(input_size):
        d_overall_runs = []
        d_per_out_runs = []
        orig_col = past[:, :, c].copy()                     # [N, seq]
        for s in range(n_shuffles):
            past[:, :, c] = orig_col[perms[c][s]]           # permute channel across samples
            pred = _forward_all(model, past, device, batch_size)  # [N, 10]
            perm_abs = np.abs(pred - cont_y)                # [N, 10]
            perm_mae_per_out = perm_abs.mean(axis=0)        # [10]
            d_per_out_runs.append(perm_mae_per_out - baseline_mae_per_out)
            d_overall_runs.append(float(perm_mae_per_out.mean()) - baseline_mae)
        past[:, :, c] = orig_col                            # restore (no leakage)
        d_per_out = np.mean(d_per_out_runs, axis=0)         # [10]
        per_feature.append({
            "index": c,
            "name": feature_cols[c],
            "is_control": c >= NUM_CONTINUOUS,
            "dMAE_overall": float(np.mean(d_overall_runs)),
            "dMAE_overall_std": float(np.std(d_overall_runs)),
            "dMAE_per_out": [float(v) for v in d_per_out],
        })

    return {
        "baseline_mae": baseline_mae,
        "baseline_mae_per_out": [float(v) for v in baseline_mae_per_out],
        "per_feature": per_feature,
    }


# ---------------------------------------------------------------------------
# 2/3. Baseline AR rollout -> per-output error + per-scenario per-output error.
# ---------------------------------------------------------------------------


def per_output_ar_error(predictions_dict, true_dict, num_continuous=NUM_CONTINUOUS):
    """From the (predictions_dict, true_dict) of a baseline (beta=0) AR rollout,
    compute per-output-variable MAE and p99 pooled over all (scenario, step), AND the
    per-scenario per-output absolute-error sums (for worst-scenario attribution).

    Returns:
      per_out_mae   [10]  pooled mean |pred-true| per output channel,
      per_out_p99   [10]  pooled p99 of |pred-true| per output channel,
      per_scen: {sid: {"abs_sum": [10], "n": L, "scen_mean": float}}
        abs_sum = sum over the scenario's steps of |pred-true| per output,
        scen_mean = mean over 10 vars & steps (== compute_micro_macro per-scenario MAE).
    """
    # Pool all per-step abs errors per output to get global MAE/p99.
    abs_pool = [[] for _ in range(num_continuous)]
    per_scen = {}
    for sc in predictions_dict:
        if not predictions_dict[sc]:
            continue
        P = np.vstack([np.ravel(p) for p in predictions_dict[sc]]).astype(np.float64)
        Y = np.vstack([np.ravel(y) for y in true_dict[sc]]).astype(np.float64)
        A = np.abs(P - Y)                                   # [L, 10]
        for k in range(num_continuous):
            abs_pool[k].append(A[:, k])
        per_scen[int(sc)] = {
            "abs_sum": A.sum(axis=0),                       # [10]
            "n": int(A.shape[0]),
            "scen_mean": float(A.mean()),                  # mean over 10 vars & steps
        }
    per_out_mae = np.zeros(num_continuous)
    per_out_p99 = np.zeros(num_continuous)
    for k in range(num_continuous):
        col = np.concatenate(abs_pool[k]) if abs_pool[k] else np.zeros((0,))
        per_out_mae[k] = float(col.mean()) if col.size else float("nan")
        per_out_p99[k] = float(np.quantile(col, 0.99)) if col.size else float("nan")
    return per_out_mae, per_out_p99, per_scen


def worst_scenario_attribution(per_scen, cont_cols, worst_n=WORST_N, topk=TOPK_CONTRIB):
    """Top-`worst_n` worst scenarios by per-scenario mean AR error; decompose the
    POOLED absolute error over those scenarios by output variable.

    share[k] = (sum over worst scenarios of abs_sum[k]) / (total abs over worst).
    Returns dict with the worst scenario ids, their mean errors, per-variable share
    (ranked desc), and the cumulative share of the top-`topk` variables."""
    ranked = sorted(per_scen.items(), key=lambda kv: kv[1]["scen_mean"], reverse=True)
    worst = ranked[:worst_n]
    worst_ids = [sid for sid, _ in worst]
    worst_means = [v["scen_mean"] for _, v in worst]

    total_abs_per_out = np.zeros(len(cont_cols))
    for _, v in worst:
        total_abs_per_out += np.asarray(v["abs_sum"], dtype=np.float64)
    grand = float(total_abs_per_out.sum())
    share = (total_abs_per_out / grand) if grand > 0 else np.zeros_like(total_abs_per_out)

    order = np.argsort(share)[::-1]
    contributors = [{
        "name": cont_cols[k],
        "index": int(k),
        "share": float(share[k]),
        "abs_sum": float(total_abs_per_out[k]),
    } for k in order]
    top3_cum = float(share[order[:topk]].sum())
    return {
        "worst_n": len(worst_ids),
        "worst_scenario_ids": [int(s) for s in worst_ids],
        "worst_scenario_means": [float(m) for m in worst_means],
        "worst_scenario_mean_avg": float(np.mean(worst_means)) if worst_means else float("nan"),
        "share_per_out": [float(s) for s in share],
        "contributors_ranked": contributors,
        f"top{topk}_cumulative_share": top3_cum,
        "top_contributor": contributors[0]["name"] if contributors else None,
    }


# ---------------------------------------------------------------------------
# 4. Link importance <-> error (Spearman over the 10 continuous).
# ---------------------------------------------------------------------------


def _ranks_desc(values):
    """Dense competition ranks (1 = largest). Ties share the average rank (needed for
    a correct Spearman). Uses pandas rank on the negated values."""
    return pd.Series(-np.asarray(values, dtype=np.float64)).rank(method="average").to_numpy()


def link_importance_error(imp_result, per_out_mae, worst_share, cont_cols):
    """Rank-compare, over the 10 continuous variables, the input IMPORTANCE (#1,
    restricted to the continuous channels) against the per-output AR error (#2) and
    the worst-scenario share (#3). Uses Spearman correlation (scipy).

    Returns dict with the aligned per-continuous importance/error/share vectors,
    the top variable in each, Spearman(import, output-error) and
    Spearman(import, worst-share), and a short verdict."""
    from scipy.stats import spearmanr

    # continuous-only importance (first 10 features are the continuous, by schema).
    imp_cont = np.array([imp_result["per_feature"][k]["dMAE_overall"]
                         for k in range(NUM_CONTINUOUS)], dtype=np.float64)
    err = np.asarray(per_out_mae, dtype=np.float64)
    share = np.asarray(worst_share, dtype=np.float64)

    rho_ie, p_ie = spearmanr(imp_cont, err)
    rho_is, p_is = spearmanr(imp_cont, share)

    top_import = cont_cols[int(np.argmax(imp_cont))]
    top_error = cont_cols[int(np.argmax(err))]
    top_share = cont_cols[int(np.argmax(share))]

    imp_rank = _ranks_desc(imp_cont)
    err_rank = _ranks_desc(err)
    share_rank = _ranks_desc(share)

    top_import_idx = int(np.argmax(imp_cont))
    top_import_err_rank = float(err_rank[top_import_idx])
    top_import_share_rank = float(share_rank[top_import_idx])

    aligned = (top_import == top_error) and (top_import == top_share)
    if aligned:
        verdict = (f"ALIGNED: the most important input ({top_import}) is also the "
                   f"hardest output and the top worst-scenario driver.")
    else:
        parts = []
        parts.append(f"most important input = {top_import}")
        parts.append(f"hardest output = {top_error}"
                     + ("" if top_import == top_error
                        else f" (import var is #{int(top_import_err_rank)} hardest)"))
        parts.append(f"top worst-scenario driver = {top_share}"
                     + ("" if top_import == top_share
                        else f" (import var carries share-rank #{int(top_import_share_rank)})"))
        verdict = "DIVERGENT: " + "; ".join(parts) + "."

    return {
        "continuous_vars": list(cont_cols),
        "importance_continuous": [float(v) for v in imp_cont],
        "output_error_mae": [float(v) for v in err],
        "worst_scenario_share": [float(v) for v in share],
        "importance_rank": [float(v) for v in imp_rank],
        "output_error_rank": [float(v) for v in err_rank],
        "worst_scenario_share_rank": [float(v) for v in share_rank],
        "top_important_input": top_import,
        "top_hardest_output": top_error,
        "top_worst_scenario_driver": top_share,
        "top_important_input_output_error_rank": top_import_err_rank,
        "top_important_input_worst_share_rank": top_import_share_rank,
        "spearman_import_vs_output_error": float(rho_ie),
        "spearman_import_vs_output_error_p": float(p_ie),
        "spearman_import_vs_worst_share": float(rho_is),
        "spearman_import_vs_worst_share_p": float(p_is),
        "top_import_is_top_output": bool(top_import == top_error),
        "top_import_is_top_worst_driver": bool(top_import == top_share),
        "all_aligned": bool(aligned),
        "verdict": verdict,
    }


def ranked_table(items, key, name_key="name"):
    """Sort a list of dicts by `key` descending, return [(name, value, rank), ...]."""
    order = sorted(items, key=lambda d: d[key], reverse=True)
    return [{"rank": i + 1, "name": d[name_key], "value": float(d[key])}
            for i, d in enumerate(order)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    args = ap.parse_args()

    cfg = utils.load_config("error_mlp_accident")
    cells = cfg["cells"]
    assert args.cell in cells, f"--cell {args.cell} not in {list(cells)}"
    cell = cells[args.cell]
    cfg_tr = cfg["training"]
    cfg_data = cfg["data"]
    device = torch.device(cfg_tr["device"] if torch.cuda.is_available() else "cpu")

    num_workers = int(cfg_tr["num_workers"])
    collect_batch = int(cfg_tr.get("collect_batch", 2048))
    seq_len = int(cfg_data["seq_len"])
    pred_len = int(cfg_data["pred_len"])
    cache_dir = cfg_data.get("cache_dir")
    nc = int(cfg_data["num_continuous"])
    assert nc == NUM_CONTINUOUS

    run_dir = cell["run_dir"]
    model, input_size = load_frozen_backbone_acc(run_dir, device)
    assert_frozen(model)   # NO retraining: hard-assert the backbone is frozen.

    feature_cols, schema_nc, num_controls = infer_schema_from_csv(cell["test_csv"])
    assert schema_nc == nc and len(feature_cols) == input_size, (
        f"schema mismatch: nc={schema_nc}/{nc} in={len(feature_cols)}/{input_size}")
    cont_cols = list(CONTINUOUS_COLS)
    control_cols = feature_cols[nc:]
    print(f"[{args.cell}] input_size={input_size} num_controls={num_controls} "
          f"controls={control_cols}")

    test_ds = AccidentWindowDataset(cell["test_csv"], seq_len=seq_len,
                                    pred_len=pred_len, cache_dir=cache_dir)
    assert test_ds.num_controls == num_controls and test_ds.input_size == input_size

    # ---- 1. permutation importance (teacher forcing next-step) ----
    print(f"[{args.cell}] collecting teacher-forcing baseline...")
    past, cont_y, base_pred = _collect_teacher_forcing(
        model, test_ds, device, collect_batch, num_workers, nc)
    print(f"[{args.cell}] TF windows N={past.shape[0]} seq={past.shape[1]} "
          f"input={past.shape[2]}")
    imp = permutation_importance(model, past, cont_y, base_pred, feature_cols,
                                 device, collect_batch)
    imp_table = ranked_table(imp["per_feature"], "dMAE_overall")
    print(f"[{args.cell}][import] baseline TF MAE={imp['baseline_mae']:.6f}")
    for r in imp_table[:5]:
        print(f"    #{r['rank']} {r['name']:<24} dMAE={r['value']:+.6f}")

    # ---- 2. baseline uncorrected AR rollout -> per-output error ----
    print(f"[{args.cell}] baseline AR rollout (beta=0)...")
    pd0, td0 = autoregressive_corrected_batched_acc(
        model, None, 0.0, test_ds, num_controls, float(cell["step_norm_const"]),
        device=device, num_continuous=nc, collect_batch=collect_batch,
        num_workers=num_workers)
    per_out_mae, per_out_p99, per_scen = per_output_ar_error(pd0, td0, nc)
    out_items = [{"name": cont_cols[k], "mae": float(per_out_mae[k]),
                  "p99": float(per_out_p99[k])} for k in range(nc)]
    out_table_mae = ranked_table(out_items, "mae")
    out_table_p99 = ranked_table(out_items, "p99")
    print(f"[{args.cell}][output] hardest by MAE:")
    for r in out_table_mae[:3]:
        print(f"    #{r['rank']} {r['name']:<24} MAE={r['value']:.6f}")

    # ---- 3. worst-scenario attribution ----
    attribution = worst_scenario_attribution(per_scen, cont_cols)
    print(f"[{args.cell}][worst-{attribution['worst_n']}] top contributor="
          f"{attribution['top_contributor']} "
          f"top{TOPK_CONTRIB}_cum_share="
          f"{attribution[f'top{TOPK_CONTRIB}_cumulative_share']:.3f}")

    # ---- 4. link importance <-> error ----
    link = link_importance_error(imp, per_out_mae, attribution["share_per_out"],
                                 cont_cols)
    print(f"[{args.cell}][link] import<->output rho="
          f"{link['spearman_import_vs_output_error']:+.3f} "
          f"import<->worst-share rho={link['spearman_import_vs_worst_share']:+.3f}")
    print(f"[{args.cell}][link] {link['verdict']}")

    result = {
        "cell": args.cell,
        "seq_len": seq_len,
        "num_controls": num_controls,
        "input_size": input_size,
        "n_test_windows": int(past.shape[0]),
        "n_scenarios": len(per_scen),
        "continuous_cols": cont_cols,
        "control_cols": control_cols,
        "n_shuffles": N_SHUFFLES,
        "perm_seed": PERM_SEED,
        "step_norm_const": float(cell["step_norm_const"]),
        "importance": {
            "baseline_tf_mae": imp["baseline_mae"],
            "baseline_tf_mae_per_out": imp["baseline_mae_per_out"],
            "per_feature": imp["per_feature"],
            "ranked_by_dMAE_overall": imp_table,
        },
        "output_error": {
            "per_out_mae": [float(v) for v in per_out_mae],
            "per_out_p99": [float(v) for v in per_out_p99],
            "ranked_by_mae": out_table_mae,
            "ranked_by_p99": out_table_p99,
        },
        "worst_scenario_attribution": attribution,
        "link_importance_error": link,
    }

    out_dir = os.path.join(cfg["out_root"], args.cell)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "variable_analysis.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{args.cell}][done] wrote {out_path}")


if __name__ == "__main__":
    main()
