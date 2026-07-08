"""Per-accident-type AR-ROLLOUT permutation importance (NO retraining; frozen backbone).

ADDITIVE, read-only-on-the-backbone counterpart to experiments/variable_analysis_acc.py.
That file computed permutation importance in TEACHER FORCING (single forward, shuffle
each input channel across the test windows -> increase in next-step MAE). THIS file
computes permutation importance in AR ROLLOUT: it shuffles an input channel across the
S rollout scenarios and *keeps it shuffled through the whole lockstep rollout*, so the
corruption propagates through the model's own feedback (continuous channels are the
fed-back predictions; controls are the injected known-future). The question it answers
is whether feedback + error compounding changes WHICH variables matter vs the single-step
TF sensitivity, and whether the TF importance<->output-error DIVERGENCE survives under AR.

It REUSES (imports, does not modify):
  * load_frozen_backbone_acc                -- the per-cell frozen backbone loader,
  * _collect_pass / _pack                   -- the byte-faithful Pass-1 collect + dense
                                               [S, maxL, .] packing used by the baseline
                                               autoregressive_corrected_batched_acc,
  * AccidentWindowDataset / infer_schema_from_csv / CONTINUOUS_COLS / NUM_CONTINUOUS,
  * the committed TF variable_analysis.json (for the TF-vs-AR comparison block).

The backbone is hard-asserted frozen after load (assert_frozen). NOTHING is trained.

Faithful AR-rollout permutation importance (per the task design):
  Batched lockstep rollout over the cell's TEST scenarios (SAME collect as the baseline),
  optionally restricted to a FIXED subsample (first SUBSAMPLE scenario ids, sorted
  ascending, so the set is deterministic regardless of dataset iteration order). For each
  INPUT channel c (the 10 continuous AND the controls):
    * draw ONE permutation pi of the S active scenarios (per shuffle-seed, FIXED across
      all rollout steps for that run);
    * run the full lockstep rollout, but at EVERY step, before calling the model, copy
      the current input `window` [S, seq, input] to `window_c` and set
          window_c[:, :, c] = window[pi, :, c]
      (scenario i's whole channel-c sub-column replaced by scenario pi(i)'s), then
          out = model(window_c)
      and feed THAT out back:  next_row = cat([out, BY[:,t]]);  window = roll.
      So the corruption compounds through the rollout (for a control channel it corrupts
      the injected known-future control; for a continuous channel it corrupts the
      fed-back trajectory -- symmetric with the TF "shuffle across samples").
    * metric  dMAE_overall_AR = MAE_rollout(corrupted_c) - MAE_rollout(baseline)
      where MAE_rollout = mean over (scenario, step, 10 outputs) of |pred - true| (the
      SAME per-step MAE unit as compute_micro_macro / the TF baseline_mae). Also the
      per-output  dMAE_per_out_AR[k].
  Averaged over N_SHUFFLES fixed seeds (default 2; AR is expensive, 2 is enough for a
  stable ranking -- noted in the output). Channels ranked by dMAE_overall_AR.

Writes <out_root>/<cell>/ar_permutation_importance.json with:
  * baseline rollout MAE (on the subsample) + per-output baseline MAE,
  * per-channel dMAE_overall_AR (+std over seeds) + dMAE_per_out_AR, ranked table,
  * config (n_shuffles, seeds, subsample size, subsample scenario ids, approx GPU-time),
  * a TF-vs-AR COMPARISON block: TF dMAE_overall for the same channels, Spearman(TF,AR)
    over ALL input channels AND over the 10 continuous only, top-3 set overlap, and
    per-channel rank shifts; PLUS the recomputed importance<->output-error link (Spearman
    of AR-importance-over-10-continuous vs the committed output_error.per_out_mae) to see
    whether the TF DIVERGENT conclusion still holds under AR importance.

Usage (from src/):
    NONINTERACTIVE=1 python experiments/ar_permutation_importance_acc.py --cell SBO
    NONINTERACTIVE=1 python experiments/ar_permutation_importance_acc.py --cell SBO --subsample 800
"""
import os
import sys
import json
import time
import argparse
from collections import defaultdict

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

os.environ.setdefault("NONINTERACTIVE", "1")

import numpy as np
import pandas as pd
import torch
import tqdm

import utils
from accident_dataset import (
    AccidentWindowDataset,
    infer_schema_from_csv,
    CONTINUOUS_COLS,
    NUM_CONTINUOUS,
)
from error_rollout_acc import (
    load_frozen_backbone_acc,
    _collect_pass,
    _pack,
)

N_SHUFFLES = 2               # AR is expensive; 2 fixed seeds is enough for a stable rank.
PERM_SEEDS = (1234, 5678)    # fixed shuffle seeds (one per shuffle), averaged.
SUBSAMPLE = 800              # fixed representative scenario subsample (first N ids asc).


def assert_frozen(model):
    """Hard-assert the backbone carries no trainable parameters (NO retraining)."""
    n_trainable = sum(int(p.requires_grad) for p in model.parameters())
    assert n_trainable == 0, f"backbone is NOT frozen: {n_trainable} trainable params"
    assert not model.training, "backbone must be in eval() mode"


# ---------------------------------------------------------------------------
# AR-rollout MAE (baseline + per-channel permuted), lockstep, batched.
# ---------------------------------------------------------------------------


def _select_subsample(order, subsample):
    """Deterministic FIXED subsample: sort the collected scenario ids ascending and keep
    the first `subsample` (or all if subsample is None/0/>=S). Returns the sorted list of
    kept scenario ids AND the row-indices into the `order`-aligned packed arrays that
    select them (also in ascending-id order, so the packed subarrays are reproducible)."""
    order_arr = np.asarray([int(s) for s in order], dtype=np.int64)
    sorted_ids = np.sort(order_arr)
    if subsample and 0 < int(subsample) < len(sorted_ids):
        kept_ids = sorted_ids[: int(subsample)]
    else:
        kept_ids = sorted_ids
    kept_set = set(int(s) for s in kept_ids.tolist())
    # row indices into the packed [S, ...] arrays (order-aligned), taken in ascending id.
    id_to_row = {int(s): i for i, s in enumerate(order)}
    rows = np.asarray([id_to_row[int(s)] for s in kept_ids], dtype=np.int64)
    return kept_ids, rows


@torch.inference_mode()
def _rollout_mae(model, window0, UY_t, CY_t, lengths_t, num_continuous,
                 perm=None, chan=None):
    """One lockstep AR rollout over S scenarios; return (mae_overall, mae_per_out).

    window0 [S, seq, input] (a fresh clone is made internally so the caller's tensor is
    untouched), UY_t [S, maxL, num_controls] = truth controls fed each step, CY_t
    [S, maxL, 10] = truth continuous targets, lengths_t [S] active length per scenario.

    If (perm, chan) are given, at EVERY step channel `chan` of the current window is
    replaced across scenarios by the FIXED permutation `perm` (perm is a LongTensor of
    scenario row-indices) BEFORE the forward -- and the (uncorrupted-fed-back) prediction
    is what rolls forward, so the corruption compounds through the feedback loop.

    MAE = mean over ACTIVE (scenario, step, 10 outputs) of |pred - true| (same unit as
    compute_micro_macro's per-step MAE / the TF baseline_mae)."""
    window = window0.clone()
    S = window.shape[0]
    maxL = UY_t.shape[1]
    device = window.device

    abs_sum = torch.zeros(num_continuous, device=device, dtype=torch.float64)
    n_active_total = 0
    for t in range(maxL):
        if perm is not None:
            win_in = window.clone()
            win_in[:, :, chan] = window[perm, :, chan]
        else:
            win_in = window
        out = model(win_in)                                    # [S, 10]
        active = (t < lengths_t)                               # [S] bool
        if active.any():
            diff = (CY_t[:, t, :] - out).abs()                 # [S, 10]
            abs_sum += diff[active].sum(dim=0).to(torch.float64)
            n_active_total += int(active.sum().item())
        # feed the (corruption-affected) prediction back; controls are truth.
        next_row = torch.cat([out, UY_t[:, t, :]], dim=1)      # [S, input]
        window = torch.cat([window[:, 1:, :], next_row[:, None, :]], dim=1)

    per_out = (abs_sum / max(1, n_active_total)).detach().cpu().numpy().astype(np.float64)
    overall = float(per_out.mean())
    return overall, per_out


def ar_permutation_importance(model, window0, UY_t, CY_t, lengths_t, feature_cols,
                              num_continuous, seeds=PERM_SEEDS):
    """Per-INPUT-channel AR-rollout permutation importance.

    Runs ONE baseline rollout (unperturbed) then, for each input channel c and each fixed
    seed, a full corrupted rollout (channel c shuffled across the S scenarios, held fixed
    across steps). dMAE = corrupted MAE - baseline MAE (overall + per-output), averaged
    over seeds. Returns dict mirroring the TF permutation_importance shape."""
    S = window0.shape[0]
    input_size = window0.shape[2]
    assert input_size == len(feature_cols), (
        f"input_size {input_size} != len(feature_cols) {len(feature_cols)}")
    device = window0.device

    base_overall, base_per_out = _rollout_mae(
        model, window0, UY_t, CY_t, lengths_t, num_continuous)
    print(f"    baseline AR rollout MAE={base_overall:.6f} (S={S})")

    # Pre-draw one scenario-permutation per (channel, seed); channel-c is shuffled by the
    # SAME permutation across all its seq positions (a whole channel sub-column swap).
    perms = {}
    for si, seed in enumerate(seeds):
        rng = np.random.RandomState(seed)
        for c in range(input_size):
            perms[(c, si)] = torch.from_numpy(
                rng.permutation(S).astype(np.int64)).to(device)

    per_feature = []
    for c in range(input_size):
        d_overall_runs, d_per_out_runs = [], []
        for si in range(len(seeds)):
            ov, po = _rollout_mae(
                model, window0, UY_t, CY_t, lengths_t, num_continuous,
                perm=perms[(c, si)], chan=c)
            d_overall_runs.append(ov - base_overall)
            d_per_out_runs.append(po - base_per_out)
        d_per_out = np.mean(d_per_out_runs, axis=0)
        per_feature.append({
            "index": c,
            "name": feature_cols[c],
            "is_control": c >= num_continuous,
            "dMAE_overall": float(np.mean(d_overall_runs)),
            "dMAE_overall_std": float(np.std(d_overall_runs)),
            "dMAE_per_out": [float(v) for v in d_per_out],
        })
        print(f"    [{c:2d}] {feature_cols[c]:<26} dMAE_AR={np.mean(d_overall_runs):+.6f} "
              f"(std={np.std(d_overall_runs):.2e})")

    return {
        "baseline_rollout_mae": base_overall,
        "baseline_rollout_mae_per_out": [float(v) for v in base_per_out],
        "per_feature": per_feature,
    }


# ---------------------------------------------------------------------------
# Ranking + TF-vs-AR comparison.
# ---------------------------------------------------------------------------


def _ranks_desc(values):
    """Dense (average-tie) competition ranks, 1 = largest. Matches variable_analysis_acc
    so Spearman is computed identically."""
    return pd.Series(-np.asarray(values, dtype=np.float64)).rank(method="average").to_numpy()


def ranked_table(items, key, name_key="name"):
    """Sort a list of dicts by `key` descending -> [{rank,name,value}, ...]."""
    order = sorted(items, key=lambda d: d[key], reverse=True)
    return [{"rank": i + 1, "name": d[name_key], "value": float(d[key])}
            for i, d in enumerate(order)]


def tf_vs_ar_comparison(ar_result, tf_json, feature_cols, cont_cols, num_continuous):
    """Compare AR-rollout importance to the committed TEACHER-FORCING importance.

    Aligns per-channel dMAE_overall by feature NAME (both share feature_cols order).
    Computes Spearman(TF_rank, AR_rank) over ALL input channels and over the 10 continuous
    only, the top-3 set overlap, and per-channel rank shifts (TF_rank - AR_rank). Also
    recomputes the importance<->output-error link under AR importance: Spearman of the
    AR importance restricted to the 10 continuous vs the committed output_error.per_out_mae
    (so we see if the TF DIVERGENT conclusion survives)."""
    from scipy.stats import spearmanr

    tf_feat = {d["name"]: d for d in tf_json["importance"]["per_feature"]}
    ar_feat = {d["name"]: d for d in ar_result["per_feature"]}
    names = list(feature_cols)
    tf_imp = np.array([tf_feat[n]["dMAE_overall"] for n in names], dtype=np.float64)
    ar_imp = np.array([ar_feat[n]["dMAE_overall"] for n in names], dtype=np.float64)

    tf_rank_all = _ranks_desc(tf_imp)
    ar_rank_all = _ranks_desc(ar_imp)
    rho_all, p_all = spearmanr(tf_imp, ar_imp)

    cont_idx = list(range(num_continuous))
    tf_cont = tf_imp[cont_idx]
    ar_cont = ar_imp[cont_idx]
    tf_rank_cont = _ranks_desc(tf_cont)
    ar_rank_cont = _ranks_desc(ar_cont)
    rho_cont, p_cont = spearmanr(tf_cont, ar_cont)

    tf_ranked = ranked_table(tf_json["importance"]["per_feature"], "dMAE_overall")
    ar_ranked = ranked_table(ar_result["per_feature"], "dMAE_overall")
    tf_top3 = [r["name"] for r in tf_ranked[:3]]
    ar_top3 = [r["name"] for r in ar_ranked[:3]]
    top3_overlap = sorted(set(tf_top3) & set(ar_top3))

    per_channel = []
    for i, n in enumerate(names):
        per_channel.append({
            "name": n,
            "index": i,
            "is_control": i >= num_continuous,
            "tf_dMAE": float(tf_imp[i]),
            "ar_dMAE": float(ar_imp[i]),
            "tf_rank": int(round(float(tf_rank_all[i]))),
            "ar_rank": int(round(float(ar_rank_all[i]))),
            "rank_shift_tf_minus_ar": int(round(float(tf_rank_all[i] - ar_rank_all[i]))),
        })
    # biggest movers (by absolute rank shift), most positive shift = climbed under AR.
    movers = sorted(per_channel, key=lambda d: abs(d["rank_shift_tf_minus_ar"]),
                    reverse=True)

    # importance<->output-error link recomputed under AR (10 continuous only).
    per_out_mae = np.asarray(tf_json["output_error"]["per_out_mae"], dtype=np.float64)
    rho_ie_ar, p_ie_ar = spearmanr(ar_cont, per_out_mae)
    top_import_ar = cont_cols[int(np.argmax(ar_cont))]
    top_error = cont_cols[int(np.argmax(per_out_mae))]
    err_rank = _ranks_desc(per_out_mae)
    top_import_ar_err_rank = int(round(float(err_rank[int(np.argmax(ar_cont))])))
    ar_top_is_top_output = bool(top_import_ar == top_error)

    # the committed TF verdict/rho for reference.
    tf_link = tf_json.get("link_importance_error", {})
    tf_rho_ie = tf_link.get("spearman_import_vs_output_error")
    tf_all_aligned = tf_link.get("all_aligned")

    if ar_top_is_top_output:
        ar_link_verdict = (f"ALIGNED-under-AR: most important AR input ({top_import_ar}) "
                           f"is also the hardest output.")
    else:
        ar_link_verdict = (f"DIVERGENT-under-AR: most important AR input = {top_import_ar} "
                           f"but hardest output = {top_error} "
                           f"(AR-import var is #{top_import_ar_err_rank} hardest output).")

    return {
        "feature_cols": names,
        "tf_dMAE_overall": [float(v) for v in tf_imp],
        "ar_dMAE_overall": [float(v) for v in ar_imp],
        "spearman_tf_ar_all_channels": float(rho_all),
        "spearman_tf_ar_all_channels_p": float(p_all),
        "spearman_tf_ar_continuous": float(rho_cont),
        "spearman_tf_ar_continuous_p": float(p_cont),
        "tf_top3": tf_top3,
        "ar_top3": ar_top3,
        "top3_overlap": top3_overlap,
        "n_top3_overlap": len(top3_overlap),
        "per_channel": per_channel,
        "biggest_movers": movers[:5],
        # importance<->output-error link under AR:
        "ar_link": {
            "importance_continuous_ar": [float(v) for v in ar_cont],
            "output_error_mae": [float(v) for v in per_out_mae],
            "top_important_input_ar": top_import_ar,
            "top_hardest_output": top_error,
            "top_important_input_ar_output_error_rank": top_import_ar_err_rank,
            "ar_top_import_is_top_output": ar_top_is_top_output,
            "spearman_ar_import_vs_output_error": float(rho_ie_ar),
            "spearman_ar_import_vs_output_error_p": float(p_ie_ar),
            "verdict": ar_link_verdict,
        },
        # committed TF reference:
        "tf_reference": {
            "spearman_tf_import_vs_output_error": (
                float(tf_rho_ie) if tf_rho_ie is not None else None),
            "tf_all_aligned": tf_all_aligned,
            "tf_verdict": tf_link.get("verdict"),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--subsample", type=int, default=SUBSAMPLE,
                    help="fixed scenario subsample (first N ids asc); 0 = all scenarios")
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

    # ---- Pass-1 collect (SAME as the baseline AR rollout) then fixed subsample. ----
    print(f"[{args.cell}] AR Pass-1 collect...")
    order, init_win, cont, ctrl = _collect_pass(test_ds, collect_batch, num_workers)
    S_full, lengths, maxL, W, CY, UY = _pack(order, init_win, cont, ctrl, nc, num_controls)

    kept_ids, rows = _select_subsample(order, args.subsample)
    S = len(rows)
    subsample_used = (S < S_full)
    print(f"[{args.cell}] scenarios total={S_full} using={S} "
          f"(subsample={'first %d by id asc' % args.subsample if subsample_used else 'ALL'})")

    W_s = W[rows]
    CY_s = CY[rows]
    UY_s = UY[rows]
    lengths_s = lengths[rows]
    maxL_s = int(lengths_s.max())
    # truncate the time axis to the subsample's max active length (all-equal per cell, but
    # this keeps it correct if lengths ever vary).
    CY_s = CY_s[:, :maxL_s, :]
    UY_s = UY_s[:, :maxL_s, :]

    window0 = torch.from_numpy(W_s).to(device)
    UY_t = torch.from_numpy(UY_s).to(device)
    CY_t = torch.from_numpy(CY_s).to(device)
    lengths_t = torch.from_numpy(lengths_s.astype(np.int64)).to(device)

    # ---- AR-rollout permutation importance ----
    print(f"[{args.cell}] AR permutation importance "
          f"(n_shuffles={N_SHUFFLES}, seeds={PERM_SEEDS}, maxL={maxL_s})...")
    t0 = time.time()
    ar = ar_permutation_importance(model, window0, UY_t, CY_t, lengths_t,
                                   feature_cols, nc, seeds=PERM_SEEDS)
    gpu_seconds = time.time() - t0
    ar_table = ranked_table(ar["per_feature"], "dMAE_overall")
    print(f"[{args.cell}][AR-import] baseline rollout MAE={ar['baseline_rollout_mae']:.6f} "
          f"(elapsed {gpu_seconds:.1f}s)")
    for r in ar_table[:5]:
        print(f"    #{r['rank']} {r['name']:<26} dMAE_AR={r['value']:+.6f}")

    # ---- TF-vs-AR comparison (load committed TF json) ----
    tf_path = os.path.join(cfg["out_root"], args.cell, "variable_analysis.json")
    assert os.path.exists(tf_path), (
        f"committed TF importance not found: {tf_path} "
        f"(run experiments/variable_analysis_acc.py --cell {args.cell} first)")
    with open(tf_path) as f:
        tf_json = json.load(f)
    cmp = tf_vs_ar_comparison(ar, tf_json, feature_cols, cont_cols, nc)
    print(f"[{args.cell}][cmp] Spearman(TF,AR) all={cmp['spearman_tf_ar_all_channels']:+.3f} "
          f"cont={cmp['spearman_tf_ar_continuous']:+.3f} "
          f"top3_overlap={cmp['n_top3_overlap']}/3 {cmp['top3_overlap']}")
    print(f"[{args.cell}][cmp] TF top3={cmp['tf_top3']}")
    print(f"[{args.cell}][cmp] AR top3={cmp['ar_top3']}")
    print(f"[{args.cell}][cmp] AR import<->output rho="
          f"{cmp['ar_link']['spearman_ar_import_vs_output_error']:+.3f} | "
          f"{cmp['ar_link']['verdict']}")

    result = {
        "cell": args.cell,
        "method": "AR-rollout permutation importance (frozen backbone, NO retraining)",
        "seq_len": seq_len,
        "num_controls": num_controls,
        "input_size": input_size,
        "continuous_cols": cont_cols,
        "control_cols": control_cols,
        "step_norm_const": float(cell["step_norm_const"]),
        "config": {
            "n_shuffles": N_SHUFFLES,
            "perm_seeds": list(PERM_SEEDS),
            "n_scenarios_total": int(S_full),
            "n_scenarios_used": int(S),
            "subsample": bool(subsample_used),
            "subsample_size": int(args.subsample) if subsample_used else None,
            "subsample_rule": ("first %d scenario ids ascending" % args.subsample
                               if subsample_used else "ALL scenarios"),
            "subsample_scenario_ids": [int(s) for s in kept_ids.tolist()],
            "maxL_rollout": int(maxL_s),
            "approx_gpu_seconds": float(gpu_seconds),
            "metric": ("MAE over (scenario, step, 10 outputs) of |pred-true| in AR "
                       "lockstep rollout; dMAE = corrupted - baseline, avg over seeds"),
        },
        "importance": {
            "baseline_rollout_mae": ar["baseline_rollout_mae"],
            "baseline_rollout_mae_per_out": ar["baseline_rollout_mae_per_out"],
            "per_feature": ar["per_feature"],
            "ranked_by_dMAE_overall_AR": ar_table,
        },
        "tf_vs_ar": cmp,
    }

    out_dir = os.path.join(cfg["out_root"], args.cell)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ar_permutation_importance.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{args.cell}][done] wrote {out_path}")


if __name__ == "__main__":
    main()
