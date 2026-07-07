"""Evaluate one trained run with BOTH teacher-forcing (Table 7) and
autoregressive / rollout (Table 8), using the official batched passes from
predict_batched.py (same TransformerDataset + same metric definition as
predict.regressive/autoregressive_predictions_absolute).

For a run dir (training_logs_layer8/<interval>/seq<k>/seed<seed>):
  - load config_used.yaml + best checkpoint
  - build the interval's test dataset (TransformerDataset, absolute)
  - TF  : predict_batched.regressive_predictions_absolute_batched
  - AR  : predict_batched.autoregressive_predictions_absolute_batched
  - metrics via compute_micro_macro (micro = pooled over steps,
    macro = mean of per-scenario means = Table 7/8 convention)
  - writes <run_dir>/eval_metrics.json and per-mode eval_summary_{tf,ar}.txt

Usage (from src/):
    NONINTERACTIVE=1 python experiments/sweep_eval.py --interval 60min --k 3 --seed 42
"""
import os
import sys
import re
import glob
import json
import argparse

import torch

# make src/ importable
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import yaml
import utils
from model_selector import ModelSelector
from predict_batched import (
    regressive_predictions_absolute_batched,
    autoregressive_predictions_absolute_batched,
    compute_micro_macro,
)
import experiments.sweep_config as C

os.environ.setdefault("NONINTERACTIVE", "1")


def find_best_ckpt(run_dir):
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    cands = [p for p in glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
             if os.path.basename(p) != "last.ckpt"]
    if not cands:
        cands = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not cands:
        return None

    def val_of(p):
        m = re.search(r"val_loss=([0-9.]+)", os.path.basename(p))
        return float(m.group(1)) if m else float("inf")
    return min(cands, key=val_of)


def load_model(run_dir, device):
    with open(os.path.join(run_dir, "config_used.yaml")) as f:
        cfg = yaml.safe_load(f)
    backbone_kwargs = cfg["model"]["backbone_kwargs"]
    lightning_kwargs = cfg["model"].get("lightning_kwargs", {})
    _, lit_model = ModelSelector(
        "transformer_decoder", backbone_kwargs=backbone_kwargs,
        lightning_kwargs=lightning_kwargs,
    )
    ckpt_path = find_best_ckpt(run_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint in {run_dir}")
    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    lit_model.load_state_dict(state)
    model = lit_model.backbone.to(device=device, dtype=torch.float32).eval()
    seq_len = int(cfg["data"]["sequence_length"])
    return model, seq_len, ckpt_path, cfg


def _write_summary_txt(path, mode, interval, k, seed, summary, ckpt):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"=== {mode} evaluation summary ===\n")
        f.write(f"interval={interval}  k={k}  seed={seed}\n")
        f.write(f"checkpoint={ckpt}\n")
        f.write(f"scenarios used: {summary['n_scen']}\n")
        f.write(f"MICRO  MAE={summary['micro_mae']:.8f}  RMSE={summary['micro_rmse']:.8f}\n")
        f.write(f"MACRO  MAE={summary['macro_mae']:.8f}  RMSE={summary['macro_rmse']:.8f}\n")


def evaluate_run(interval, k, seed, device="cuda:0", num_workers=4):
    run_dir = C.run_dir(interval, k, seed)
    _, test_csv = C.abs_paths(interval)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model, seq_len, ckpt_path, cfg = load_model(run_dir, device)
    test_ds = utils.get_dataset(test_csv, seq_len, C.PRED_LEN, C.PREDICTION_TYPE)

    # Teacher-forcing (Table 7)
    tf_pred, tf_true = regressive_predictions_absolute_batched(
        model, test_ds, device=device, num_workers=num_workers)
    tf_summary = compute_micro_macro(tf_pred, tf_true)

    # Autoregressive / rollout (Table 8)
    ar_pred, ar_true = autoregressive_predictions_absolute_batched(
        model, test_ds, device=device, num_continuous=C.NUM_CONTINUOUS,
        num_workers=num_workers)
    ar_summary = compute_micro_macro(ar_pred, ar_true)

    result = dict(
        interval=interval, seq_len=seq_len, seed=seed,
        n_test_samples=len(test_ds), checkpoint=os.path.basename(ckpt_path),
        teacher_forcing=tf_summary, autoregressive=ar_summary,
    )
    with open(os.path.join(run_dir, "eval_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    _write_summary_txt(os.path.join(run_dir, "eval_summary_tf.txt"),
                       "Teacher-forcing", interval, k, seed, tf_summary,
                       os.path.basename(ckpt_path))
    _write_summary_txt(os.path.join(run_dir, "eval_summary_ar.txt"),
                       "Autoregressive (rollout)", interval, k, seed, ar_summary,
                       os.path.basename(ckpt_path))

    print(f"[eval] {interval} seq{k} seed{seed}")
    print(f"  TF : MAE macro={tf_summary['macro_mae']:.6f} micro={tf_summary['micro_mae']:.6f}"
          f" | RMSE macro={tf_summary['macro_rmse']:.6f} micro={tf_summary['micro_rmse']:.6f}")
    print(f"  AR : MAE macro={ar_summary['macro_mae']:.6f} micro={ar_summary['micro_mae']:.6f}"
          f" | RMSE macro={ar_summary['macro_rmse']:.6f} micro={ar_summary['micro_rmse']:.6f}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", required=True, choices=C.INTERVALS)
    ap.add_argument("--k", type=int, required=True, choices=C.SEQ_LENS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()
    evaluate_run(args.interval, args.k, args.seed, device=args.device,
                 num_workers=args.num_workers)


if __name__ == "__main__":
    main()
