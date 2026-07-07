"""Train a single (interval, seed) model for the layer8/seq3 sweep.

Usage (run from the src/ directory so `import utils` etc. resolve):
    NONINTERACTIVE=1 python experiments/sweep_train.py --interval 60min --seed 42

Resumable: if a best checkpoint already exists for this run, it is skipped.
Model config is fixed in sweep_config.py (d_model 64, head 4, layer 8, dropout 0.1, k=3).
"""
import os
import sys
import argparse
import glob
import yaml
import logging

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")  # real FP32 matmuls

# make src/ importable regardless of cwd
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

import utils
from model_selector import ModelSelector
import experiments.sweep_config as C

logging.basicConfig(level=logging.INFO)


def best_ckpt_path(ckpt_dir):
    """Return the best (non-last) checkpoint path if present, else None."""
    if not os.path.isdir(ckpt_dir):
        return None
    cands = [p for p in glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
             if os.path.basename(p) != "last.ckpt"]
    if not cands:
        return None
    # filename pattern: epoch=..-val_loss=..-step=..ckpt -> pick lowest val_loss
    def val_of(p):
        import re
        m = re.search(r"val_loss=([0-9.]+)", os.path.basename(p))
        return float(m.group(1)) if m else float("inf")
    return min(cands, key=val_of)


def train_one(interval, k, seed, max_epochs=None, device_id=0, num_workers=4):
    train_csv, _ = C.abs_paths(interval)
    logs_dir = C.run_dir(interval, k, seed)
    ckpt_dir = os.path.join(logs_dir, "checkpoints")
    os.makedirs(logs_dir, exist_ok=True)

    existing = best_ckpt_path(ckpt_dir)
    if existing is not None:
        logging.info(f"[skip] {interval} seq{k} seed{seed}: checkpoint exists -> {existing}")
        return existing

    seed_everything(seed, workers=True)

    # --- model (fixed config) ---
    base_model, lit_model = ModelSelector(
        "transformer_decoder",
        backbone_kwargs=dict(C.BACKBONE_KWARGS),
        lightning_kwargs=dict(C.LIGHTNING_KWARGS),
    )

    # --- data ---
    dataset = utils.get_dataset(train_csv, k, C.PRED_LEN, C.PREDICTION_TYPE)
    n_train = int(C.TRAIN_SPLIT * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    logging.info(f"{interval} seq{k} seed{seed}: train={len(train_ds)} val={len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True,
                          num_workers=num_workers, pin_memory=True,
                          persistent_workers=num_workers > 0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=C.BATCH_SIZE, shuffle=False,
                        num_workers=num_workers, pin_memory=True,
                        persistent_workers=num_workers > 0)

    # --- loggers/callbacks (csv_logs/version_0/metrics.csv layout for eval) ---
    csv_logger = CSVLogger(save_dir=logs_dir, name="csv_logs")
    tb_logger = TensorBoardLogger(save_dir=logs_dir, name="tb_logs")
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir, monitor="val_loss", mode="min",
        save_top_k=1, save_last=True,
        filename="{epoch}-{val_loss:.8f}-{step}",
    )
    es_cb = EarlyStopping(monitor="val_loss", mode="min",
                          patience=C.EARLY_STOP_PATIENCE, verbose=True)

    # --- config snapshot ---
    snapshot = {
        "model": {"name": "transformer_decoder",
                  "backbone_kwargs": dict(C.BACKBONE_KWARGS),
                  "lightning_kwargs": dict(C.LIGHTNING_KWARGS)},
        "data": {"data_path": train_csv,
                 "test_data_path": C.abs_paths(interval)[1],
                 "sequence_length": k, "prediction_length": C.PRED_LEN,
                 "batch_size": C.BATCH_SIZE, "prediction_type": C.PREDICTION_TYPE},
        "training": {"epochs": max_epochs or C.MAX_EPOCHS, "seed": seed,
                     "train_split": C.TRAIN_SPLIT,
                     "early_stop_patience": C.EARLY_STOP_PATIENCE},
        "sweep": {"interval": interval, "seq_len": k, "seed": seed},
    }
    with open(os.path.join(logs_dir, "config_used.yaml"), "w") as f:
        yaml.safe_dump(snapshot, f)

    trainer = Trainer(
        accelerator="gpu", devices=[device_id], precision="32-true",
        max_epochs=max_epochs or C.MAX_EPOCHS,
        logger=[csv_logger, tb_logger],
        callbacks=[ckpt_cb, es_cb],
        default_root_dir=logs_dir, log_every_n_steps=10,
        enable_progress_bar=True,
    )
    trainer.fit(lit_model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    best = best_ckpt_path(ckpt_dir) or ckpt_cb.best_model_path
    logging.info(f"[done] {interval} seed{seed}: best={best}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", required=True, choices=C.INTERVALS)
    ap.add_argument("--k", type=int, required=True, choices=C.SEQ_LENS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()
    os.environ.setdefault("NONINTERACTIVE", "1")
    train_one(args.interval, args.k, args.seed, max_epochs=args.epochs,
              device_id=args.device, num_workers=args.num_workers)


if __name__ == "__main__":
    main()
