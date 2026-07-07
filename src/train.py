import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")  # forces real FP32 matmuls

from model_selector import ModelSelector
import utils
import logging
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import os
import yaml

logging.basicConfig(level=logging.INFO)
DEVICE_ID = 7


def save_config_snapshot(config, logs_dir):
    """Save the used config for reproducibility."""
    os.makedirs(logs_dir, exist_ok=True)
    with open(os.path.join(logs_dir, "config_used.yaml"), "w") as f:
        yaml.safe_dump(config, f)


def main():
    selected_model = (
        "transformer_decoder"  # Only place to change for hyperparameter tuning
    )
    config = utils.load_config(selected_model)
    logging.info(f"Loaded config for {selected_model}: {config}")
    dataset_type = config["data"]["prediction_type"]

    model_name = config["model"]["name"]
    backbone_kwargs = config["model"]["backbone_kwargs"]
    lightning_kwargs = config["model"].get("lightning_kwargs", {})
    base_model, lit_model = ModelSelector(
        model_name, backbone_kwargs=backbone_kwargs, lightning_kwargs=lightning_kwargs
    )

    dataset = utils.get_dataset(
        config["data"]["data_path"],
        config["data"]["sequence_length"],
        config["data"]["prediction_length"],
    )

    train_size = int(config["training"].get("train_split", 0.9) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    logs_dir = f"training_logs/{model_name}"
    os.makedirs(logs_dir, exist_ok=True)

    # --- Logging, Checkpointing, Early Stopping ---
    tb_logger = TensorBoardLogger(save_dir=logs_dir, name="tb_logs")
    csv_logger = CSVLogger(save_dir=logs_dir, name="csv_logs")
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(
            logs_dir, f"{selected_model}_wonung_checkpoints_{dataset_type}"
        ),
        monitor="val_loss",  # or "val_loss_mean" or your metric
        mode="min",
        save_top_k=-1,  # Save best 2 models
        save_last=True,
        filename="{epoch}-{val_loss:.8f}-{step}",
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=10,
        verbose=True,
    )

    # --- Save config snapshot ---
    save_config_snapshot(config, logs_dir)

    # --- Trainer ---
    trainer = Trainer(
        accelerator="gpu",
        precision="32-true",
        devices=[DEVICE_ID],
        max_epochs=config["training"]["epochs"],
        logger=[tb_logger, csv_logger],
        callbacks=[checkpoint_callback, early_stopping_callback],
        default_root_dir=logs_dir,
        log_every_n_steps=10,
    )

    # --- Fit ---
    trainer.fit(
        lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

    logging.info(f"Training complete! Check logs and checkpoints in: {logs_dir}")


if __name__ == "__main__":
    seed_everything(42, workers=True)  # For reproducibility
    main()
