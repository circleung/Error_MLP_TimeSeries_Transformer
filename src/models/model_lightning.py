import os
import sys
import torch
from torch import nn
from pytorch_lightning import LightningModule


class LitRNNBaseModel(LightningModule):
    def __init__(self, backbone, alpha=0.5, tolerance=0.05, y_type="difference_y_cont"):
        super().__init__()
        self.backbone = backbone
        self.criterion = nn.MSELoss()
        self.val_losses = []
        self.alpha = alpha
        self.tolerance = tolerance
        self.y_type = y_type
        print(f"Using {self.y_type} as target variable for training and validation.")
        # Pause for user confirmation only in an interactive terminal.
        # Skip when running non-interactively (sweeps/background) or when
        # NONINTERACTIVE=1 is set, so automated runs don't block on stdin.
        if os.environ.get("NONINTERACTIVE", "") != "1" and sys.stdin and sys.stdin.isatty():
            _ = input("Press Enter to continue...")

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x = batch["past_values"].float()
        cont_y = batch[self.y_type].float().squeeze(1)
        # y is [batch, 1, input_size], change to [batch, input_size]
        cont_y_hat = self(x)
        loss = self.criterion(cont_y_hat, cont_y)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["past_values"].float()
        cont_y = batch[self.y_type].float().squeeze(1)
        # y is [batch, 1, input_size], change to [batch, input_size]
        cont_y_hat = self(x)
        loss = self.criterion(cont_y_hat, cont_y)
        self.val_losses.append(loss.item())
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log(
            "Accuracy",
            self.get_accuracy(cont_y, cont_y_hat),
            prog_bar=True,
            logger=True,
        )

        return loss

    def get_accuracy(self, y, y_hat):
        with torch.no_grad():
            # Continuous accuracy
            cont_diff = (y_hat - y).abs()
            cont_accuracy = (cont_diff < self.tolerance).float().mean()

            return cont_accuracy

    def on_validation_epoch_end(self):
        val_losses_tensor = torch.tensor(self.val_losses)
        self.log("val_loss_mean", val_losses_tensor.mean(), prog_bar=True)
        self.val_losses = []  # Reset for the next epoch

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=2, gamma=0.1
        )  # weirdly, step size is epochs.
        return [optimizer], [scheduler]
