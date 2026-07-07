import pytorch_lightning as pl
import torch
import torch.nn as nn


class TimeSeriesLightningModule(pl.LightningModule):
    def __init__(self, base_model, learning_rate=1e-3):
        """
        Args:
            base_model: Your initialized SimpleTimeSeriesTransformer.
            learning_rate: Learning rate for optimizer.
        """
        super().__init__()
        self.save_hyperparameters(
            ignore=["base_model"]
        )  # Don't serialize the model weights themselves
        self.model = base_model
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate
        # If you have both continuous and binary outputs, add a BCE loss and sum/weight

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch["past_values"]  # [B, seq_len, num_features]
        y = batch["continuous_y"]  # [B, output_dim]
        y_hat = self(x)  # [B, 1, output_dim]
        y_hat = y_hat.squeeze(1)  # [B, output_dim]
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["past_values"]
        y = batch["continuous_y"]
        y_hat = self(x)
        y_hat = y_hat.squeeze(1)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch["past_values"]
        y = batch["continuous_y"]
        y_hat = self(x)
        y_hat = y_hat.squeeze(1)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
