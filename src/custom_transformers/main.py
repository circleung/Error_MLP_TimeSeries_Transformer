import dataset
from torch.utils.data import DataLoader, random_split
import torch
from model import SimpleTimeSeriesTransformer
from model_lightning import TimeSeriesLightningModule
from model_decoder import SimpleDecoderOnlyTransformer
from decoder_lightning import TimeSeriesLightningDecoderModule
import pytorch_lightning as pl
import logging

logging.basicConfig(level=logging.INFO)

A_100_DEVICE_ID = 2


def get_dataset(data_path, sequence_length, pred_length):
    return dataset.BaseDataset(
        [data_path], seq_len=sequence_length, pred_len=pred_length
    )


def get_dataloader(dataset, batch_size, shuffle, num_workers=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=num_workers,
    )


class Config:
    """
    Maximum of lags + n_features should be equal to data coming out from Dataloader
    """

    context_length = 20
    n_time_features = 1
    prediction_length = 1
    input_size = 7
    output_size = 7


if __name__ == "__main__":
    train_data_path = "/media/8TB_hardisk/sangam/timeseries_forecasting/train_data.csv"
    base_dataset = get_dataset(train_data_path, 20, 1)
    test_data_path = "/media/8TB_hardisk/sangam/timeseries_forecasting/test_data.csv"
    base_test_dataset = get_dataset(test_data_path, 20, 1)
    transformer_dataset = dataset.TransformerDataset(base_dataset)
    test_dataset = dataset.TransformerDataset(base_test_dataset)
    train_size = int(0.9 * len(transformer_dataset))
    val_size = len(transformer_dataset) - train_size
    train_dataset, val_dataset = random_split(
        transformer_dataset, [train_size, val_size]
    )
    logging.info(f"Length of training set: {len(train_dataset)}")
    logging.info(f"Length of validation set: {len(val_dataset)}")
    train_dataloader = get_dataloader(
        train_dataset, batch_size=32, shuffle=True, num_workers=20
    )
    val_dataloader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = get_dataloader(test_dataset, batch_size=32, shuffle=False)
    # model = SimpleTimeSeriesTransformer(
    #     num_features=Config.input_size,
    #     d_model=32,
    #     nhead=4,
    #     num_layers=2,
    #     dropout=0.1,
    #     pred_len=Config.prediction_length,
    #     output_dim=Config.output_size,
    # )
    model = SimpleDecoderOnlyTransformer(
        num_features=Config.input_size,
        d_model=32,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        pred_len=Config.prediction_length,
    )
    lit_module = TimeSeriesLightningDecoderModule(base_model=model, learning_rate=1e-3)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[A_100_DEVICE_ID],
        max_epochs=20,
        default_root_dir="outputs_decoder",
        log_every_n_steps=10,
        callbacks=[pl.callbacks.ModelCheckpoint(monitor="val_loss")],
    )
    trainer.fit(lit_module, train_dataloader, val_dataloader)
    results = trainer.test(lit_module, dataloaders=test_dataloader)
    print(results)

    # for batch in test_dataloader:
    #     past_values = batch["past_values"]
    #     y = batch["continuous_y"]
    #     print("Batch shape:", past_values.shape, y.shape)
    #     out = model(past_values)
    #     print(f"Output shape: {out.shape}")
    #     break
