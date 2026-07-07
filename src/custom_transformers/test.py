import dataset
from torch.utils.data import DataLoader, random_split
import torch
from model import SimpleTimeSeriesTransformer
from model_lightning import TimeSeriesLightningModule
import pytorch_lightning as pl
import logging

logging.basicConfig(level=logging.INFO)

A_100_DEVICE_ID = 3


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
    input_size = 17
    output_size = 7


def load_lit_model(base_model, model_path):
    lit_model = TimeSeriesLightningModule(base_model=base_model, learning_rate=1e-3)
    lit_model.load_state_dict(torch.load(model_path)["state_dict"])
    logging.info(f"Loaded model from {model_path}")
    logging.info(f"Model architecture: {lit_model}")
    lit_model.eval()
    return lit_model


def get_next_train_data(past_values: torch.Tensor, next_value: torch.Tensor):
    """
    What this should do:
    1. Remove the first value from past_values
    2. append next_value to past_values (at last position)

    past values shape: batch_size, seq_len, input_size
    next_value shape: batch_size, input_size

    Returns:
        next_train_data: shape: batch_size, seq_len, input_size
    """
    # print(past_values)
    # print(next_value)
    return torch.cat((past_values[:, 1:, :], next_value.unsqueeze(1)), dim=1)


@torch.no_grad()
def autoregressive_predictions(model, test_dataloader, thresh_mae=0.1):
    model.eval()
    model.to(A_100_DEVICE_ID)

    test_iter = iter(test_dataloader)  # create the iterator once
    curr_batch = next(test_iter)
    batch = {k: v.to(A_100_DEVICE_ID) for k, v in curr_batch.items()}
    n_predictions = 0
    mae_arr = []

    while True:
        past_values = batch["past_values"]
        pred = lit_model(past_values)
        true_cont_y = batch["continuous_y"]
        true_bin_y = batch["binary_y"]

        print(true_cont_y.shape, true_bin_y.shape, pred.squeeze(1).shape)
        output = torch.cat([pred.squeeze(1), true_bin_y], dim=1)
        mae = torch.mean(torch.abs(pred - true_cont_y))
        rmse = torch.sqrt(torch.mean((pred - true_cont_y) ** 2))
        print(f"MAE: {mae.item()}, RMSE: {rmse.item()}")
        mae_arr.append(mae.item())
        n_predictions += 1

        try:
            curr_batch = next(test_iter)  # continue from where we left off
        except StopIteration:
            print("End of test dataloader reached.")
            break

        updated_values = get_next_train_data(past_values, output)
        curr_batch["past_values"] = updated_values  # autoregressive update
        print(f"True continuous y: {true_cont_y.squeeze(1)}")
        print(f"Predicted continuous y: {pred.squeeze(1)}")

        if mae > thresh_mae or n_predictions > 100:
            break

        batch = {k: v.to(A_100_DEVICE_ID) for k, v in curr_batch.items()}
        _ = input("Press Enter to continue to next prediction...")

    print(f"MAE array: {mae_arr}")
    print(f"Number of predictions made: {n_predictions}")


if __name__ == "__main__":
    train_data_path = "/media/8TB_hardisk/sangam/timeseries_forecasting/train_data.csv"
    base_dataset = get_dataset(train_data_path, 20, 1)
    test_data_path = "/media/8TB_hardisk/sangam/timeseries_forecasting/test_data.csv"
    base_test_dataset = get_dataset(test_data_path, 20, 1)
    transformer_dataset = dataset.TransformerDataset(base_dataset)
    test_dataset = dataset.TransformerDataset(base_test_dataset)
    test_dataloader = get_dataloader(test_dataset, batch_size=1, shuffle=False)
    model = SimpleTimeSeriesTransformer(
        num_features=Config.input_size,
        d_model=32,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        pred_len=Config.prediction_length,
        output_dim=Config.output_size,
    )
    lit_model_path = "/media/8TB_hardisk/sangam/timeseries_forecasting/src/custom_transformers/outputs/lightning_logs/version_0/epoch=3-step=65288.ckpt"
    lit_model = load_lit_model(model, lit_model_path)
    autoregressive_predictions(lit_model.model, test_dataloader, thresh_mae=0.5)
