import os
import pandas as pd
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset


class BaseDataset(Dataset):

    bool_dtypes = [
        "RCP_pump",
        "HX",
        "HPI",
        "LPI",
        "CNMT_Spray",
        "MDAFW",
        "Charging_pump",
        "SAMG_1",
        "SAMG_2",
        "SAMG_3",
    ]

    def __init__(
        self, filepath_arr, seq_len=24, pred_len=1, log_dir="logs", cache_dir="cache"
    ):
        self.filepath_arr = filepath_arr
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.log_dir = log_dir
        self.cache_dir = cache_dir

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)

        self.all_data = None
        self.read_files()
        # self.convert_data_types() # Useless to convert.
        self.series_index = self.arrange_indexes()

    def read_files(self):
        self.all_data = pd.read_csv(self.filepath_arr[0])
        assert self.all_data.isnull().sum().sum() == 0, "There should be no null values"
        duplicate_df = self.check_duplicates()
        duplicate_df.to_csv(f"{self.log_dir}/duplicates.csv")
        self.all_data = self.all_data.drop_duplicates().reset_index(drop=True)

    def check_duplicates(self):
        return self.all_data[self.all_data.duplicated(keep=False)]

    def convert_data_types(self):
        for col in self.bool_dtypes:
            self.all_data[col] = self.all_data[col].map(lambda x: 0 if x == 0.2 else 1)

    def get_cache_path(self):
        filename = os.path.basename(self.filepath_arr[0])
        cache_filename = filename + f"_s{self.seq_len}_p{self.pred_len}.cache"
        return os.path.join(self.cache_dir, cache_filename)

    def arrange_indexes(self):
        cache_path = self.get_cache_path()
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                print(f"Loading cached indexes from {cache_path}")
                return pickle.load(f)

        print("Generating valid indexes...")
        valid_indexes = []
        df = self.all_data
        total_len = len(df)

        for i in range(total_len - self.seq_len - self.pred_len + 1):
            window = df.iloc[i : i + self.seq_len + self.pred_len]
            if window["scenario_number"].nunique() == 1:
                valid_indexes.append(i)

        with open(cache_path, "wb") as f:
            pickle.dump(valid_indexes, f)
            print(f"Cached indexes saved to {cache_path}")

        return valid_indexes

    def __getitem__(self, index):
        start_idx = self.series_index[index]
        end_idx = start_idx + self.seq_len
        label_idx = end_idx
        label_end_idx = label_idx + self.pred_len

        # Core values
        x = (
            self.all_data.iloc[start_idx:end_idx]
            .drop(columns=["scenario_number", "TIME"])
            .values
        )
        y = (
            self.all_data.iloc[label_idx:label_end_idx]
            .drop(columns=["scenario_number", "TIME"])
            .values
        )
        past_time_features = self.all_data.iloc[start_idx:end_idx][["TIME"]].values
        future_time_features = self.all_data.iloc[label_idx:label_end_idx][
            ["TIME"]
        ].values

        # Time features (simple position encoding)

        return {
            "past_values": torch.tensor(
                x, dtype=torch.float32
            ),  # [seq_len, num_features]
            "future_values": torch.tensor(
                y, dtype=torch.float32
            ),  # [pred_len, num_features]
            "past_time_features": torch.tensor(
                past_time_features, dtype=torch.float32
            ),  # [seq_len, 1]     # [pred_len, 1]
            "past_observed_mask": torch.ones_like(
                torch.tensor(x, dtype=torch.float32)
            ),  # Assume no missing
            "future_observed_mask": torch.ones_like(
                torch.tensor(y), dtype=torch.float32
            ),
            "future_time_features": torch.tensor(
                future_time_features, dtype=torch.float32
            ),
        }

    def get_debug_item(self, index):
        """
        Return the raw window slices with TIME and optionally scenario_number for debugging.
        Includes encoder input, decoder input range, and target.
        """
        start_idx = self.series_index[index]
        end_idx = start_idx + self.seq_len
        label_idx = end_idx
        label_end_idx = label_idx + self.pred_len

        # Extract dataframes with all columns (including TIME)
        enc_window = self.all_data.iloc[start_idx:end_idx].copy()
        dec_window = self.all_data.iloc[end_idx - 1 : end_idx + self.pred_len].copy()
        target_window = self.all_data.iloc[label_idx:label_end_idx].copy()

        # Optional: add feature for position index within window (for readability)
        enc_window["__step__"] = range(-self.seq_len, 0)
        dec_window["__step__"] = range(-1, self.pred_len)
        target_window["__step__"] = range(0, self.pred_len)

        return {
            "enc_df": enc_window,  # The encoder input window [t-k:t)
            "dec_df": dec_window,  # The decoder input window [t-1:t+pred_len)
            "target_df": target_window,  # The true target values [t:t+pred_len)
        }

    def __len__(self):
        return len(self.series_index)


class TransformerDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __getitem__(self, index):
        item = self.base_dataset[index]
        # First 7 are continuous, rest are binary.
        x = item["past_values"]
        y = item["future_values"]
        y = y.view(-1)  # Remove the second dimension
        continuous_y = y[:7]
        binary_y = y[7:]
        binary_y = (binary_y > 0.5).float()
        # Binary not needed, still. Whatever, code should be complete
        return {
            "past_values": x,
            "continuous_y": continuous_y,
            "binary_y": binary_y,
        }

    def __len__(self):
        return len(self.base_dataset)


if __name__ == "__main__":
    data_path = "/media/8TB_hardisk/sangam/timeseries_forecasting/train_data.csv"
    base_dataset = BaseDataset([data_path], 20, 1)
    transformer_dataset = TransformerDataset(base_dataset)
    for i, curr_data in enumerate(transformer_dataset):
        print(curr_data["past_values"].shape)
        print(curr_data["continuous_y"].shape)
        print(curr_data["binary_y"].shape)
        print(curr_data["past_values"])
        print("--" * 20)
        if i == 2:
            break
