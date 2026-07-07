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
    # bool_dtypes = [
    #     "SAMG-01 POSRV",
    #     "SAMG-02 SG Injection",
    #     "SAMG-03 RCS Injection",
    #     "SAMG-06 Spray Pump",
    #     "SAMG-06 ECSBS",
    # ]

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
        print(self.all_data.columns)
        assert self.all_data.isnull().sum().sum() == 0, "There should be no null values"
        duplicate_df = self.check_duplicates()
        duplicate_df.to_csv(f"{self.log_dir}/duplicates.csv")
        self.all_data = self.all_data.drop_duplicates().reset_index(drop=True)

    def check_duplicates(self):
        return self.all_data[self.all_data.duplicated(keep=False)]

    def convert_data_types(self):
        for col in self.bool_dtypes:
            self.all_data[col] = (self.all_data[col] >= 1e-3).astype(int)

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
        # Check that we are not exceeding dataset length
        if label_end_idx > len(self.all_data):
            raise IndexError(
                f"Sample at index {index} would exceed dataset bounds: "
                f"label_end_idx={label_end_idx}, len(all_data)={len(self.all_data)}"
            )

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
        y_metadata = self.all_data.iloc[label_idx:label_end_idx][
            ["scenario_number", "TIME"]
        ].values
        x_metadata = self.all_data.iloc[start_idx:end_idx][
            ["scenario_number", "TIME"]
        ].values
        past_time_features = self.all_data.iloc[start_idx:end_idx][["TIME"]].values
        future_time_features = self.all_data.iloc[label_idx:label_end_idx][
            ["TIME"]
        ].values

        # Time features (simple position encoding)

        return {
            "past_values": torch.tensor(
                x, dtype=torch.float64
            ),  # [seq_len, num_features]
            "future_values": torch.tensor(
                y, dtype=torch.float64
            ),  # [pred_len, num_features]
            "past_time_features": torch.tensor(
                past_time_features, dtype=torch.float64
            ),  # [seq_len, 1]     # [pred_len, 1]
            "past_observed_mask": torch.ones_like(
                torch.tensor(x, dtype=torch.float64)
            ),  # Assume no missing
            "future_observed_mask": torch.ones_like(
                torch.tensor(y), dtype=torch.float64
            ),
            "future_time_features": torch.tensor(
                future_time_features, dtype=torch.float64
            ),
            "y_metadata": torch.tensor(
                y_metadata, dtype=torch.float64
            ),  # [pred_len, 2] for scenario_number and TIME
            "x_metadata": torch.tensor(
                x_metadata, dtype=torch.float64
            ),  # [seq_len, 2] for scenario_number and TIME
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
    allowed_dataset_types = ["diff", "absolute"]

    def __init__(self, base_dataset, dataset_type="diff"):
        self.base_dataset = base_dataset
        if dataset_type not in self.allowed_dataset_types:
            raise ValueError(
                f"Invalid dataset_type '{dataset_type}'. Allowed types: {self.allowed_dataset_types}"
            )
        self.dataset_type = dataset_type

    def calculate_difference(self, past_values, future_values):
        return future_values - past_values

    def __getitem__(self, index):
        item = self.base_dataset[index]
        # First 7 are continuous, rest are binary.
        x = item["past_values"]
        y = item["future_values"]
        y = y.view(-1)  # Remove the second dimension
        y_metadata = item["y_metadata"].view(-1)
        continuous_y = y[:10]
        difference_y_cont = self.calculate_difference(x[-1][:10], continuous_y)
        binary_y = y[10:]
        # binary_y = (binary_y > 0.5).float()
        # Binary not needed, still. Whatever, code should be complete
        return {
            "past_values": x,
            "x_metadata": item["x_metadata"],
            "continuous_y": continuous_y,
            "binary_y": binary_y,
            "y_metadata": y_metadata,
            "difference_y_cont": difference_y_cont,
        }

    def __len__(self):
        return len(self.base_dataset)


def test_real_dataset():
    # Use your actual data path
    data_path = "/media/8TB_hardisk/sangam/timeseries_forecasting/train_data.csv"
    seq_len = 20
    pred_len = 1

    # Import your dataset classes (adjust import if needed)
    # Instantiate datasets
    base_dataset = BaseDataset([data_path], seq_len=seq_len, pred_len=pred_len)
    transformer_dataset = TransformerDataset(base_dataset)

    # Check lengths
    assert len(base_dataset) > 0, "BaseDataset is empty!"
    assert len(transformer_dataset) == len(base_dataset), "Dataset lengths mismatch!"

    # Test first batch from BaseDataset
    base_sample = base_dataset[0]
    print("BaseDataset __getitem__ output keys:", base_sample.keys())
    assert isinstance(base_sample, dict)
    assert base_sample["past_values"].shape[0] == seq_len, "past_values wrong seq_len"
    assert (
        base_sample["future_values"].shape[0] == pred_len
    ), "future_values wrong pred_len"

    # Test first batch from TransformerDataset
    trans_sample = transformer_dataset[0]
    print("TransformerDataset __getitem__ output keys:", trans_sample.keys())
    assert (
        trans_sample["past_values"].shape[0] == seq_len
    ), "TransformerDataset past_values wrong seq_len"
    assert (
        trans_sample["continuous_y"].numel() == 7
    ), "continuous_y should have 7 elements"
    assert trans_sample["binary_y"].numel() == 10, "binary_y should have 10 elements"
    assert torch.all(
        (trans_sample["binary_y"] == 0) | (trans_sample["binary_y"] == 1)
    ), "binary_y must be 0 or 1"

    # Test get_debug_item
    debug = base_dataset.get_debug_item(0)
    print("get_debug_item keys:", debug.keys())
    assert debug["enc_df"].shape[0] == seq_len, "enc_df window shape mismatch"
    assert debug["target_df"].shape[0] == pred_len, "target_df window shape mismatch"

    print("✅ Real-data BaseDataset and TransformerDataset smoke test passed!")


def test_no_cross_scenario(base_dataset):
    for idx in np.random.choice(len(base_dataset), size=100, replace=False):
        win = base_dataset.get_debug_item(idx)
        assert win["enc_df"]["scenario_number"].nunique() == 1
        assert win["dec_df"]["scenario_number"].nunique() == 1
        assert win["target_df"]["scenario_number"].nunique() == 1


def test_time_monotonic(base_dataset):
    for idx in np.random.choice(len(base_dataset), 50, False):
        t = base_dataset.get_debug_item(idx)["enc_df"]["TIME"].values
        assert np.all(np.diff(t) >= 0), "TIME must be non-decreasing"


def test_index_cache(tmp_path, data_path):
    # build once
    ds1 = BaseDataset([data_path], cache_dir=tmp_path)
    n1 = len(ds1)
    # reload cached file
    ds2 = BaseDataset([data_path], cache_dir=tmp_path)
    assert len(ds2) == n1
    assert ds1.series_index == ds2.series_index


def test_dataset_type_switch(base_dataset):
    diff_ds = TransformerDataset(base_dataset, "diff")
    abs_ds = TransformerDataset(base_dataset, "abs")
    for i in range(10):
        diff_item = diff_ds[i]
        abs_item = abs_ds[i]
        diff = diff_item["difference_y_cont"]
        should_be_same = abs_item["difference_y_cont"]
        # for 'abs' they should match the raw continuous_y
        assert torch.allclose(
            should_be_same, abs_item["continuous_y"]
        ), "dataset_type=abs broken"


if __name__ == "__main__":
    data_path = "/media/8TB_hardisk/sangam/timeseries_forecasting/data/timeseries_kaist_data/APR1400_train_data_fixed_normalized15.csv"
    seq_len = 5
    pred_len = 1

    # Import your dataset classes (adjust import if needed)
    # Instantiate datasets
    base_dataset = BaseDataset([data_path], seq_len=seq_len, pred_len=pred_len)
    transformer_dataset = TransformerDataset(base_dataset)
    print(len(transformer_dataset.base_dataset))
    dataset_length = len(transformer_dataset) - 1
    print(transformer_dataset[dataset_length])

    test_time_monotonic(base_dataset)
    test_index_cache(tmp_path="/tmp", data_path=data_path)
    test_dataset_type_switch(base_dataset)
    # test_real_dataset()
