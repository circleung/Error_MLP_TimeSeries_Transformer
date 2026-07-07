import yaml
from dataset import BaseDataset, TransformerDataset
from pathlib import Path


# Directory that holds *this* .py file
_THIS_DIR = Path(__file__).resolve().parent
# Project root  (adjust “..” if this file lives deeper)
_ROOT_DIR = _THIS_DIR
_CONFIG_DIR = _ROOT_DIR / "configs"


# def load_config(selected_model):
#     """
#     Load the configuration for the selected model from a YAML file.
#     """
#     config_path = f"configs/{selected_model}.yaml"
#     with open(config_path, "r") as file:
#         config = yaml.safe_load(file)
#     return config


def load_config(selected_model: str):
    cfg_file = _CONFIG_DIR / f"{selected_model}.yaml"
    with cfg_file.open("r") as f:
        return yaml.safe_load(f)


def get_dataset(data_path, sequence_length, pred_length, prediction_type="absolute"):
    """
    Create a dataset from the given data path with specified sequence and prediction lengths.
    """
    base_dataset = BaseDataset(
        [data_path], seq_len=sequence_length, pred_len=pred_length
    )
    return TransformerDataset(base_dataset, dataset_type=prediction_type)
