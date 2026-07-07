import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")  # real FP32 kernels

from torch.utils.data import DataLoader
from model_selector import ModelSelector
import utils
import logging
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import warnings
import matplotlib.pyplot as plt
import tqdm
from dtw_utils import compute_dtw_individually
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)


def load_weights(config, model_path):
    model_name = config["model"]["name"]
    backbone_kwargs = config["model"]["backbone_kwargs"]
    lightning_kwargs = config["model"].get("lightning_kwargs", {})
    base_model, lit_model = ModelSelector(
        model_name, backbone_kwargs=backbone_kwargs, lightning_kwargs=lightning_kwargs
    )
    lit_model.load_state_dict(torch.load(model_path, map_location="cpu")["state_dict"])
    logging.info(f"Loaded model from {model_path}")
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


@torch.inference_mode()  # <<< changed (added parentheses)
def autoregressive_predictions_difference(model, dataset):
    model = model.to(dtype=torch.float32).eval()
    predictions_dict = defaultdict(list)
    true_dict = defaultdict(list)

    data_point = dataset[0]  # Get the first sample

    # for i in range(1, len(dataset)):
    for i in tqdm.tqdm(range(1, len(dataset))):
        data_point = {
            k: v.unsqueeze(0).cuda().float() for k, v in data_point.items()
        }  # Move to GPU

        past_values = data_point["past_values"]
        metadata = data_point["y_metadata"]
        true_bin_y = data_point["binary_y"]

        scenario_number = metadata.view(-1)[0].item()  # <<< changed (robust read)

        pred = model(past_values.to(torch.float32))

        # Convert difference to absolute
        pred = pred.squeeze(1) + past_values[:, -1, :10]

        predictions_dict[scenario_number].append(pred.cpu().detach().numpy())
        true_dict[scenario_number].append(
            data_point["continuous_y"].squeeze(1).cpu().detach().numpy()
        )

        full_output = torch.cat([pred, true_bin_y], dim=1)
        updated_values = get_next_train_data(past_values, full_output)

        # --- minimal fix: decide using the NEXT sample's scenario ---
        next_data_point = dataset[i]  # <<< changed (peek next sample)
        next_scenario = next_data_point["y_metadata"][0].item()  # <<< changed

        if next_scenario == scenario_number:  # <<< changed (compare current vs next)
            # same scenario -> carry AR state into next sample
            next_data_point["past_values"] = updated_values.squeeze(
                0
            ).cpu()  # <<< changed
            data_point = next_data_point  # <<< changed (prepare for next loop; batching happens at top)
        else:
            # scenario changed -> reset (use dataset's own past_values)
            print(
                f"Scenario changed from {scenario_number} to {next_scenario}. Resetting past values."
            )  # <<< changed message to reflect current->next
            data_point = next_data_point  # <<< changed

    scenario_wise_metrics(predictions_dict, true_dict, debug=False)
    variable_names = [
        "PPS",
        "TGRCS(10)",
        "TGRCS(15)",
        "ZWV",
        "PSGGEN(1)",
        "ZWDC2SG(1)s",
        "MAX_CET",
        "CTMTP",
        "PZRP",
        "PZRWL",
    ]
    # plot_variables_distribution(predictions_dict, true_dict, variable_names, debug=True)
    # plot_variables_individually(predictions_dict, true_dict, variable_names, debug=True)
    df = compute_dtw_individually(predictions_dict, true_dict, variable_names)
    df.to_csv("plots/temp_dtw.csv", index=False)
    return predictions_dict, true_dict


def autoregressive_predictions(model, dataset, config):
    if config["data"]["prediction_type"] == "diff":
        print("Using difference prediction type")
        return autoregressive_predictions_difference(model, dataset)
    elif config["data"]["prediction_type"] == "absolute":
        print("Using absolute prediction type")
        return autoregressive_predictions_absolute(model, dataset)


def regressive_predictions(model, dataset, config):
    if config["data"]["prediction_type"] == "diff":
        print("Using difference prediction type")
        return regressive_predictions_difference(model, dataset)
    elif config["data"]["prediction_type"] == "absolute":
        print("Using absolute prediction type")
        return regressive_predictions_absolute(model, dataset)


@torch.inference_mode()
def regressive_predictions_difference(model, dataset):
    model = model.to(dtype=torch.float32).eval()
    predictions_dict = defaultdict(list)
    true_dict = defaultdict(list)

    data_point = dataset[0]  # Get the first sample
    for i in tqdm.tqdm(range(1, len(dataset))):
        data_point = {
            k: v.unsqueeze(0).cuda().float() for k, v in data_point.items()
        }  # Move to GPU
        past_values = data_point["past_values"]
        metadata = data_point["y_metadata"]
        true_cont_y = data_point["difference_y_cont"]

        scenario_number = metadata.view(-1)[0].item()  # <<< changed (robust read)

        pred = model(
            past_values.to(torch.float32)
        )  # IMP: We don't need to get to absolute value in difference prediction.
        predictions_dict[scenario_number].append(pred.squeeze(1).cpu().detach().numpy())
        true_dict[scenario_number].append(true_cont_y.squeeze(1).cpu().detach().numpy())
        data_point = dataset[i]  # <<< changed (peek next sample)
    scenario_wise_metrics(predictions_dict, true_dict, debug=False)
    variable_names = [
        "PPS",
        "TGRCS(10)",
        "TGRCS(15)",
        "ZWV",
        "PSGGEN(1)",
        "ZWDC2SG(1)s",
        "MAX_CET",
        "CTMTP",
        "PZRP",
        "PZRWL",
    ]
    # plot_variables_distribution(predictions_dict, true_dict, variable_names, debug=True)
    # plot_variables_individually(predictions_dict, true_dict, variable_names, debug=True)
    df = compute_dtw_individually(predictions_dict, true_dict, variable_names)
    df.to_csv("plots/temp_dtw.csv", index=False)
    return predictions_dict, true_dict


@torch.inference_mode
def regressive_predictions_absolute(model, dataset):
    model = model.to(dtype=torch.float32).eval()
    predictions_dict = defaultdict(list)
    true_dict = defaultdict(list)

    data_point = dataset[0]  # Get the first sample
    for i in tqdm.tqdm(range(1, len(dataset))):
        data_point = {
            k: v.unsqueeze(0).cuda().float() for k, v in data_point.items()
        }  # Move to GPU
        past_values = data_point["past_values"]
        metadata = data_point["y_metadata"]
        true_cont_y = data_point["continuous_y"]

        scenario_number = metadata.view(-1)[0].item()  # <<< changed (robust read)

        pred = model(past_values.to(torch.float32))

        # Metrics only for continuous_y
        predictions_dict[scenario_number].append(pred.squeeze(1).cpu().detach().numpy())
        true_dict[scenario_number].append(true_cont_y.squeeze(1).cpu().detach().numpy())
        data_point = dataset[i]  # <<< changed (peek next sample)

    scenario_wise_metrics(predictions_dict, true_dict, debug=False)
    variable_names = [
        "PPS",
        "TGRCS(10)",
        "TGRCS(15)",
        "ZWV",
        "PSGGEN(1)",
        "ZWDC2SG(1)s",
        "MAX_CET",
        "CTMTP",
        "PZRP",
        "PZRWL",
    ]
    # plot_variables_distribution(predictions_dict, true_dict, variable_names, debug=True)
    # plot_variables_individually(predictions_dict, true_dict, variable_names, debug=True)
    # df = compute_dtw_individually(predictions_dict, true_dict, variable_names)
    # df.to_csv("plots/temp_dtw.csv", index=False)
    return predictions_dict, true_dict


@torch.inference_mode
def autoregressive_predictions_absolute(model, dataset):
    model = model.to(dtype=torch.float32).eval()
    predictions_dict = defaultdict(list)
    true_dict = defaultdict(list)

    data_point = dataset[0]  # Get the first sample

    for i in tqdm.tqdm(range(1, len(dataset))):
        # for i in range(1, 500):
        data_point = {
            k: v.unsqueeze(0).cuda().float() for k, v in data_point.items()
        }  # Move to GPU

        past_values = data_point["past_values"]
        metadata = data_point["y_metadata"]
        true_cont_y = data_point["continuous_y"]
        true_bin_y = data_point["binary_y"]

        scenario_number = metadata.view(-1)[0].item()  # <<< changed (robust read)

        pred = model(past_values.to(torch.float32))

        # Metrics only for continuous_y
        predictions_dict[scenario_number].append(pred.squeeze(1).cpu().detach().numpy())
        true_dict[scenario_number].append(true_cont_y.squeeze(1).cpu().detach().numpy())

        full_output = torch.cat([pred.squeeze(1), true_bin_y], dim=1)
        updated_values = get_next_train_data(past_values, full_output)

        # --- minimal fix: decide the NEXT step before building its batch ---
        next_data_point = dataset[i]  # <<< changed (peek next sample)
        next_scenario = next_data_point["y_metadata"][0].item()  # <<< changed

        if next_scenario == scenario_number:  # <<< changed (compare to next)
            # same scenario -> carry AR state into next sample
            next_data_point["past_values"] = updated_values.squeeze(
                0
            ).cpu()  # <<< changed
            data_point = next_data_point  # <<< changed (set for next loop; will be batched at top)
        else:
            # scenario changed -> reset (use dataset's own past_values)
            print(
                f"Scenario changed from {scenario_number} to {next_scenario}. Resetting past values."
            )  # <<< changed message to reflect current->next
            data_point = next_data_point  # <<< changed

    scenario_wise_metrics(predictions_dict, true_dict, debug=False)
    variable_names = [
        "PPS",
        "TGRCS(10)",
        "TGRCS(15)",
        "ZWV",
        "PSGGEN(1)",
        "ZWDC2SG(1)s",
        "MAX_CET",
        "CTMTP",
        "PZRP",
        "PZRWL",
    ]
    plot_mean_error_accumulation(
        predictions_dict, true_dict, variable_names=variable_names, reduction="mae"
    )
    # plot_variables_distribution(predictions_dict, true_dict, variable_names, debug=True)
    # plot_variables_individually(predictions_dict, true_dict, variable_names, debug=True)
    df = compute_dtw_individually(predictions_dict, true_dict, variable_names)
    df.to_csv("plots/temp_dtw.csv", index=False)
    return predictions_dict, true_dict


def plot_variables_distribution(
    predictions_dict, true_dict, variable_names, debug=False
):
    """
    For each scenario, plot true vs predicted for all variables in one plot.
    Saves as plots/<scenario_number>/variables_distribution.png
    """
    plots_dir = "plots"
    if debug:
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir, exist_ok=True)
        elif os.path.exists(plots_dir):
            warnings.warn(f"Directory '{plots_dir}' already exists.")

    colors = plt.get_cmap(
        "tab10"
    ).colors  # Get 10 distinct colors for up to 10 variables

    for scenario in sorted(predictions_dict):
        preds = predictions_dict[scenario]  # List of (1, 7) or (7,) arrays per timestep
        trues = true_dict[scenario]

        # Stack arrays to get shape (timesteps, 7)
        y_pred = np.vstack([np.ravel(p) for p in preds])
        y_true = np.vstack([np.ravel(y) for y in trues])

        scenario_dir = os.path.join(plots_dir, str(int(scenario)))
        if not os.path.exists(scenario_dir):
            os.makedirs(scenario_dir, exist_ok=True)
        elif os.path.exists(scenario_dir):
            warnings.warn(f"Directory '{scenario_dir}' already exists.")

        plt.figure(figsize=(12, 6))
        for idx, var_name in enumerate(variable_names):
            color = colors[idx % len(colors)]
            # True: bold solid
            plt.plot(
                y_true[:, idx], label=f"True - {var_name}", color=color, linewidth=2.5
            )
            # Pred: dotted
            plt.plot(
                y_pred[:, idx],
                label=f"Predicted - {var_name}",
                color=color,
                linewidth=2,
                linestyle="dotted",
            )
        plt.xlabel("Timestep")
        plt.ylabel("Value")
        plt.title(f"Scenario {scenario} - True vs Predicted (All Variables)")
        plt.legend(fontsize=10)
        plt.tight_layout()
        out_path = os.path.join(scenario_dir, "variables_distribution.png")
        plt.savefig(out_path, dpi=200)
        plt.close()


def plot_variables_individually(
    predictions_dict, true_dict, variable_names, debug=False
):
    """
    For each scenario and each variable, plot true vs predicted as a separate file.
    Saves as plots/<scenario_number>/<variable_name>.png
    """
    plots_dir = "plots"
    if debug:
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir, exist_ok=True)
        elif os.path.exists(plots_dir):
            warnings.warn(f"Directory '{plots_dir}' already exists.")

    colors = plt.get_cmap("tab10").colors

    for scenario in sorted(predictions_dict):
        preds = predictions_dict[scenario]
        trues = true_dict[scenario]

        y_pred = np.vstack([np.ravel(p) for p in preds])
        y_true = np.vstack([np.ravel(y) for y in trues])

        scenario_dir = os.path.join(plots_dir, str(int(scenario)))
        if not os.path.exists(scenario_dir):
            os.makedirs(scenario_dir, exist_ok=True)
        elif os.path.exists(scenario_dir):
            warnings.warn(f"Directory '{scenario_dir}' already exists.")

        for idx, var_name in enumerate(variable_names):
            plt.figure(figsize=(8, 4))
            color = colors[idx % len(colors)]
            # True: bold solid
            plt.plot(y_true[:, idx], label="True", color=color, linewidth=2.5)
            # Predicted: dotted
            plt.plot(
                y_pred[:, idx],
                label="Predicted",
                color=color,
                linewidth=2,
                linestyle="dotted",
            )
            plt.xlabel("Timestep")
            plt.ylabel(var_name)
            plt.title(f"Scenario {scenario} - {var_name}")
            plt.legend(fontsize=10)
            plt.tight_layout()
            out_path = os.path.join(scenario_dir, f"zz_{var_name}.png")
            plt.savefig(out_path, dpi=200)
            plt.close()


def plot_mean_error_accumulation(
    predictions_dict: Dict[int, List[np.ndarray]],
    true_dict: Dict[int, List[np.ndarray]],
    variable_names: List[str],
    out_dir: str = "./error_accumulation",
    reduction: str = "mae",  # "mae" or "mse"
    normalize: str = "running_mean",  # "none" | "running_mean" | "per_Tmax"
    return_arrays: bool = False,
) -> Optional[Tuple[np.ndarray, List[str]]]:
    """
    Compute and plot per-variable error accumulation across scenarios and save one plot per variable.

    - reduction="mae": |pred-true|
    - reduction="mse": (pred-true)^2
    - normalize:
        * "none":         cumulative sum (will grow with t)
        * "running_mean": cumulative sum / t (bounded, intuitive)
        * "per_Tmax":     cumulative sum / T_max (compares growth vs full horizon)
    """
    os.makedirs(out_dir, exist_ok=True)

    cum_err_list = []
    V_resolved = None
    for sid in sorted(set(predictions_dict.keys()) & set(true_dict.keys())):
        preds_seq = predictions_dict[sid]
        trues_seq = true_dict[sid]
        if not preds_seq or not trues_seq:
            continue

        T = min(len(preds_seq), len(trues_seq))
        preds = np.stack(
            [np.asarray(p).reshape(-1) for p in preds_seq[:T]], axis=0
        )  # (T, V)
        trues = np.stack(
            [np.asarray(t).reshape(-1) for t in trues_seq[:T]], axis=0
        )  # (T, V)

        if V_resolved is None:
            V_resolved = preds.shape[1]
        elif preds.shape[1] != V_resolved or trues.shape[1] != V_resolved:
            raise ValueError(f"Variable dimension mismatch in scenario {sid}.")

        err = (
            np.abs(preds - trues)
            if reduction.lower() == "mae"
            else (preds - trues) ** 2
        )
        cum_err = np.cumsum(err, axis=0)  # (T, V)
        cum_err_list.append(cum_err)

    if len(cum_err_list) == 0:
        raise ValueError("No overlapping scenarios with non-empty sequences found.")

    T_max = max(arr.shape[0] for arr in cum_err_list)
    V = V_resolved
    if len(variable_names) != V:
        raise ValueError(f"Expected {V} variable names, got {len(variable_names)}.")

    # Stack with NaN padding to handle variable-length scenarios
    stack = np.full((len(cum_err_list), T_max, V), np.nan, dtype=float)
    for i, arr in enumerate(cum_err_list):
        T_i = arr.shape[0]
        stack[i, :T_i, :] = arr

    # Apply normalization before averaging across scenarios
    if normalize not in {"none", "running_mean", "per_Tmax"}:
        raise ValueError('normalize must be one of {"none","running_mean","per_Tmax"}')

    if normalize == "running_mean":
        # divide each scenario's cumulative curve by its own step index
        for i in range(stack.shape[0]):
            # build divisors: [1,2,...,T_i] then pad with NaN to T_max
            valid = ~np.isnan(stack[i, :, 0])
            t_idx = np.arange(1, valid.sum() + 1, dtype=float)
            div = np.full((T_max, 1), np.nan, dtype=float)
            div[: len(t_idx), 0] = t_idx
            stack[i, :, :] = stack[i, :, :] / div
    elif normalize == "per_Tmax":
        stack = stack / float(T_max)

    mean_curve = np.nanmean(stack, axis=0)  # (T_max, V)

    # Plot per variable
    x = np.arange(1, T_max + 1)
    for j, name in enumerate(variable_names):
        y = mean_curve[:, j]
        plt.figure(figsize=(7, 4))
        plt.plot(x, y, linewidth=2)
        ttl_norm = {
            "none": "Cumulative",
            "running_mean": "Running Mean (Cum/t)",
            "per_Tmax": f"Normalized by T_max={T_max}",
        }[normalize]
        plt.title(f"{ttl_norm} Error — {name} ({reduction.upper()})")
        plt.xlabel("Time step")
        ylabel = "Error" if normalize != "none" else "Cumulative Error"
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        safe_name = name.replace("/", "_").replace(" ", "_")
        fname = os.path.join(out_dir, f"error_accum_{j:02d}_{safe_name}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()

    return (mean_curve, variable_names) if return_arrays else None


def scenario_wise_metrics(predictions_dict, true_dict, debug=False):
    """
    For each scenario, returns a DataFrame with per-timestep metrics (MAE, RMSE, R2).

    Args:
        predictions_dict (dict): scenario -> iterable of prediction arrays per timestep
        true_dict (dict): scenario -> iterable of ground-truth arrays per timestep
        debug (bool): if True, warns on existing plot dirs, pauses, and saves plots

    Returns:
        scenario_metrics_dfs (List[pd.DataFrame]): one per scenario with columns:
            ["timestep", "MAE", "RMSE", "R2", "scenario"]

    Side effects:
        - Writes a human-readable summary txt at plots/metrics_summary.txt with:
            * Overall micro and macro averages
            * Per-scenario means (MAE, RMSE, R2)
        - When debug=True, saves per-scenario line plots for each metric.
    """
    plots_dir = "plots"
    out_txt_path = os.path.join(plots_dir, "metrics_summary.txt")

    if debug:
        if os.path.exists(plots_dir):
            warnings.warn(f"Directory '{plots_dir}' already exists.")
        else:
            os.makedirs(plots_dir, exist_ok=True)

    scenario_metrics_dfs = []
    scenario_means = []

    for scenario in sorted(predictions_dict):  # Sorted for consistent order
        preds = predictions_dict[scenario]
        trues = true_dict[scenario]

        data = {"timestep": [], "MAE": [], "RMSE": [], "R2": []}
        for t, (p, y) in enumerate(zip(preds, trues)):
            p_flat = np.ravel(p)
            y_flat = np.ravel(y)

            data["timestep"].append(t)
            data["MAE"].append(mean_absolute_error(y_flat, p_flat))
            data["RMSE"].append(mean_squared_error(y_flat, p_flat) ** 0.5)
            # Note: r2_score may warn for constant y; we let the NaN/negative reflect that case.
            data["R2"].append(r2_score(y_flat, p_flat))

        df = pd.DataFrame(data)
        df["scenario"] = scenario
        scenario_metrics_dfs.append(df)

        scenario_means.append(
            {
                "scenario": scenario,
                "MAE": df["MAE"].mean(),
                "RMSE": df["RMSE"].mean(),
                "R2": df["R2"].mean(),
            }
        )

    print(len(scenario_metrics_dfs), "scenarios processed.")

    if debug:
        _ = input("Press Enter to continue...")

    if debug:
        for df in scenario_metrics_dfs:
            scenario = df["scenario"].iloc[0]
            scenario_dir = os.path.join(plots_dir, str(scenario))
            if os.path.exists(scenario_dir):
                warnings.warn(f"Directory '{scenario_dir}' already exists.")
            else:
                os.makedirs(scenario_dir, exist_ok=True)

            for metric in ["MAE", "RMSE", "R2"]:
                plt.figure()
                plt.plot(df["timestep"], df[metric], marker="o")
                plt.title(f"Scenario {scenario} - {metric}")
                plt.xlabel("Timestep")
                plt.ylabel(metric)
                plt.tight_layout()
                plt.savefig(os.path.join(scenario_dir, f"{metric}.jpg"))
                plt.close()

    # Overall summaries
    all_df = pd.concat(scenario_metrics_dfs, ignore_index=True)
    micro_mae = all_df["MAE"].mean()
    micro_rmse = all_df["RMSE"].mean()
    micro_r2 = all_df["R2"].mean()

    means_df = pd.DataFrame(scenario_means)
    macro_mae = means_df["MAE"].mean()
    macro_rmse = means_df["RMSE"].mean()
    macro_r2 = means_df["R2"].mean()

    # Write summary txt
    os.makedirs(plots_dir, exist_ok=True)
    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write("=== Metrics Summary ===\n")
        f.write("\n-- Overall Averages --\n")
        f.write(
            f"Micro (weighted by timesteps): MAE={micro_mae:.6f}, RMSE={micro_rmse:.6f}, R2={micro_r2:.6f}\n"
        )
        f.write(
            f"Macro (mean of per-scenario means): MAE={macro_mae:.6f}, RMSE={macro_rmse:.6f}, R2={macro_r2:.6f}\n"
        )
        f.write("\n-- Per-Scenario Means --\n")
        for row in means_df.sort_values(
            by="scenario", key=lambda s: s.astype(str)
        ).itertuples(index=False):
            f.write(
                f"Scenario {row.scenario}: MAE={row.MAE:.6f}, RMSE={row.RMSE:.6f}, R2={row.R2:.6f}\n"
            )

    print(f"Summary written to: {out_txt_path}")
    return scenario_metrics_dfs


if __name__ == "__main__":
    selected_model = "transformer_decoder"
    # model_path = "/media/8TB_hardisk/sangam/timeseries_forecasting/src/training_logs/transformer_decoder/checkpoints/epoch=8-val_loss=0.00016208-step=745920.ckpt"
    # model_path = "/media/8TB_hardisk/sangam/timeseries_forecasting/src/training_logs/transformer_decoder/transformer_decoder_wonung_checkpoints_diff/epoch=9-val_loss=0.00027996-step=198460.ckpt"
    model_path = "/media/8TB_hardisk/sangam/timeseries_forecasting/src/training_logs/transformer_decoder/transformer_decoder_wonung_checkpoints_absolute/epoch=8-val_loss=0.00014254-step=538335.ckpt"
    config = utils.load_config(selected_model)
    logging.info(f"Loaded config for {selected_model}: {config}")
    lit_model = load_weights(config, model_path)
    lit_model.cuda()

    test_dataset = utils.get_dataset(
        config["data"]["test_data_path"],
        config["data"]["sequence_length"],
        config["data"]["prediction_length"],
        config["data"]["prediction_type"],
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
    )
    autoregressive_predictions_absolute(lit_model, test_dataset)
