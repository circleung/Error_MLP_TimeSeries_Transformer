import numpy as np
import pandas as pd

# dtw-python (https://dynamictimewarping.github.io/python/)
from dtw import dtw, symmetric2  # step pattern commonly used


def compute_dtw_individually(
    predictions_dict,
    true_dict,
    variable_names,
    step_pattern=symmetric2,  # default step pattern
    open_end=False,
    open_begin=False,
    window_type=None,  # e.g., "sakoechiba"
    window_args=None,  # e.g., {"window_size": 10}
    distance_only=True,  # faster; we only need distance
    na_policy="drop",  # "drop" -> drop NaNs; "raise" -> error
    dt_minutes=15,
):
    """
    Compute DTW distance per variable per scenario and return a tidy DataFrame.

    Parameters
    ----------
    predictions_dict : dict
        {scenario_number: list/iterable of arrays} shaped like your plotting fn.
    true_dict : dict
        {scenario_number: list/iterable of arrays} shaped like your plotting fn.
    variable_names : list[str]
        Names for each variable/column.
    step_pattern : callable
        DTW step pattern from dtw-python (default: symmetric2).
    open_end, open_begin : bool
        Open-ended alignment flags passed to dtw().
    window_type : str or None
        Local constraint window (e.g., "sakoechiba" or "itakura") or None.
    window_args : dict or None
        Args for the chosen window (e.g., {"window_size": 10}).
    distance_only : bool
        If True, dtw() computes only the distance for speed.
    na_policy : {"drop","raise"}
        How to handle NaNs/inf in the sequences.

    Returns
    -------
    pd.DataFrame with columns ["scenario_number", "variable_name", "dtw"]
    """
    rows = []

    # Iterate scenarios in same way as your plotting function
    for scenario in sorted(predictions_dict):
        if scenario not in true_dict:
            raise KeyError(f"Scenario {scenario} missing in true_dict.")

        preds = predictions_dict[scenario]
        trues = true_dict[scenario]

        # Stack like your plotting: T x D
        y_pred = np.vstack([np.ravel(p) for p in preds])
        y_true = np.vstack([np.ravel(y) for y in trues])

        if y_pred.shape[1] != y_true.shape[1]:
            raise ValueError(
                f"Dim mismatch in scenario {scenario}: "
                f"pred D={y_pred.shape[1]} vs true D={y_true.shape[1]}"
            )
        if len(variable_names) != y_true.shape[1]:
            raise ValueError(
                f"variable_names length ({len(variable_names)}) does not match "
                f"data dimension ({y_true.shape[1]}) in scenario {scenario}."
            )

        # Compute DTW for each variable (column)
        for idx, var_name in enumerate(variable_names):
            s_true = y_true[:, idx].astype(float)
            s_pred = y_pred[:, idx].astype(float)

            # Handle NaNs/Infs
            if na_policy not in {"drop", "raise"}:
                raise ValueError("na_policy must be 'drop' or 'raise'.")

            if na_policy == "drop":
                mask = np.isfinite(s_true) & np.isfinite(s_pred)
                s_true = s_true[mask]
                s_pred = s_pred[mask]
                if s_true.size == 0 or s_pred.size == 0:
                    raise ValueError(
                        f"After dropping NaNs/Infs, empty series for "
                        f"scenario {scenario}, variable '{var_name}'."
                    )
            else:  # "raise"
                if not (np.all(np.isfinite(s_true)) and np.all(np.isfinite(s_pred))):
                    raise ValueError(
                        f"Found NaN/Inf in scenario {scenario}, variable '{var_name}'."
                    )

            # dtw-python expects 1D arrays (or sequences)
            # Configure window if requested
            wtype = window_type
            wargs = window_args or {}

            align = dtw(
                s_true,
                s_pred,
                step_pattern=step_pattern,
                open_end=open_end,
                open_begin=open_begin,
                window_type=wtype,
                window_args=wargs,
                distance_only=distance_only,  # fastest path since we only need distance
                keep_internals=not distance_only,  # unnecessary when distance_only=True
            )

            # In distance-only mode, align is a float distance in recent versions.
            # In some versions it returns an object with `.distance`.
            # distance = float(getattr(align, "distance", align))
            distance = float(align.distance)
            distance = distance**0.5

            rows.append(
                {
                    "scenario_number": int(scenario),
                    "variable_name": var_name,
                    "dtw": distance,
                }
            )

    df = pd.DataFrame(rows, columns=["scenario_number", "variable_name", "dtw"])
    summary_df = (
        df.groupby("variable_name")["dtw"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "dtw_mean", "std": "dtw_std"})
    )
    print("DTW Summary:")
    print(summary_df)
    return df
