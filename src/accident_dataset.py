"""Self-contained, positional, scenario-bounded sliding-window dataset for the
per-accident-type ErrorMLP AR-correction pipeline.

This is ADDITIVE and self-contained: it does NOT import the cross-repo
`MIMO_TimeSeries_Transformer/data/dataset.py`; a compact leakage-safe port of its
windowing + scenario-level split lives here so THIS repo stays self-contained. It
also does NOT touch the existing 60min path (dataset.py / TransformerDataset).

Schema rule (positional, robust) -- see the 5-cell CSV headers:
    header = [scenario_number, TIME, <10 continuous>, <controls...>]
    feature_cols  = header[2:]           # everything after metadata
    num_continuous = 10 (fixed, asserted); first 10 feature names must match
    num_controls  = len(header) - 12
The 10 continuous (fixed order, ALL cells):
    PPS, TGRCS(10), TGRCS(15), PSGGEN(1), ZWDC2SG(1), ZWRB(1), PEX0(17), TWSG(1),
    TGRB(17), ZWRB(6)
Controls differ per cell (SBO: 4 SAMG, LLOCA: 4 SAMG (diff set), TLOFW: 5 SAMG);
they are the KNOWN-FUTURE covariates fed as ground truth each rollout step. The
backbone routes channels purely by column position, so a different control set is
handled transparently.

The AR rollout in error_rollout_acc.py mirrors the byte-faithful lockstep of the
existing error_rollout.py: it reads `past_values` (init window) once, then rolls
`num_continuous` back and feeds `control_y` (=truth) each step.

Item dict keys (chosen to be consumed by error_rollout_acc):
    past_values : [seq_len, input_size]  float32  (10 continuous + controls)
    continuous_y: [num_continuous]       float32  (the pred_len==1 continuous target)
    control_y   : [num_controls]         float32  (the pred_len==1 known-future controls)
    scenario_id : scalar long
"""
from __future__ import annotations

import os
import pickle
import hashlib
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset


# The 10 continuous targets, fixed order for EVERY cell (single source of truth).
CONTINUOUS_COLS: List[str] = [
    "PPS",
    "TGRCS(10)",
    "TGRCS(15)",
    "PSGGEN(1)",
    "ZWDC2SG(1)",
    "ZWRB(1)",
    "PEX0(17)",
    "TWSG(1)",
    "TGRB(17)",
    "ZWRB(6)",
]
METADATA_COLS: List[str] = ["scenario_number", "TIME"]
NUM_CONTINUOUS: int = len(CONTINUOUS_COLS)  # 10


def infer_schema_from_csv(csv_path: str) -> Tuple[List[str], int, int]:
    """Read only the header of the SCALED csv and derive the positional schema.

    Returns (feature_cols, num_continuous, num_controls). Asserts:
      * the first two columns are the metadata columns,
      * num_continuous == 10,
      * the first 10 feature names match CONTINUOUS_COLS exactly (order incl.).
    """
    header = list(pd.read_csv(csv_path, nrows=0).columns)
    assert header[:2] == METADATA_COLS, (
        f"unexpected metadata header in {csv_path}: {header[:2]} != {METADATA_COLS}"
    )
    feature_cols = header[2:]
    num_controls = len(header) - 12
    assert num_controls >= 1, f"no control columns inferred from {csv_path}: header={header}"
    num_continuous = NUM_CONTINUOUS
    got_cont = feature_cols[:num_continuous]
    assert got_cont == CONTINUOUS_COLS, (
        "continuous-column mismatch (positional schema broken).\n"
        f"  expected: {CONTINUOUS_COLS}\n  got:      {got_cont}\n  csv: {csv_path}"
    )
    return feature_cols, num_continuous, num_controls


class AccidentWindowDataset(Dataset):
    """Scenario-bounded sliding-window dataset (leakage-safe by construction).

    A window never crosses a `scenario_number` boundary. Windows are indexed once
    and cached per (csv, seq_len, pred_len) like the existing dataset. Reads the
    SCALED csv with `pd.read_csv` -- NO parquet swap (the parquet siblings are
    UNSCALED; the models were trained on `_Scaled.csv`).
    """

    def __init__(
        self,
        csv_path: str,
        seq_len: int = 50,
        pred_len: int = 1,
        feature_cols: Optional[Sequence[str]] = None,
        num_continuous: Optional[int] = None,
        cache_dir: Optional[str] = None,
        max_scenarios: Optional[int] = None,
    ) -> None:
        self.csv_path = csv_path
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)

        if feature_cols is None or num_continuous is None:
            fcols, ncont, _ = infer_schema_from_csv(csv_path)
            feature_cols = feature_cols or fcols
            num_continuous = num_continuous or ncont
        self.feature_cols = list(feature_cols)
        self.num_continuous = int(num_continuous)
        self.num_controls = len(self.feature_cols) - self.num_continuous
        assert self.num_continuous == NUM_CONTINUOUS, (
            f"num_continuous must be 10, got {self.num_continuous}"
        )
        assert self.num_controls >= 1, "at least one control column expected"
        self._cont_slice = slice(0, self.num_continuous)
        self._ctrl_slice = slice(self.num_continuous, len(self.feature_cols))

        df = pd.read_csv(csv_path)
        need = METADATA_COLS + self.feature_cols
        missing = [c for c in need if c not in df.columns]
        assert not missing, f"{csv_path} missing configured columns: {missing}"
        assert int(df.isnull().sum().sum()) == 0, f"{csv_path} contains nulls"

        if max_scenarios is not None:
            keep = pd.unique(df[METADATA_COLS[0]])[: int(max_scenarios)]
            df = df[df[METADATA_COLS[0]].isin(keep)].reset_index(drop=True)

        self._feats = df[self.feature_cols].to_numpy(dtype=np.float32)          # [N, input]
        self._scenario = df[METADATA_COLS[0]].to_numpy()                        # [N]
        self.input_size = self._feats.shape[1]

        self.series_index = self._arrange_indexes(cache_dir)

    def _cache_path(self, cache_dir: str) -> str:
        os.makedirs(cache_dir, exist_ok=True)
        tag = hashlib.md5(os.path.abspath(self.csv_path).encode()).hexdigest()[:12]
        base = os.path.splitext(os.path.basename(self.csv_path))[0]
        return os.path.join(cache_dir, f"{base}_{tag}_s{self.seq_len}_p{self.pred_len}.cache")

    def _arrange_indexes(self, cache_dir: Optional[str]) -> List[int]:
        if cache_dir:
            cache_path = self._cache_path(cache_dir)
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    return pickle.load(f)

        scen = self._scenario
        n = len(scen)
        span = self.seq_len + self.pred_len
        valid: List[int] = []
        # A window [i : i+span) is valid iff the whole span is one scenario.
        # Cheap necessary check on endpoints first, then confirm the whole span.
        for i in range(n - span + 1):
            if scen[i] == scen[i + span - 1] and np.all(scen[i:i + span] == scen[i]):
                valid.append(i)

        if cache_dir:
            with open(self._cache_path(cache_dir), "wb") as f:
                pickle.dump(valid, f)
        return valid

    def window_scenarios(self) -> np.ndarray:
        return np.asarray([self._scenario[s] for s in self.series_index])

    def __len__(self) -> int:
        return len(self.series_index)

    def __getitem__(self, index: int) -> dict:
        start = self.series_index[index]
        enc_end = start + self.seq_len
        dec_end = enc_end + self.pred_len

        past = self._feats[start:enc_end]                      # [seq_len, input]
        future = self._feats[enc_end:dec_end]                  # [pred_len, input]
        # pred_len==1 for this pipeline: squeeze the horizon so the rollout code
        # sees a per-step [nc] / [nctrl] target (matches the existing lockstep).
        cont_y = future[0, self._cont_slice]                   # [num_continuous]
        ctrl_y = future[0, self._ctrl_slice]                   # [num_controls]

        return {
            "past_values": torch.from_numpy(np.ascontiguousarray(past)).float(),
            "continuous_y": torch.from_numpy(np.ascontiguousarray(cont_y)).float(),
            "control_y": torch.from_numpy(np.ascontiguousarray(ctrl_y)).float(),
            "scenario_id": torch.tensor(int(self._scenario[start]), dtype=torch.long),
        }


def scenario_disjoint_split(
    dataset: AccidentWindowDataset, train_split: float = 0.85, seed: int = 42
) -> Tuple[Subset, Subset]:
    """Split by UNIQUE scenario_number so train/val window sets are provably
    disjoint (no window leakage). Mirrors the MIMO scenario_level_split contract."""
    window_scen = dataset.window_scenarios()
    unique = np.unique(window_scen)
    rng = np.random.RandomState(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    n_train = int(round(train_split * len(shuffled)))
    if len(shuffled) > 1:
        n_train = max(1, min(n_train, len(shuffled) - 1))
    train_scen = set(shuffled[:n_train].tolist())
    val_scen = set(shuffled[n_train:].tolist())
    assert train_scen.isdisjoint(val_scen), "scenario leakage in split"
    tr_idx = [i for i, s in enumerate(window_scen) if s in train_scen]
    va_idx = [i for i, s in enumerate(window_scen) if s in val_scen]
    return Subset(dataset, tr_idx), Subset(dataset, va_idx)
