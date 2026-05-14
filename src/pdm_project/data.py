from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


COLUMN_NAMES = [
    "machine_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    *[f"sensor_{i}" for i in range(1, 22)],
]

DEFAULT_SUBSET = "FD001"
ALL_SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
DEFAULT_SENSOR_COLUMNS = ["sensor_7", "sensor_11", "sensor_12", "sensor_15", "sensor_20", "sensor_21"]
REPRESENTATIVE_SENSOR_COLUMNS = ["sensor_7", "sensor_12", "sensor_21"]
SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]


@dataclass(frozen=True)
class CMAPSSConfig:
    subset: str = DEFAULT_SUBSET
    data_dir: Path = Path("data/CMAPSSData")
    label_horizon: int = 30
    train_units: int = 70
    validation_units: int = 15
    test_units: int = 15
    split_seed: int = 248


def _read_cmapss_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected C-MAPSS file at '{path}'. Place NASA C-MAPSS data under "
            f"'data/CMAPSSData/' with files like 'train_FD001.txt'."
        )

    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] > len(COLUMN_NAMES):
        df = df.iloc[:, : len(COLUMN_NAMES)]
    df.columns = COLUMN_NAMES
    return df


def load_cmapss_train_frame(config: CMAPSSConfig | None = None) -> pd.DataFrame:
    config = config or CMAPSSConfig()
    train_path = config.data_dir / f"train_{config.subset}.txt"
    df = _read_cmapss_file(train_path).copy()
    df["age"] = df["cycle"].astype(int)
    df["failure_time"] = df.groupby("machine_id", sort=True)["cycle"].transform("max").astype(int)
    df["time_to_failure"] = (df["failure_time"] - df["cycle"]).astype(int)
    df["failed"] = (df["time_to_failure"] == 0).astype(int)
    return df


def split_train_valid_test(
    df: pd.DataFrame,
    train_units: int,
    validation_units: int,
    test_units: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unit_ids = np.array(sorted(df["machine_id"].unique()), dtype=int)
    n_units = len(unit_ids)
    if train_units + validation_units + test_units > n_units:
        raise ValueError(
            f"Requested {train_units + validation_units + test_units} unit splits, but only found {n_units} units."
        )

    rng = np.random.default_rng(seed)
    shuffled = unit_ids.copy()
    rng.shuffle(shuffled)

    train_ids = set(shuffled[:train_units])
    valid_ids = set(shuffled[train_units : train_units + validation_units])
    test_ids = set(shuffled[train_units + validation_units : train_units + validation_units + test_units])

    train_df = df[df["machine_id"].isin(train_ids)].copy()
    valid_df = df[df["machine_id"].isin(valid_ids)].copy()
    test_df = df[df["machine_id"].isin(test_ids)].copy()
    return train_df, valid_df, test_df


def load_cmapss_splits(
    config: CMAPSSConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    config = config or CMAPSSConfig()
    df = load_cmapss_train_frame(config)

    sensor_cols = [col for col in DEFAULT_SENSOR_COLUMNS if col in df.columns]
    train_df, valid_df, test_df = split_train_valid_test(
        df=df,
        train_units=config.train_units,
        validation_units=config.validation_units,
        test_units=config.test_units,
        seed=config.split_seed,
    )
    return train_df, valid_df, test_df, sensor_cols
