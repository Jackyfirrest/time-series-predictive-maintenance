from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_slope(values: np.ndarray) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x_centered = x - x.mean()
    y_centered = values - values.mean()
    denom = np.sum(x_centered**2)
    if denom == 0.0:
        return 0.0
    return float(np.sum(x_centered * y_centered) / denom)


def _holt_features(values: np.ndarray, alpha: float = 0.35, beta: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    level = np.zeros(len(values))
    trend = np.zeros(len(values))
    if len(values) == 0:
        return level, trend
    level[0] = values[0]
    trend[0] = 0.0
    for i in range(1, len(values)):
        prev_level = level[i - 1]
        level[i] = alpha * values[i] + (1.0 - alpha) * (level[i - 1] + trend[i - 1])
        trend[i] = beta * (level[i] - prev_level) + (1.0 - beta) * trend[i - 1]
    return level, trend


def build_supervised_frame(
    df: pd.DataFrame,
    sensor_cols: list[str],
    setting_cols: list[str] | None = None,
    horizon: int = 30,
    windows: tuple[int, ...] = (5, 15, 30),
    lags: tuple[int, ...] = (1, 3, 5),
) -> pd.DataFrame:
    setting_cols = setting_cols or []
    primary_sensor = sensor_cols[-1]
    frames = []
    for _, group in df.groupby("machine_id", sort=True):
        group = group.copy().reset_index(drop=True)
        engineered: dict[str, pd.Series | np.ndarray] = {
            "fail_within_horizon": (group["time_to_failure"] <= horizon).astype(int),
            "cycle_ratio": group["age"] / max(float(group["age"].max()), 1.0),
        }

        for sensor in sensor_cols:
            for window in windows:
                rolling = group[sensor].rolling(window, min_periods=1)
                engineered[f"{sensor}_mean_{window}"] = rolling.mean()
                engineered[f"{sensor}_std_{window}"] = rolling.std().fillna(0.0)
                engineered[f"{sensor}_slope_{window}"] = (
                    group[sensor].rolling(window, min_periods=2).apply(_rolling_slope, raw=True).fillna(0.0)
                )

            for lag in lags:
                engineered[f"{sensor}_lag_{lag}"] = group[sensor].shift(lag).bfill()

            engineered[f"{sensor}_diff_1"] = group[sensor].diff().fillna(0.0)
            engineered[f"{sensor}_diff_2"] = group[sensor].diff(2).fillna(0.0)
            engineered[f"{sensor}_ewm_04"] = group[sensor].ewm(alpha=0.4, adjust=False).mean()
            engineered[f"{sensor}_ewm_resid"] = group[sensor] - engineered[f"{sensor}_ewm_04"]

        for setting in setting_cols:
            engineered[f"{setting}_demeaned"] = group[setting] - group[setting].mean()

        level, trend = _holt_features(group[primary_sensor].to_numpy())
        engineered["holt_level"] = level
        engineered["holt_trend"] = trend
        engineered["primary_sensor_resid"] = group[primary_sensor] - engineered["holt_level"]
        engineered["mean_sensor_level_5"] = pd.DataFrame(
            {sensor: engineered[f"{sensor}_mean_{windows[0]}"] for sensor in sensor_cols}
        ).mean(axis=1)
        engineered["mean_sensor_level_30"] = pd.DataFrame(
            {sensor: engineered[f"{sensor}_mean_{windows[-1]}"] for sensor in sensor_cols}
        ).mean(axis=1)
        engineered["sensor_stress_gap"] = engineered["mean_sensor_level_5"] - engineered["mean_sensor_level_30"]
        engineered["primary_sensor_acceleration"] = (
            engineered[f"{primary_sensor}_slope_{windows[0]}"] - engineered[f"{primary_sensor}_slope_{windows[-1]}"]
        )
        group = pd.concat([group, pd.DataFrame(engineered, index=group.index)], axis=1)
        frames.append(group)
    return pd.concat(frames, ignore_index=True)


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = pos.sum()
    n_neg = neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=float)
    pos_rank_sum = ranks[pos].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)
