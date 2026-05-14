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
    horizon: int = 8,
    windows: tuple[int, ...] = (3, 8, 15),
    lags: tuple[int, ...] = (1, 2, 4),
) -> pd.DataFrame:
    frames = []
    for _, group in df.groupby("machine_id", sort=True):
        group = group.copy().reset_index(drop=True)
        group["fail_within_horizon"] = (group["time_to_failure"] <= horizon).astype(int)

        for sensor in ["vibration", "temperature", "pressure"]:
            for window in windows:
                group[f"{sensor}_mean_{window}"] = group[sensor].rolling(window, min_periods=1).mean()
                group[f"{sensor}_std_{window}"] = (
                    group[sensor].rolling(window, min_periods=1).std().fillna(0.0)
                )
                group[f"{sensor}_slope_{window}"] = (
                    group[sensor].rolling(window, min_periods=2).apply(_rolling_slope, raw=True).fillna(0.0)
                )

            for lag in lags:
                group[f"{sensor}_lag_{lag}"] = group[sensor].shift(lag).bfill()

            group[f"{sensor}_diff_1"] = group[sensor].diff().fillna(0.0)
            group[f"{sensor}_diff_2"] = group[sensor].diff(2).fillna(0.0)
            group[f"{sensor}_ewm_04"] = group[sensor].ewm(alpha=0.4, adjust=False).mean()
            group[f"{sensor}_ewm_resid"] = group[sensor] - group[f"{sensor}_ewm_04"]

        level, trend = _holt_features(group["vibration"].to_numpy())
        group["holt_level"] = level
        group["holt_trend"] = trend
        group["vibration_resid"] = group["vibration"] - group["holt_level"]
        group["combined_stress"] = (
            group["vibration_mean_3"] + 0.025 * group["temperature_mean_3"] - 0.03 * group["pressure_mean_3"]
        )
        group["health_proxy"] = (
            0.65 * group["vibration_mean_8"]
            + 0.03 * group["temperature_mean_8"]
            - 0.035 * group["pressure_mean_8"]
        )
        group["pressure_drop_vs_long_run"] = group["pressure_mean_3"] - group["pressure_mean_15"]
        group["temp_rise_vs_long_run"] = group["temperature_mean_3"] - group["temperature_mean_15"]
        group["vibration_acceleration"] = group["vibration_slope_3"] - group["vibration_slope_15"]
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
