from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from statsmodels.tsa.stattools import adfuller

from .features import auc_score


BASELINE_FEATURES = [
    "age",
    "vibration",
    "temperature",
    "pressure",
]

ADVANCED_FEATURES = [
    "age",
    "vibration",
    "temperature",
    "pressure",
    "vibration_mean_8",
    "vibration_std_8",
    "vibration_slope_8",
    "temperature_mean_8",
    "temperature_std_8",
    "temperature_slope_8",
    "pressure_mean_8",
    "pressure_std_8",
    "pressure_slope_8",
    "holt_level",
    "holt_trend",
    "vibration_resid",
    "health_proxy",
]

RICH_TS_FEATURES = [
    "age",
    "regime",
    "vibration",
    "temperature",
    "pressure",
    "vibration_mean_3",
    "vibration_std_3",
    "vibration_slope_3",
    "vibration_mean_8",
    "vibration_std_8",
    "vibration_slope_8",
    "vibration_mean_15",
    "vibration_std_15",
    "vibration_slope_15",
    "temperature_mean_3",
    "temperature_std_3",
    "temperature_slope_3",
    "temperature_mean_8",
    "temperature_std_8",
    "temperature_slope_8",
    "temperature_mean_15",
    "temperature_std_15",
    "temperature_slope_15",
    "pressure_mean_3",
    "pressure_std_3",
    "pressure_slope_3",
    "pressure_mean_8",
    "pressure_std_8",
    "pressure_slope_8",
    "pressure_mean_15",
    "pressure_std_15",
    "pressure_slope_15",
    "vibration_lag_1",
    "vibration_lag_2",
    "vibration_lag_4",
    "temperature_lag_1",
    "temperature_lag_2",
    "pressure_lag_1",
    "pressure_lag_2",
    "vibration_diff_1",
    "temperature_diff_1",
    "pressure_diff_1",
    "vibration_diff_2",
    "temperature_diff_2",
    "pressure_diff_2",
    "vibration_ewm_04",
    "temperature_ewm_04",
    "pressure_ewm_04",
    "vibration_ewm_resid",
    "temperature_ewm_resid",
    "pressure_ewm_resid",
    "holt_level",
    "holt_trend",
    "vibration_resid",
    "combined_stress",
    "health_proxy",
    "pressure_drop_vs_long_run",
    "temp_rise_vs_long_run",
    "vibration_acceleration",
]


@dataclass
class RiskModel:
    result: object
    feature_cols: list[str]
    means: pd.Series
    stds: pd.Series
    backend: str = "glm"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df[self.feature_cols].copy()
        x = (x - self.means) / self.stds
        return x

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        x = self.transform(df)
        if self.backend == "glm":
            x_with_const = sm.add_constant(x, has_constant="add")
            return self.result.predict(x_with_const).to_numpy()
        return self.result.predict_proba(x.to_numpy())[:, 1]


def fit_risk_model(df: pd.DataFrame, feature_cols: list[str]) -> RiskModel:
    x = df[feature_cols].copy()
    means = x.mean()
    stds = x.std().replace(0.0, 1.0)
    x = (x - means) / stds
    y = df["fail_within_horizon"].astype(float)
    x_with_const = sm.add_constant(x, has_constant="add")
    result = sm.GLM(y, x_with_const, family=sm.families.Binomial()).fit()
    return RiskModel(result=result, feature_cols=feature_cols, means=means, stds=stds, backend="glm")


def fit_tree_risk_model(df: pd.DataFrame, feature_cols: list[str]) -> RiskModel:
    x = df[feature_cols].copy()
    means = x.mean()
    stds = x.std().replace(0.0, 1.0)
    x = (x - means) / stds
    y = df["fail_within_horizon"].astype(int).to_numpy()
    result = RandomForestClassifier(
        n_estimators=180,
        max_depth=8,
        min_samples_leaf=10,
        random_state=248,
        n_jobs=1,
    )
    result.fit(x.to_numpy(), y)
    return RiskModel(result=result, feature_cols=feature_cols, means=means, stds=stds, backend="sklearn")


def evaluate_risk_model(model: RiskModel, df: pd.DataFrame, label: str) -> dict[str, float | str]:
    preds = model.predict(df)
    y = df["fail_within_horizon"].to_numpy()
    brier = float(np.mean((preds - y) ** 2))
    log_loss = float(-np.mean(y * np.log(preds + 1e-9) + (1 - y) * np.log(1 - preds + 1e-9)))
    auc = auc_score(y, preds)
    return {"model": label, "auc": auc, "brier": brier, "log_loss": log_loss}


def calibration_table(model: RiskModel, df: pd.DataFrame, label: str, n_bins: int = 10) -> pd.DataFrame:
    preds = model.predict(df)
    eval_df = pd.DataFrame({"pred": preds, "y": df["fail_within_horizon"].to_numpy()})
    eval_df["bin"] = pd.qcut(eval_df["pred"], q=n_bins, duplicates="drop")
    summary = (
        eval_df.groupby("bin", observed=False)
        .agg(mean_pred=("pred", "mean"), observed_rate=("y", "mean"), count=("y", "size"))
        .reset_index()
    )
    summary["model"] = label
    return summary


def residual_autocorrelation(model: RiskModel, df: pd.DataFrame, label: str, max_lag: int = 8) -> pd.DataFrame:
    preds = model.predict(df)
    residuals = df["fail_within_horizon"].to_numpy() - preds
    rows: list[dict[str, float | int | str]] = []
    for lag in range(1, max_lag + 1):
        corr = np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]
        rows.append({"model": label, "lag": lag, "residual_acf": float(corr)})
    return pd.DataFrame(rows)


def stationarity_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sensor in ["vibration", "temperature", "pressure"]:
        series = (
            df.groupby("machine_id", sort=True)[sensor]
            .diff()
            .dropna()
            .reset_index(drop=True)
        )
        stat, pvalue, _, _, crit_vals, _ = adfuller(series.to_numpy(), autolag="AIC")
        rows.append(
            {
                "series": f"{sensor}_first_difference",
                "adf_stat": float(stat),
                "p_value": float(pvalue),
                "crit_5pct": float(crit_vals["5%"]),
            }
        )
    return pd.DataFrame(rows)


def feature_importance_table(model: RiskModel, top_n: int = 10) -> pd.DataFrame:
    if not hasattr(model.result, "feature_importances_"):
        raise ValueError("Feature importance is only available for tree-based sklearn models.")
    importance_df = pd.DataFrame(
        {
            "feature": model.feature_cols,
            "importance": model.result.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    return importance_df.head(top_n).reset_index(drop=True)
