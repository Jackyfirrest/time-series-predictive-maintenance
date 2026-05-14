from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from statsmodels.tsa.stattools import adfuller

from .features import auc_score


def build_feature_sets(
    sensor_cols: list[str],
    setting_cols: list[str] | None = None,
    windows: tuple[int, ...] = (5, 15, 30),
    lags: tuple[int, ...] = (1, 3, 5),
) -> tuple[list[str], list[str], list[str]]:
    setting_cols = setting_cols or []
    primary_sensor = sensor_cols[-1]

    # Main comparison is snapshot vs temporal context.
    # We exclude explicit lifecycle position from the baseline because on
    # run-to-failure C-MAPSS splits, age can dominate the task and obscure
    # whether temporal summaries themselves add predictive value.
    baseline = [*setting_cols, *sensor_cols]

    advanced = baseline + [
        item
        for sensor in sensor_cols
        for item in (
            f"{sensor}_mean_{windows[1]}",
            f"{sensor}_std_{windows[1]}",
            f"{sensor}_slope_{windows[1]}",
        )
    ] + [
        "holt_level",
        "holt_trend",
        "primary_sensor_resid",
        "sensor_stress_gap",
    ]

    rich = baseline.copy()
    for sensor in sensor_cols:
        for window in windows:
            rich.extend(
                [
                    f"{sensor}_mean_{window}",
                    f"{sensor}_std_{window}",
                    f"{sensor}_slope_{window}",
                ]
            )
        for lag in lags:
            rich.append(f"{sensor}_lag_{lag}")
        rich.extend(
            [
                f"{sensor}_diff_1",
                f"{sensor}_diff_2",
                f"{sensor}_ewm_04",
                f"{sensor}_ewm_resid",
            ]
        )

    rich.extend(
        [
            "holt_level",
            "holt_trend",
            "primary_sensor_resid",
            "mean_sensor_level_5",
            "mean_sensor_level_30",
            "sensor_stress_gap",
            "primary_sensor_acceleration",
            f"{primary_sensor}_mean_{windows[-1]}",
            f"{primary_sensor}_slope_{windows[0]}",
        ]
    )
    return baseline, advanced, rich


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
    return fit_tree_risk_model_with_params(df, feature_cols, {})


def fit_tree_risk_model_with_params(
    df: pd.DataFrame,
    feature_cols: list[str],
    params: dict[str, int | float | str | None],
) -> RiskModel:
    x = df[feature_cols].copy()
    means = x.mean()
    stds = x.std().replace(0.0, 1.0)
    x = (x - means) / stds
    y = df["fail_within_horizon"].astype(int).to_numpy()
    model_params = {
        "n_estimators": 180,
        "max_depth": 8,
        "min_samples_leaf": 10,
        "random_state": 248,
        "n_jobs": 1,
    }
    model_params.update(params)
    result = RandomForestClassifier(
        **model_params,
    )
    result.fit(x.to_numpy(), y)
    return RiskModel(result=result, feature_cols=feature_cols, means=means, stds=stds, backend="sklearn")


def tune_tree_risk_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[RiskModel, pd.DataFrame]:
    candidate_params = [
        {"n_estimators": 180, "max_depth": 8, "min_samples_leaf": 10, "max_features": "sqrt"},
        {"n_estimators": 240, "max_depth": 8, "min_samples_leaf": 8, "max_features": "sqrt"},
        {"n_estimators": 240, "max_depth": 10, "min_samples_leaf": 8, "max_features": "sqrt"},
        {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 6, "max_features": 0.5},
        {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 6, "max_features": 0.5},
    ]

    rows: list[dict[str, float | int | str]] = []
    best_model: RiskModel | None = None
    best_key: tuple[float, float] | None = None

    for idx, params in enumerate(candidate_params, start=1):
        model = fit_tree_risk_model_with_params(train_df, feature_cols, params)
        metrics = evaluate_risk_model(model, valid_df, f"candidate_{idx}")
        row = {
            "candidate": idx,
            "n_estimators": int(params["n_estimators"]),
            "max_depth": int(params["max_depth"]),
            "min_samples_leaf": int(params["min_samples_leaf"]),
            "max_features": str(params["max_features"]),
            "auc": float(metrics["auc"]),
            "pr_auc": float(metrics["pr_auc"]),
            "brier": float(metrics["brier"]),
            "log_loss": float(metrics["log_loss"]),
        }
        rows.append(row)
        key = (float(metrics["brier"]), -float(metrics["auc"]))
        if best_key is None or key < best_key:
            best_key = key
            best_model = model

    tuning_df = pd.DataFrame(rows).sort_values(["brier", "auc"], ascending=[True, False]).reset_index(drop=True)
    if best_model is None:
        raise RuntimeError("Tree-model tuning failed to select a model.")
    return best_model, tuning_df


def evaluate_risk_model(model: RiskModel, df: pd.DataFrame, label: str) -> dict[str, float | str]:
    preds = model.predict(df)
    y = df["fail_within_horizon"].to_numpy()
    brier = float(np.mean((preds - y) ** 2))
    log_loss = float(-np.mean(y * np.log(preds + 1e-9) + (1 - y) * np.log(1 - preds + 1e-9)))
    auc = auc_score(y, preds)
    pr_auc = float(average_precision_score(y, preds))
    return {"model": label, "auc": auc, "pr_auc": pr_auc, "brier": brier, "log_loss": log_loss}


def bootstrap_risk_metrics(
    model: RiskModel,
    df: pd.DataFrame,
    label: str,
    n_bootstrap: int = 100,
    seed: int = 248,
) -> dict[str, float | str]:
    point = evaluate_risk_model(model, df, label)
    machine_ids = np.array(sorted(df["machine_id"].unique()))
    rng = np.random.default_rng(seed)

    preds = model.predict(df)
    eval_df = df[["machine_id", "fail_within_horizon"]].copy()
    eval_df["pred"] = preds

    auc_samples: list[float] = []
    pr_samples: list[float] = []
    brier_samples: list[float] = []
    logloss_samples: list[float] = []

    for _ in range(n_bootstrap):
        sampled_ids = rng.choice(machine_ids, size=len(machine_ids), replace=True)
        sampled = pd.concat(
            [eval_df[eval_df["machine_id"] == machine_id] for machine_id in sampled_ids],
            ignore_index=True,
        )
        y = sampled["fail_within_horizon"].to_numpy()
        pred = sampled["pred"].to_numpy()
        auc_samples.append(auc_score(y, pred))
        pr_samples.append(float(average_precision_score(y, pred)))
        brier_samples.append(float(np.mean((pred - y) ** 2)))
        logloss_samples.append(float(-np.mean(y * np.log(pred + 1e-9) + (1 - y) * np.log(1 - pred + 1e-9))))

    point.update(
        {
            "auc_ci_low": float(np.nanpercentile(auc_samples, 2.5)),
            "auc_ci_high": float(np.nanpercentile(auc_samples, 97.5)),
            "pr_auc_ci_low": float(np.nanpercentile(pr_samples, 2.5)),
            "pr_auc_ci_high": float(np.nanpercentile(pr_samples, 97.5)),
            "brier_ci_low": float(np.nanpercentile(brier_samples, 2.5)),
            "brier_ci_high": float(np.nanpercentile(brier_samples, 97.5)),
            "log_loss_ci_low": float(np.nanpercentile(logloss_samples, 2.5)),
            "log_loss_ci_high": float(np.nanpercentile(logloss_samples, 97.5)),
        }
    )
    return point


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


def stationarity_diagnostics(df: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
    rows = []
    for sensor in sensor_cols:
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
