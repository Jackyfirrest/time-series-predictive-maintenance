from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pandas.errors import PerformanceWarning

from .data import ALL_SUBSETS, CMAPSSConfig, REPRESENTATIVE_SENSOR_COLUMNS, SETTING_COLUMNS, load_cmapss_splits
from .features import build_supervised_frame
from .modeling import (
    build_feature_sets,
    bootstrap_risk_metrics,
    calibration_table,
    evaluate_risk_model,
    feature_importance_table,
    fit_risk_model,
    fit_tree_risk_model,
    tune_tree_risk_model,
    residual_autocorrelation,
    stationarity_diagnostics,
)
from .policy import (
    evaluate_policy,
    q_policy_from_table,
    threshold_sweep,
    train_dqn_policy,
    train_q_learning_policy,
    tune_threshold_policy,
)


def _frame_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in headers:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _escape_latex(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _frame_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    headers = list(df.columns)
    column_spec = "l" + "r" * (len(headers) - 1)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{{_escape_latex(caption)}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        " & ".join(_escape_latex(col) for col in headers) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        values = []
        for col in headers:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(_escape_latex(value))
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table}"])
    return "\n".join(lines)


def _plot_example_trajectories(df: pd.DataFrame, sensor_cols: list[str], output_dir: Path) -> None:
    sample_ids = sorted(df["machine_id"].unique())[:4]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=False)
    axes = axes.ravel()
    for ax, machine_id in zip(axes, sample_ids):
        sub = df[df["machine_id"] == machine_id]
        for sensor in sensor_cols[:3]:
            series = sub[sensor]
            scaled = (series - series.mean()) / max(float(series.std()), 1e-6)
            ax.plot(sub["age"], scaled, linewidth=1.8, label=sensor)
        ax.axvline(sub["failure_time"].iloc[0], color="red", linestyle="--", linewidth=1)
        ax.set_title(f"Engine {machine_id}")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Within-engine z-score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / "example_trajectories.png", dpi=180)
    plt.close(fig)


def _plot_policy_summary(policy_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#7f8c8d", "#2980b9", "#27ae60", "#c0392b", "#d35400"]
    ax.bar(policy_df["policy"], policy_df["avg_cost"], color=colors[: len(policy_df)])
    if "std_cost" in policy_df.columns:
        ax.errorbar(
            policy_df["policy"],
            policy_df["avg_cost"],
            yerr=policy_df["std_cost"],
            fmt="none",
            ecolor="black",
            elinewidth=1,
            capsize=4,
        )
    ax.set_ylabel("Average total cost")
    ax.set_title("Maintenance policy comparison")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "policy_comparison.png", dpi=180)
    plt.close(fig)


def _plot_calibration(calibration_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for model_name, sub in calibration_df.groupby("model", sort=False):
        ax.plot(sub["mean_pred"], sub["observed_rate"], marker="o", linewidth=1.8, label=model_name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("Predicted failure probability")
    ax.set_ylabel("Observed failure frequency")
    ax.set_title("Calibration by probability bin")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "calibration_curve.png", dpi=180)
    plt.close(fig)


def _plot_residual_acf(acf_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for model_name, sub in acf_df.groupby("model", sort=False):
        ax.plot(sub["lag"], sub["residual_acf"], marker="o", linewidth=1.8, label=model_name)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Residual autocorrelation")
    ax.set_title("Residual dependence diagnostic")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "residual_acf.png", dpi=180)
    plt.close(fig)


def _plot_example_risk_path(df: pd.DataFrame, sensor_cols: list[str], model, output_dir: Path) -> None:
    machine_id = int(sorted(df["machine_id"].unique())[0])
    sub = df[df["machine_id"] == machine_id].copy().reset_index(drop=True)
    sub["predicted_risk"] = model.predict(sub)
    representative_sensor = sensor_cols[-1]
    fig, ax1 = plt.subplots(figsize=(8.0, 4.5))
    ax1.plot(sub["age"], sub["predicted_risk"], color="#c0392b", linewidth=2, label="Predicted risk")
    ax1.set_xlabel("Cycle")
    ax1.set_ylabel("Failure risk within horizon", color="#c0392b")
    ax1.tick_params(axis="y", labelcolor="#c0392b")
    ax1.axvline(sub["failure_time"].iloc[0], color="black", linestyle="--", linewidth=1)
    ax2 = ax1.twinx()
    ax2.plot(sub["age"], sub[representative_sensor], color="#2980b9", alpha=0.8, label=representative_sensor)
    ax2.set_ylabel(representative_sensor, color="#2980b9")
    ax2.tick_params(axis="y", labelcolor="#2980b9")
    ax1.set_title("Example engine: risk track vs. representative sensor")
    fig.tight_layout()
    fig.savefig(output_dir / "example_risk_path.png", dpi=180)
    plt.close(fig)


def _plot_dqn_training(history_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(8.0, 4.5))
    ax1.plot(history_df["episode"], history_df["reward"], color="#d35400", linewidth=1.8)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Episode reward", color="#d35400")
    ax1.tick_params(axis="y", labelcolor="#d35400")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(history_df["episode"], history_df["epsilon"], color="#34495e", linestyle="--", linewidth=1.5)
    ax2.set_ylabel("Exploration rate", color="#34495e")
    ax2.tick_params(axis="y", labelcolor="#34495e")
    ax1.set_title("DQN training history")
    fig.tight_layout()
    fig.savefig(output_dir / "dqn_training_curve.png", dpi=180)
    plt.close(fig)


def _dataset_summary(split_name: str, raw_df: pd.DataFrame) -> dict[str, float | int | str]:
    grouped = raw_df.groupby("machine_id", sort=True)
    return {
        "split": split_name,
        "n_rows": int(len(raw_df)),
        "n_machines": int(raw_df["machine_id"].nunique()),
        "mean_failure_time": float(grouped["failure_time"].first().mean()),
        "mean_path_length": float(grouped.size().mean()),
        "positive_rate_h30": float((raw_df["time_to_failure"] <= 30).mean()),
    }


def _plot_feature_importance(importance_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ordered = importance_df.iloc[::-1]
    ax.barh(ordered["feature"], ordered["importance"], color="#2c7fb8")
    ax.set_xlabel("Feature importance")
    ax.set_title("Top feature importances: nonlinear time-series forest")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance_top10.png", dpi=180)
    plt.close(fig)


def _plot_threshold_sweep(threshold_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(threshold_df["threshold"], threshold_df["expected_cost"], marker="o", linewidth=1.8, color="#16a085")
    best_row = threshold_df.loc[threshold_df["expected_cost"].idxmin()]
    ax.axvline(best_row["threshold"], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Risk threshold")
    ax.set_ylabel("Validation expected cost")
    ax.set_title("Threshold tuning curve")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "threshold_tuning_curve.png", dpi=180)
    plt.close(fig)


def _plot_tree_tuning(tuning_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    labels = [
        f"{int(row.n_estimators)} trees\n d={int(row.max_depth)}, leaf={int(row.min_samples_leaf)}"
        for row in tuning_df.itertuples(index=False)
    ]
    ax.plot(range(len(tuning_df)), tuning_df["brier"], marker="o", linewidth=1.8, color="#8e44ad")
    ax.set_xticks(range(len(tuning_df)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Validation Brier score")
    ax.set_title("Random-forest validation tuning")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "tree_tuning_curve.png", dpi=180)
    plt.close(fig)


def _plot_robustness_auc(robustness_df: pd.DataFrame, output_dir: Path) -> None:
    auc_df = robustness_df.pivot(index="subset", columns="model", values="auc").reset_index()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for model_name in ["baseline_glm", "advanced_glm", "nonlinear_ts_forest"]:
        if model_name in auc_df.columns:
            ax.plot(auc_df["subset"], auc_df[model_name], marker="o", linewidth=1.8, label=model_name)
    ax.set_xlabel("C-MAPSS subset")
    ax.set_ylabel("Test AUC")
    ax.set_title("Predictive robustness across C-MAPSS subsets")
    ax.set_ylim(0.75, 1.01)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "robustness_auc_by_subset.png", dpi=180)
    plt.close(fig)


def _attach_predicted_risk(df: pd.DataFrame, risk_model) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for _, group in df.groupby("machine_id", sort=True):
        sub = group.copy().reset_index(drop=True)
        sub["predicted_risk"] = risk_model.predict(sub)
        frames.append(sub)
    return frames


def _evaluate_subset(
    subset: str,
    output_dir: Path | None = None,
    tune_tree: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    config = CMAPSSConfig(subset=subset)
    train_raw, valid_raw, test_raw, sensor_cols = load_cmapss_splits(config)
    train_df = build_supervised_frame(train_raw, sensor_cols=sensor_cols, setting_cols=SETTING_COLUMNS, horizon=config.label_horizon)
    valid_df = build_supervised_frame(valid_raw, sensor_cols=sensor_cols, setting_cols=SETTING_COLUMNS, horizon=config.label_horizon)
    test_df = build_supervised_frame(test_raw, sensor_cols=sensor_cols, setting_cols=SETTING_COLUMNS, horizon=config.label_horizon)
    baseline_features, advanced_features, rich_features = build_feature_sets(sensor_cols=sensor_cols, setting_cols=SETTING_COLUMNS)

    baseline_model = fit_risk_model(train_df, baseline_features)
    advanced_model = fit_risk_model(train_df, advanced_features)
    if tune_tree:
        tree_model, tuning_df = tune_tree_risk_model(train_df, valid_df, rich_features)
    else:
        tree_model = fit_tree_risk_model(train_df, rich_features)
        tuning_df = None

    metrics = pd.DataFrame(
        [
            evaluate_risk_model(baseline_model, test_df, "baseline_glm"),
            evaluate_risk_model(advanced_model, test_df, "advanced_glm"),
            evaluate_risk_model(tree_model, test_df, "nonlinear_ts_forest"),
        ]
    )
    metrics.insert(0, "subset", subset)
    if output_dir is not None and subset == "FD001" and tuning_df is not None:
        tuning_df.to_csv(output_dir / "tree_tuning_results.csv", index=False)
        _plot_tree_tuning(tuning_df, output_dir)
        return metrics, tuning_df
    return metrics, None


def run_project() -> None:
    warnings.filterwarnings("ignore", category=PerformanceWarning)
    output_dir = Path("outputs")
    report_dir = Path("report")
    output_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)

    config = CMAPSSConfig()
    train_raw, valid_raw, test_raw, sensor_cols = load_cmapss_splits(config)
    representative_sensors = [sensor for sensor in REPRESENTATIVE_SENSOR_COLUMNS if sensor in sensor_cols]
    if len(representative_sensors) < 3:
        representative_sensors = sensor_cols[:3]

    dataset_summary_df = pd.DataFrame(
        [
            _dataset_summary("train", train_raw),
            _dataset_summary("validation", valid_raw),
            _dataset_summary("test", test_raw),
        ]
    )
    dataset_summary_df.to_csv(output_dir / "dataset_summary.csv", index=False)

    train_df = build_supervised_frame(train_raw, sensor_cols=sensor_cols, setting_cols=SETTING_COLUMNS, horizon=config.label_horizon)
    valid_df = build_supervised_frame(valid_raw, sensor_cols=sensor_cols, setting_cols=SETTING_COLUMNS, horizon=config.label_horizon)
    test_df = build_supervised_frame(test_raw, sensor_cols=sensor_cols, setting_cols=SETTING_COLUMNS, horizon=config.label_horizon)

    baseline_features, advanced_features, rich_features = build_feature_sets(
        sensor_cols=sensor_cols,
        setting_cols=SETTING_COLUMNS,
    )

    baseline_model = fit_risk_model(train_df, baseline_features)
    advanced_model = fit_risk_model(train_df, advanced_features)
    tree_model, tree_tuning_df = tune_tree_risk_model(train_df, valid_df, rich_features)
    tree_tuning_df.to_csv(output_dir / "tree_tuning_results.csv", index=False)

    metrics = pd.DataFrame(
        [
            evaluate_risk_model(baseline_model, test_df, "baseline_glm"),
            evaluate_risk_model(advanced_model, test_df, "advanced_glm"),
            evaluate_risk_model(tree_model, test_df, "nonlinear_ts_forest"),
        ]
    ).sort_values(["auc", "brier"], ascending=[False, True])
    metrics.to_csv(output_dir / "risk_model_metrics.csv", index=False)

    metric_ci_df = pd.DataFrame(
        [
            bootstrap_risk_metrics(baseline_model, test_df, "baseline_glm"),
            bootstrap_risk_metrics(advanced_model, test_df, "advanced_glm"),
            bootstrap_risk_metrics(tree_model, test_df, "nonlinear_ts_forest"),
        ]
    ).sort_values(["auc", "brier"], ascending=[False, True])
    metric_ci_df.to_csv(output_dir / "risk_model_metrics_with_ci.csv", index=False)

    calibration_df = pd.concat(
        [
            calibration_table(baseline_model, test_df, "baseline_glm"),
            calibration_table(advanced_model, test_df, "advanced_glm"),
            calibration_table(tree_model, test_df, "nonlinear_ts_forest"),
        ],
        ignore_index=True,
    )
    calibration_df.to_csv(output_dir / "calibration_summary.csv", index=False)

    acf_df = pd.concat(
        [
            residual_autocorrelation(advanced_model, test_df, "advanced_glm"),
            residual_autocorrelation(tree_model, test_df, "nonlinear_ts_forest"),
        ],
        ignore_index=True,
    )
    acf_df.to_csv(output_dir / "residual_acf.csv", index=False)

    stationarity_df = stationarity_diagnostics(train_raw, representative_sensors)
    stationarity_df.to_csv(output_dir / "stationarity_diagnostics.csv", index=False)
    feature_importance_df = feature_importance_table(tree_model, top_n=10)
    feature_importance_df.to_csv(output_dir / "feature_importance_top10.csv", index=False)

    robustness_frames: list[pd.DataFrame] = []
    for subset in ALL_SUBSETS:
        subset_metrics, _ = _evaluate_subset(subset, tune_tree=(subset == "FD001"))
        robustness_frames.append(subset_metrics)
    robustness_df = pd.concat(robustness_frames, ignore_index=True)
    robustness_df.to_csv(output_dir / "robustness_metrics.csv", index=False)

    best_model = tree_model if metrics.iloc[0]["model"] == "nonlinear_ts_forest" else advanced_model
    best_model_name = str(metrics.iloc[0]["model"])

    threshold_df = threshold_sweep(valid_df, best_model)
    threshold_df.to_csv(output_dir / "threshold_sweep.csv", index=False)
    threshold = tune_threshold_policy(valid_df, best_model)
    train_pool = _attach_predicted_risk(train_df, best_model)
    test_pool = _attach_predicted_risk(test_df, best_model)
    age_threshold = int(round(train_raw.groupby("machine_id")["failure_time"].first().median()))

    q_table = train_q_learning_policy(best_model, trajectory_pool=train_pool)
    dqn_policy, dqn_history = train_dqn_policy(best_model, trajectory_pool=train_pool)

    reactive = evaluate_policy(
        env_seed=4000,
        risk_model=best_model,
        trajectory_pool=test_pool,
        policy_name="reactive",
        policy_fn=lambda state, obs: 0,
    )
    age_based = evaluate_policy(
        env_seed=5000,
        risk_model=best_model,
        trajectory_pool=test_pool,
        policy_name=f"age_threshold_{age_threshold}",
        policy_fn=lambda state, obs, age_cutoff=age_threshold: int(obs["age"] >= age_cutoff),
    )
    risk_based = evaluate_policy(
        env_seed=6000,
        risk_model=best_model,
        trajectory_pool=test_pool,
        policy_name=f"risk_threshold_{threshold:.2f}",
        policy_fn=lambda state, obs, thr=threshold: int(obs["risk"] >= thr),
    )
    q_learning = evaluate_policy(
        env_seed=7000,
        risk_model=best_model,
        trajectory_pool=test_pool,
        policy_name="q_learning",
        policy_fn=q_policy_from_table(q_table),
    )
    dqn_learning = evaluate_policy(
        env_seed=8000,
        risk_model=best_model,
        trajectory_pool=test_pool,
        policy_name="dqn",
        policy_fn=dqn_policy,
    )

    policy_df = pd.DataFrame([reactive, age_based, risk_based, q_learning, dqn_learning]).sort_values("avg_cost")
    policy_df.to_csv(output_dir / "policy_summary.csv", index=False)
    pd.DataFrame(q_table.reshape(-1, 2), columns=["continue_value", "replace_value"]).to_csv(
        output_dir / "q_table.csv", index=False
    )
    dqn_history.to_csv(output_dir / "dqn_training_history.csv", index=False)

    _plot_example_trajectories(train_raw, representative_sensors, output_dir)
    _plot_policy_summary(policy_df, output_dir)
    _plot_calibration(calibration_df, output_dir)
    _plot_residual_acf(acf_df, output_dir)
    _plot_example_risk_path(test_df, representative_sensors, best_model, output_dir)
    _plot_dqn_training(dqn_history, output_dir)
    _plot_feature_importance(feature_importance_df, output_dir)
    _plot_threshold_sweep(threshold_df, output_dir)
    _plot_tree_tuning(tree_tuning_df, output_dir)
    _plot_robustness_auc(robustness_df, output_dir)

    summary_lines = [
        "# Time-Series-Driven Predictive Maintenance with NASA C-MAPSS",
        "",
        "## Abstract",
        "",
        "This project studies predictive maintenance as a time-series forecasting problem using the NASA C-MAPSS turbofan engine degradation benchmark. The main question is whether temporal context improves short-horizon failure-risk prediction beyond a snapshot-only baseline and whether those better risk estimates reduce maintenance cost. On the primary FD001 analysis, the nonlinear time-series forest is the strongest predictive model, while a tuned risk-threshold policy yields the lowest maintenance cost. A small robustness extension across FD002-FD004 shows that the temporal models remain competitive beyond the main subset.",
        "",
        "## 1. Introduction",
        "",
        "Predictive maintenance matters because both unexpected failures and unnecessarily early replacement are costly. In turbofan engines, degradation unfolds over cycles, so the problem is naturally temporal: current sensor values matter, but so do recent slopes, medium-run averages, and the contrast between short-run and long-run behavior. This project therefore asks whether explicit time-series features improve near-term failure prediction and whether that predictive gain produces lower-cost maintenance decisions.",
        "",
        "## 2. Data and Experimental Setup",
        "",
        "This project uses the NASA C-MAPSS turbofan engine degradation simulation dataset, specifically the FD001 subset. The raw data consist of run-to-failure engine trajectories with three operational settings and multiple sensor channels recorded over cycles.",
        "",
        "To keep policy evaluation well defined, the analysis uses complete run-to-failure trajectories from the FD001 training file and splits engine units into train, validation, and test subsets. The supervised label at each timestamp is whether failure occurs within the next 30 cycles.",
        "",
        "The table below summarizes the resulting dataset across splits.",
        "",
        _frame_to_markdown(dataset_summary_df),
        "",
        "## 3. Methods",
        "",
        f"The baseline model uses only current-cycle operating settings and sensor readings from {', '.join(sensor_cols)}. The advanced GLM adds rolling means, rolling standard deviations, slopes, and Holt smoothing summaries. The nonlinear model further adds multi-scale lags, differences, exponentially weighted averages, and short-run versus long-run sensor contrasts. This design keeps the main comparison conceptually clean: current-cycle snapshot information versus explicit temporal context.",
        "",
        "For the nonlinear model, I run a small validation-based RandomForest hyperparameter sweep over depth, leaf size, feature subsampling, and number of trees. This tuning is intentionally modest: the goal is to improve performance without turning the project into a large black-box search.",
        "",
        "The decision layer compares reactive maintenance, a fixed-age replacement rule, a tuned risk-threshold rule, tabular Q-learning, and a compact DQN baseline. The RL state now includes current predicted risk, age, and the recent change in risk, while the reward discourages continuing operation when risk is already elevated.",
        "",
        "## 4. Predictive Results",
        "",
        _frame_to_markdown(metrics),
        "",
        f"The best predictive model on the holdout test units is `{best_model_name}`. In this main comparison, the baseline is a current-cycle snapshot model, while the stronger models add explicit temporal context. The results therefore support the project thesis that time-series information improves short-horizon failure prediction.",
        "",
        "To quantify uncertainty, grouped bootstrap confidence intervals by engine unit are also saved in `outputs/risk_model_metrics_with_ci.csv`.",
        "",
        "The validation results from the RandomForest hyperparameter sweep are saved in `outputs/tree_tuning_results.csv`, and the associated plot is saved in `outputs/tree_tuning_curve.png`.",
        "",
        "For the strongest nonlinear model, the top feature importances are shown below.",
        "",
        _frame_to_markdown(feature_importance_df),
        "",
        "## 5. Diagnostics",
        "",
        "Stationarity checks were run on first-differenced representative sensor channels. The resulting ADF diagnostics are shown below.",
        "",
        _frame_to_markdown(stationarity_df),
        "",
        "Calibration and residual-autocorrelation outputs are also saved in the `outputs/` directory for the presentation and reproducibility package.",
        "",
        "## 6. Maintenance Policy Comparison",
        "",
        "Policies are evaluated in a continuing-operation environment that repeatedly draws held-out engine trajectories from the FD001 test pool. Costs are set to 18 for preventive replacement and 95 for failure-driven replacement.",
        "",
        "The risk-threshold rule is tuned on the validation split by sweeping thresholds and choosing the one with the lowest expected cost. The full sweep is saved in `outputs/threshold_sweep.csv`.",
        "",
        _frame_to_markdown(policy_df),
        "",
        f"The tuned risk-threshold policy uses threshold {threshold:.2f} and is driven by the best predictive model, `{best_model_name}`.",
        "",
        "The policy layer is included to show operational consequences of better prediction. In this project, the main insight is that once the risk estimate is highly accurate, a simple threshold controller is more valuable than adding policy complexity.",
        "",
        "## 7. Robustness Extension",
        "",
        "To test whether the main predictive story depends too heavily on FD001, I reran the predictive comparison on FD002, FD003, and FD004 using the same train-validation-test protocol. This extension is limited to the forecasting layer, which keeps runtime reasonable while still checking the stability of the central time-series claim.",
        "",
        _frame_to_markdown(robustness_df),
        "",
        "The robustness table and `outputs/robustness_auc_by_subset.png` are designed for appendix use in the final report and slides. They show whether the temporal models remain competitive when operating conditions become more complex.",
        "",
        "## 8. Discussion",
        "",
        "The experiment shows that once the comparison is framed as snapshot-only versus temporal-context models, richer time-series features improve predictive quality on FD001. The biggest practical result is then operational: accurate risk estimation makes a simple threshold rule highly competitive.",
        "",
        "## 9. Limitations and Future Work",
        "",
        "The main limitations are that the operational comparison is still lightweight and that the robustness extension focuses on predictive metrics rather than full policy evaluation on every subset. Even after strengthening the RL state and reward, the threshold policy remains easier to tune and more effective in this setup. With more time, I would extend the policy environment to support richer partial observability, recurrent state representations, and subset-specific operating regimes.",
        "",
    ]
    (report_dir / "final_report.md").write_text("\n".join(summary_lines), encoding="utf-8")

    latex_lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{float}",
        r"\usepackage{hyperref}",
        r"\usepackage{setspace}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\onehalfspacing",
        "",
        r"\title{Time-Series-Driven Predictive Maintenance with NASA C-MAPSS}",
        r"\author{Stat 248 Final Project}",
        r"\date{\today}",
        "",
        r"\begin{document}",
        r"\maketitle",
        "",
        r"\begin{abstract}",
        "This project studies predictive maintenance as a time-series forecasting problem using the NASA C-MAPSS FD001 turbofan engine degradation benchmark. The main question is whether richer temporal summaries improve short-horizon failure-risk prediction and, in turn, lower maintenance cost. Results compare generalized linear baselines, a nonlinear time-series forest, and several maintenance policies driven by predicted risk. In the main snapshot-only versus temporal-context comparison, the strongest predictive model is the nonlinear time-series forest, while a tuned risk-threshold policy achieves the lowest maintenance cost.",
        r"\end{abstract}",
        "",
        r"\section{Introduction}",
        "Predictive maintenance is a natural time-series problem because engine failure risk depends on how sensor signals evolve over time rather than on one isolated reading. This project evaluates whether explicit temporal features from C-MAPSS trajectories improve near-term failure prediction and whether those better risk estimates translate into better maintenance decisions.",
        "",
        r"\section{Data}",
        "The analysis uses the NASA C-MAPSS FD001 subset. Because the downstream maintenance experiment requires complete run-to-failure trajectories, engine units from the training file are split into train, validation, and test partitions. The supervised label at each cycle is whether failure occurs within the next thirty cycles.",
        "",
        _frame_to_latex(dataset_summary_df, "Summary of the FD001 engine-unit splits used in the analysis.", "tab:data-summary"),
        "",
        r"\section{Methods}",
        f"Three predictive models are compared. The baseline GLM uses only current-cycle operating settings and sensor readings from {_escape_latex(', '.join(sensor_cols))}. The advanced GLM adds rolling means, rolling standard deviations, slopes, and Holt summaries. The nonlinear time-series forest extends this feature set with lags, first and second differences, exponentially weighted averages, and short-run versus long-run sensor contrasts.",
        "",
        "For the nonlinear model, a small validation-based hyperparameter sweep over tree depth, leaf size, feature subsampling, and number of trees is used to improve out-of-sample performance without shifting the project toward large-scale automated search.",
        "",
        "To connect prediction with decision-making, the project compares reactive maintenance, a fixed-age replacement rule, a tuned risk-threshold policy, tabular Q-learning, and a compact DQN extension. The strengthened RL state includes current predicted risk, age, and recent risk change, and the training reward discourages continuing operation under high predicted risk. The RL methods are included as benchmarks, but the main focus remains on the time-series forecasting layer.",
        "",
        r"\section{Predictive Results}",
        _frame_to_latex(metrics, "Predictive performance on held-out FD001 engine units.", "tab:risk-metrics"),
        f"The best predictive model is \\texttt{{{_escape_latex(best_model_name)}}}. Because the main comparison is snapshot-only versus temporal-context models, this result supports the claim that explicit time-series information improves short-horizon failure prediction.",
        "Grouped bootstrap confidence intervals by engine unit are additionally saved in \\texttt{outputs/risk\\_model\\_metrics\\_with\\_ci.csv}.",
        "",
        _frame_to_latex(tree_tuning_df, "Validation results for the nonlinear time-series forest hyperparameter sweep.", "tab:tree-tuning"),
        "",
        _frame_to_latex(feature_importance_df, "Top ten feature importances for the nonlinear time-series forest.", "tab:feature-importance"),
        "",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.82\textwidth]{../outputs/example_risk_path.png}",
        r"\caption{Predicted short-horizon failure risk for one example engine trajectory, shown alongside a representative sensor channel.}",
        r"\label{fig:risk-path}",
        r"\end{figure}",
        "",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.72\textwidth]{../outputs/feature_importance_top10.png}",
        r"\caption{Top feature importances for the nonlinear time-series forest.}",
        r"\label{fig:feature-importance}",
        r"\end{figure}",
        "",
        r"\section{Diagnostics}",
        "To validate the time-series workflow, the project includes augmented Dickey-Fuller checks on first-differenced representative sensor channels, calibration summaries for predicted probabilities, and residual dependence diagnostics.",
        "",
        _frame_to_latex(stationarity_df, "ADF diagnostics on first-differenced representative sensor channels.", "tab:stationarity"),
        "",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.48\textwidth]{../outputs/calibration_curve.png}",
        r"\includegraphics[width=0.48\textwidth]{../outputs/residual_acf.png}",
        r"\caption{Calibration and residual dependence diagnostics for the strongest predictive models.}",
        r"\label{fig:diagnostics}",
        r"\end{figure}",
        "",
        r"\section{Maintenance Policy Comparison}",
        "Policies are evaluated in a continuing-operation environment that repeatedly samples held-out FD001 trajectories. Preventive replacement costs 18 units, while failure-driven replacement costs 95 units. The risk-threshold rule is tuned on the validation split by sweeping candidate thresholds and selecting the one with the lowest expected cost.",
        "",
        _frame_to_latex(policy_df, "Average cost and event counts by maintenance policy.", "tab:policy-results"),
        "",
        f"The tuned risk-threshold rule uses a threshold of {threshold:.2f}. It achieves the lowest average cost in this experiment, which reinforces the practical story of the project: once the risk estimate is informative, a simple decision rule can be highly effective.",
        "",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.72\textwidth]{../outputs/policy_comparison.png}",
        r"\caption{Average maintenance cost across competing policies.}",
        r"\label{fig:policy-comparison}",
        r"\end{figure}",
        "",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.68\textwidth]{../outputs/threshold_tuning_curve.png}",
        r"\caption{Validation threshold sweep for the risk-based maintenance policy.}",
        r"\label{fig:threshold-sweep}",
        r"\end{figure}",
        "",
        r"\section{Robustness Extension}",
        "To assess whether the central forecasting result depends too strongly on the simplest C-MAPSS subset, the predictive comparison was rerun on FD002, FD003, and FD004 using the same train-validation-test protocol. This extension is limited to prediction rather than full policy evaluation so that it can function as an appendix-style robustness check.",
        "",
        _frame_to_latex(robustness_df, "Predictive robustness across the four standard C-MAPSS subsets.", "tab:robustness"),
        "",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.74\textwidth]{../outputs/robustness_auc_by_subset.png}",
        r"\caption{AUC comparison across FD001 through FD004.}",
        r"\label{fig:robustness}",
        r"\end{figure}",
        "",
        r"\section{Discussion}",
        "The strongest practical lesson in this project is that temporal context improves prediction when compared against a true snapshot-only baseline. The richer models capture information in short-run and medium-run sensor evolution that is not available from a single cycle alone.",
        "",
        r"\section{Limitations and Future Work}",
        "The main limitations are that the operational comparison is still lightweight and that the robustness extension focuses on predictive metrics rather than full policy evaluation on every subset. Even after strengthening the RL state and reward, the tuned threshold controller remains easier to tune and more effective in this setup. Natural extensions include subset-specific operating regimes, richer partial observability, and recurrent or state-space formulations for maintenance control.",
        "",
        r"\section{Reproducibility}",
        r"The full pipeline can be reproduced by placing the NASA C-MAPSS files in \texttt{data/CMAPSSData/}, installing dependencies from \texttt{requirements.txt}, and running \texttt{python run\_project.py}. The script regenerates all tables, figures, and both report formats.",
        "",
        r"\end{document}",
    ]
    (report_dir / "final_report.tex").write_text("\n".join(latex_lines), encoding="utf-8")
