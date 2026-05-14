# Reproducible Analysis

This repository contains a reproducible version of the Stat 248 final project on time-series-driven predictive maintenance, organized so the project is complete but still straightforward to explain.

## Narrative

The analysis has three stages.

1. Simulate a fleet of machines with latent degradation and multivariate sensor streams.
2. Build short-horizon failure-risk models using time-series features.
3. Compare maintenance policies that act on those risk estimates.

The main time-series question is whether temporal summaries improve failure prediction beyond raw sensor levels. The main decision question is whether better risk estimates lead to lower maintenance cost.

For class presentation purposes, the cleanest framing is:

- time-series modeling is the main contribution
- predictive maintenance is the application
- policy learning is an extension that shows operational value

## Raw data and preprocessing

There is no external restricted dataset in this repository. Instead, the project uses a synthetic benchmark generated in [src/pdm_project/simulator.py](</c:/Users/jackyfirst/Downloads/Analysis of Time Series/time-series-driven-predictive-maintenance-with-reinforcement-learning/src/pdm_project/simulator.py>).

Preprocessing and feature construction happen in [src/pdm_project/features.py](</c:/Users/jackyfirst/Downloads/Analysis of Time Series/time-series-driven-predictive-maintenance-with-reinforcement-learning/src/pdm_project/features.py>), including:

- multi-scale rolling means
- rolling standard deviations
- rolling slopes
- lagged sensor values
- first and second differences
- exponentially weighted moving averages
- Holt level and trend summaries

These features preserve the project's time-series focus because they summarize temporal evolution rather than treating each row as a static observation.

## Models

The core model code lives in [src/pdm_project/modeling.py](</c:/Users/jackyfirst/Downloads/Analysis of Time Series/time-series-driven-predictive-maintenance-with-reinforcement-learning/src/pdm_project/modeling.py>).

The project compares:

- `baseline_glm`: current age and current sensor levels only
- `advanced_glm`: adds rolling and smoothing summaries
- `nonlinear_ts_forest`: nonlinear tree ensemble trained on richer time-series features

## Diagnostics

To satisfy the final project requirement for model validation, the repository includes:

- ADF stationarity checks on first-differenced sensor series
- calibration summaries for predicted probabilities
- residual autocorrelation diagnostics

These outputs are saved in `outputs/stationarity_diagnostics.csv`, `outputs/calibration_summary.csv`, and `outputs/residual_acf.csv`, with corresponding figures in `outputs/`.

## Policy evaluation

The maintenance decision layer is implemented in [src/pdm_project/policy.py](</c:/Users/jackyfirst/Downloads/Analysis of Time Series/time-series-driven-predictive-maintenance-with-reinforcement-learning/src/pdm_project/policy.py>). Policies include reactive maintenance, fixed-age replacement, a risk-threshold rule, tabular Q-learning, and a compact deep Q-network baseline.

This keeps the RL portion connected to the time-series part of the project: the agents do not operate on arbitrary tabular inputs, but on state summaries derived from the time-series risk model.

## Should this project use DRL?

Yes, but only in a scoped way. If the control state is only a few discretized buckets, tabular Q-learning is a strong baseline and is easy to explain. A DRL baseline is reasonable as an extension because it can:

- operate on continuous risk and age signals without manual bucketization
- better support future extensions such as partial observability and recurrent state summaries
- make the GitHub repo easier to present as an RL/DRL project instead of only a policy simulation

For that reason, the current repository includes both tabular RL and a DQN baseline. The tabular method remains the main explainable RL benchmark, while DQN is best treated as an extension or appendix item rather than the centerpiece of the project.

## How to reproduce

1. Install dependencies from `requirements.txt`, or use editable install with `pip install -e .`.
2. Run:

```bash
python run_project.py
```

3. Review the outputs in `outputs/` and the summary write-up in [report/final_report.md](</c:/Users/jackyfirst/Downloads/Analysis of Time Series/time-series-driven-predictive-maintenance-with-reinforcement-learning/report/final_report.md>).

## Main output files

- `outputs/risk_model_metrics.csv`
- `outputs/policy_summary.csv`
- `outputs/dqn_training_history.csv`
- `outputs/stationarity_diagnostics.csv`
- `outputs/calibration_summary.csv`
- `outputs/residual_acf.csv`
- `outputs/example_trajectories.png`
- `outputs/example_risk_path.png`
- `outputs/calibration_curve.png`
- `outputs/residual_acf.png`
- `outputs/policy_comparison.png`
- `outputs/dqn_training_curve.png`
