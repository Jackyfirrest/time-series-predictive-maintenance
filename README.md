# Time-Series-Driven Predictive Maintenance

This repository contains a reproducible time-series analysis of predictive maintenance. The project studies whether temporal patterns in sensor streams improve short-horizon machine failure prediction, and whether those improved risk estimates lead to better maintenance decisions.

## Project overview

The workflow has two connected parts:

1. build failure-risk models from time-series sensor data
2. compare maintenance policies that act on those risk estimates

The main analytical question is:

`Do richer time-series features improve near-term failure prediction beyond raw sensor levels alone?`

The main operational question is:

`Do better risk estimates reduce maintenance cost compared with reactive or fixed-schedule replacement?`

## Why this is a time-series problem

Machine failure is not driven by one isolated reading. It depends on how sensor signals evolve over time. In this project, vibration, temperature, and pressure become informative through their levels, local slopes, variability, lag structure, and short-run versus long-run behavior. Ignoring temporal dependence would discard much of the signal relevant to degradation and failure risk.

## Data

The repository uses a fully synthetic but reproducible benchmark generated inside the codebase. Each machine has:

- a latent health state that degrades over time
- a stressed operating regime that accelerates degradation
- three observed sensors: vibration, temperature, and pressure

A machine fails when latent health drops too low or when a high-hazard event occurs near the end of life. At each time point, the supervised target is whether failure occurs within the next 8 periods.

Using synthetic data makes the full workflow reproducible and keeps the project self-contained for review and grading.

## Methods

### Predictive models

- `baseline_glm`: age and current sensor values only
- `advanced_glm`: adds rolling means, rolling standard deviations, rolling slopes, and Holt smoothing summaries
- `nonlinear_ts_forest`: uses a richer nonlinear feature set with lags, differences, exponentially weighted averages, and multi-scale temporal summaries

### Diagnostics

The analysis includes:

- stationarity checks on first-differenced sensor series
- calibration assessment for predicted probabilities
- residual dependence diagnostics

### Decision layer

The maintenance policies compared in the repository are:

- reactive maintenance
- fixed-age preventive maintenance
- tuned risk-threshold policy
- tabular Q-learning baseline
- compact DQN baseline

The policy layer is included to connect prediction to operations. In the current results, the strongest gains still come from better time-series risk estimation rather than from a more complicated controller.

## Main findings

Across the experiments:

- the strongest predictive model is `nonlinear_ts_forest`
- the lowest-cost maintenance policy is the tuned risk-threshold rule
- richer temporal representation matters more than controller complexity

The central takeaway is that better modeling of sensor dynamics improves failure-risk estimation, and those better risk estimates translate into better maintenance decisions.

## Repository structure

- `notebooks/final_project_analysis.ipynb`: full reproducible notebook for the code report
- `src/pdm_project/`: simulation, feature engineering, modeling, diagnostics, and policy code
- `outputs/`: generated figures and summary tables
- `report/final_report.md`: Markdown report
- `report/final_report.tex`: LaTeX report source
- `report/final_report.pdf`: compiled paper-style report
- `presentation/slide_outline.md`: presentation structure
- `run_project.py`: one-command pipeline entrypoint
- `reproduce.sh`: simple shell script for end-to-end reproduction

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

## How to reproduce the analysis

Run the full pipeline with:

```bash
python run_project.py
```

or with the provided shell script:

```bash
bash reproduce.sh
```

This regenerates the main tables, figures, and report outputs.

The notebook `notebooks/final_project_analysis.ipynb` contains a full end-to-end analysis starting from synthetic raw data generation and proceeding through preprocessing, model fitting, diagnostics, visualization, and policy evaluation.

The repository is self-contained: no external restricted dataset is required, because the synthetic benchmark is generated directly in the codebase.

## Important outputs

- `outputs/risk_model_metrics.csv`
- `outputs/dataset_summary.csv`
- `outputs/feature_importance_top10.csv`
- `outputs/policy_summary.csv`
- `outputs/stationarity_diagnostics.csv`
- `outputs/calibration_summary.csv`
- `outputs/residual_acf.csv`
- `outputs/example_trajectories.png`
- `outputs/example_risk_path.png`
- `outputs/calibration_curve.png`
- `outputs/feature_importance_top10.png`
- `outputs/policy_comparison.png`
- `report/final_report.pdf`

## Recommended reading order

If you are reviewing the project for the first time, a good order is:

1. read this `README.md`
2. open `notebooks/final_project_analysis.ipynb`
3. inspect `outputs/` for figures and summary tables
4. read `report/final_report.pdf` for the paper-style write-up
