# Time-Series-Driven Predictive Maintenance

This repository contains a reproducible time-series analysis of predictive maintenance using the `NASA C-MAPSS FD001` turbofan engine degradation dataset. The project studies whether temporal patterns in engine sensor streams improve short-horizon failure prediction, and whether those improved risk estimates lead to better maintenance decisions.

## Project overview

The workflow has two connected parts:

1. build failure-risk models from multivariate engine time series
2. compare maintenance policies that act on those risk estimates

The main analytical question is:

`Do richer time-series features improve near-term failure prediction beyond current-cycle sensor readings alone?`

The main operational question is:

`Do better risk estimates reduce maintenance cost compared with reactive or fixed-schedule replacement?`

## Why this is a time-series problem

Engine degradation is not driven by one isolated reading. It depends on how sensor channels evolve over time through trends, variability, lag structure, and short-run versus long-run behavior. Ignoring temporal dependence would discard much of the signal relevant to degradation and failure risk.

## Data

The repository uses the `NASA Turbofan Engine Degradation Simulation Dataset (C-MAPSS)`, specifically the `FD001` subset.

- source file expected by the code: `train_FD001.txt`
- expected location: `data/CMAPSSData/train_FD001.txt`
- operating variables: `setting_1`, `setting_2`, `setting_3`
- representative sensor channels used in plots and diagnostics include `sensor_7`, `sensor_12`, and `sensor_21`

For this project, engine units from the FD001 training trajectories are split by `machine_id` into train, validation, and test subsets. This design keeps the evaluation fully reproducible and preserves complete run-to-failure trajectories for the downstream maintenance-policy experiment.

At each cycle, the supervised target is whether failure occurs within the next `30` cycles.

## Data access instructions

Place the NASA C-MAPSS files under:

```text
data/CMAPSSData/
```

At minimum, this project expects:

```text
data/CMAPSSData/train_FD001.txt
```

The repository includes the C-MAPSS text files used in the analysis under `data/CMAPSSData/`, so course staff can inspect the raw data directly. The same directory layout is still documented in case the project is copied into a fresh environment.

See [data/README.md](data/README.md) for the expected directory layout.

## Methods

### Predictive models

- `baseline_glm`: current-cycle operating settings and current sensor values only
- `advanced_glm`: adds rolling means, rolling standard deviations, rolling slopes, and Holt smoothing summaries
- `nonlinear_ts_forest`: uses a richer nonlinear feature set with lags, differences, exponentially weighted averages, and multi-scale temporal summaries

### Diagnostics

The analysis includes:

- stationarity checks on first-differenced representative sensor channels
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

Across the current FD001 experiments:

- the strongest predictive model is `nonlinear_ts_forest`
- the advanced GLM is a close second
- the lowest-cost maintenance policy is the tuned risk-threshold rule

The central takeaway is that temporal context improves prediction when the comparison is framed as `snapshot-only` versus `time-series-aware` models. On this setup, rolling summaries, smoothing, and lag-based features outperform a model that only sees the current cycle.

## Repository structure

- `data/README.md`: expected C-MAPSS file layout
- `notebooks/final_project_analysis.ipynb`: full narrative notebook with reproducible analysis and displayed results
- `src/pdm_project/`: data loading, feature engineering, modeling, diagnostics, and policy code
- `outputs/`: generated figures and summary tables
- `report/final_report.md`: Markdown report
- `report/final_report.tex`: LaTeX report source
- `presentation/presentation_script.md`: presentation script and slide plan
- `run_project.py`: one-command pipeline entrypoint
- `reproduce.sh`: simple shell script for end-to-end reproduction

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

## How to reproduce the analysis

1. Confirm the NASA C-MAPSS files are present in `data/CMAPSSData/`.
2. Verify that `train_FD001.txt` is available in that directory.
3. Run:

```bash
python run_project.py
```

or:

```bash
bash reproduce.sh
```

This regenerates the main tables, figures, and report outputs.

If the data file is missing, the pipeline will raise a clear error telling you where to place it.

## Important outputs

- `outputs/risk_model_metrics.csv`
- `outputs/dataset_summary.csv`
- `outputs/feature_importance_top10.csv`
- `outputs/policy_summary.csv`
- `outputs/tree_tuning_results.csv`
- `outputs/robustness_metrics.csv`
- `outputs/stationarity_diagnostics.csv`
- `outputs/calibration_summary.csv`
- `outputs/residual_acf.csv`
- `outputs/example_trajectories.png`
- `outputs/example_risk_path.png`
- `outputs/calibration_curve.png`
- `outputs/feature_importance_top10.png`
- `outputs/policy_comparison.png`
- `outputs/tree_tuning_curve.png`
- `outputs/robustness_auc_by_subset.png`
- `report/final_report.md`
- `report/final_report.tex`

## Recommended reading order

If you are reviewing the project for the first time, a good order is:

1. read this `README.md`
2. confirm the raw C-MAPSS files are in `data/CMAPSSData/`
3. run `python run_project.py`
4. inspect `outputs/` for figures and summary tables
5. read `report/final_report.md` for the paper-style write-up
