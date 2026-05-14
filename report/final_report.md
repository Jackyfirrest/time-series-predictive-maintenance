# Final Report Draft

## Project Question

Can short-horizon time-series failure risk estimates improve maintenance decisions compared with reactive or fixed-schedule maintenance?

## Data and Setup

This project uses a synthetic fleet of degrading machines. Each machine has a latent health process, a stressed operating regime, and three observed sensor streams: vibration, temperature, and pressure. A machine fails when latent health becomes too low or when a high hazard event occurs near end of life.

The data are split into train, validation, and test fleets. For each timestamp, the supervised label is whether failure occurs within the next 8 periods.

The table below summarizes the generated dataset across splits.

| split | n_rows | n_machines | mean_failure_time | mean_path_length | vibration_min | vibration_max | temperature_min | temperature_max | pressure_min | pressure_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 4726 | 90 | 51.5111 | 52.5111 | 0.2684 | 1.6225 | 57.0486 | 75.6074 | 96.5649 | 106.9888 |
| validation | 1723 | 35 | 48.2286 | 49.2286 | 0.2871 | 1.6209 | 58.5078 | 75.6872 | 96.9107 | 106.9589 |
| test | 1941 | 35 | 54.4571 | 55.4571 | 0.3233 | 1.5428 | 58.0908 | 75.3467 | 96.9284 | 107.4286 |

## Time-Series Features

The baseline model uses only current age and raw sensors. The advanced GLM adds rolling means, rolling standard deviations, rolling slopes, and an online Holt level/trend representation of the vibration series. The strongest nonlinear model is a tree-ensemble hazard model built on richer time-series features: multi-scale rolling summaries, lagged sensors, first and second differences, exponentially weighted moving averages, and stress-contrast features that compare short-run behavior with longer-run baselines.

## Predictive Results

| model | auc | brier | log_loss |
| --- | --- | --- | --- |
| nonlinear_ts_forest | 0.8821 | 0.0922 | 0.2972 |
| advanced_glm | 0.8790 | 0.0935 | 0.2997 |
| baseline_glm | 0.8732 | 0.0949 | 0.3045 |

The best predictive model on the test set is `nonlinear_ts_forest`. This improvement is driven by explicitly modeling temporal dependence rather than treating each timestamp as an isolated cross-sectional observation.

For the strongest nonlinear model, the top feature importances are shown below.

| feature | importance |
| --- | --- |
| vibration_mean_15 | 0.0932 |
| vibration_mean_8 | 0.0728 |
| vibration_ewm_04 | 0.0627 |
| pressure_ewm_04 | 0.0518 |
| holt_level | 0.0517 |
| pressure_mean_15 | 0.0480 |
| pressure_mean_8 | 0.0468 |
| health_proxy | 0.0412 |
| pressure_mean_3 | 0.0392 |
| vibration | 0.0329 |

## Diagnostics

Stationarity checks were run on first-differenced sensor series. The resulting ADF diagnostics are shown below.

| series | adf_stat | p_value | crit_5pct |
| --- | --- | --- | --- |
| vibration_first_difference | -19.0789 | 0.0000 | -2.8622 |
| temperature_first_difference | -18.2645 | 0.0000 | -2.8622 |
| pressure_first_difference | -22.7867 | 0.0000 | -2.8622 |

Calibration and residual-autocorrelation outputs are also saved in the `outputs/` directory for the presentation and reproducibility package.

## Maintenance Policies

Policies are evaluated in a continuing-operation environment where each replacement starts a fresh machine. Costs are set to 18 for preventive replacement and 95 for failure-driven replacement.

| policy | avg_reward | avg_cost | avg_replacements | avg_failures |
| --- | --- | --- | --- | --- |
| risk_threshold_0.25 | -77.6429 | 77.6429 | 2.7857 | 0.3571 |
| q_learning | -100.4286 | 100.4286 | 4.3571 | 0.2857 |
| age_threshold_75 | -200.6429 | 200.6429 | 2.2857 | 2.0714 |
| reactive | -203.5714 | 203.5714 | 2.1429 | 2.1429 |
| dqn | -223.9286 | 223.9286 | 2.3571 | 2.3571 |

The tuned risk-threshold policy uses threshold 0.25 and is driven by the best predictive model, `nonlinear_ts_forest`.

The policy layer is included to show operational consequences of better prediction. In this project, the main insight is that stronger time-series risk estimation matters more than using a more complicated controller.

## Interpretation

The experiment supports the proposal's core claim: time-series information helps convert noisy sensors into operationally meaningful failure risk. The largest gains come from improving temporal representation. Once the risk signal is strong, even a simple threshold policy can be highly competitive.

## Extensions

Possible next steps include partially observed RL with recurrent state summaries, Bayesian hazard models, and domain adaptation to real predictive-maintenance benchmarks such as C-MAPSS. A compact DQN baseline is included in the repository as an extension, but it is not the main result of the project.
