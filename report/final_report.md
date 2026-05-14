# Time-Series-Driven Predictive Maintenance with NASA C-MAPSS

## Abstract

This project studies predictive maintenance as a time-series forecasting problem using the NASA C-MAPSS turbofan engine degradation benchmark. The main question is whether temporal context improves short-horizon failure-risk prediction beyond a snapshot-only baseline and whether those better risk estimates reduce maintenance cost. On the primary FD001 analysis, the nonlinear time-series forest is the strongest predictive model, while a tuned risk-threshold policy yields the lowest maintenance cost. A small robustness extension across FD002-FD004 shows that the temporal models remain competitive beyond the main subset.

## 1. Introduction

Predictive maintenance matters because both unexpected failures and unnecessarily early replacement are costly. In turbofan engines, degradation unfolds over cycles, so the problem is naturally temporal: current sensor values matter, but so do recent slopes, medium-run averages, and the contrast between short-run and long-run behavior. This project therefore asks whether explicit time-series features improve near-term failure prediction and whether that predictive gain produces lower-cost maintenance decisions.

## 2. Data and Experimental Setup

This project uses the NASA C-MAPSS turbofan engine degradation simulation dataset, specifically the FD001 subset. The raw data consist of run-to-failure engine trajectories with three operational settings and multiple sensor channels recorded over cycles.

To keep policy evaluation well defined, the analysis uses complete run-to-failure trajectories from the FD001 training file and splits engine units into train, validation, and test subsets. The supervised label at each timestamp is whether failure occurs within the next 30 cycles.

The table below summarizes the resulting dataset across splits.

| split | n_rows | n_machines | mean_failure_time | mean_path_length | positive_rate_h30 |
| --- | --- | --- | --- | --- | --- |
| train | 14520 | 70 | 207.4286 | 207.4286 | 0.1494 |
| validation | 3001 | 15 | 200.0667 | 200.0667 | 0.1549 |
| test | 3110 | 15 | 207.3333 | 207.3333 | 0.1495 |

## 3. Methods

The baseline model uses only current-cycle operating settings and sensor readings from sensor_7, sensor_11, sensor_12, sensor_15, sensor_20, sensor_21. The advanced GLM adds rolling means, rolling standard deviations, slopes, and Holt smoothing summaries. The nonlinear model further adds multi-scale lags, differences, exponentially weighted averages, and short-run versus long-run sensor contrasts. This design keeps the main comparison conceptually clean: current-cycle snapshot information versus explicit temporal context.

For the nonlinear model, I run a small validation-based RandomForest hyperparameter sweep over depth, leaf size, feature subsampling, and number of trees. This tuning is intentionally modest: the goal is to improve performance without turning the project into a large black-box search.

The decision layer compares reactive maintenance, a fixed-age replacement rule, a tuned risk-threshold rule, tabular Q-learning, and a compact DQN baseline. The RL state now includes current predicted risk, age, and the recent change in risk, while the reward discourages continuing operation when risk is already elevated.

## 4. Predictive Results

| model | auc | pr_auc | brier | log_loss |
| --- | --- | --- | --- | --- |
| nonlinear_ts_forest | 0.9962 | 0.9797 | 0.0193 | 0.0635 |
| advanced_glm | 0.9936 | 0.9680 | 0.0227 | 0.0739 |
| baseline_glm | 0.9808 | 0.9178 | 0.0371 | 0.1234 |

The best predictive model on the holdout test units is `nonlinear_ts_forest`. In this main comparison, the baseline is a current-cycle snapshot model, while the stronger models add explicit temporal context. The results therefore support the project thesis that time-series information improves short-horizon failure prediction.

To quantify uncertainty, grouped bootstrap confidence intervals by engine unit are also saved in `outputs/risk_model_metrics_with_ci.csv`.

The validation results from the RandomForest hyperparameter sweep are saved in `outputs/tree_tuning_results.csv`, and the associated plot is saved in `outputs/tree_tuning_curve.png`.

For the strongest nonlinear model, the top feature importances are shown below.

| feature | importance |
| --- | --- |
| sensor_11_mean_5 | 0.3179 |
| sensor_11_ewm_04 | 0.2485 |
| mean_sensor_level_5 | 0.0570 |
| sensor_11_slope_30 | 0.0564 |
| holt_level | 0.0370 |
| sensor_11_mean_15 | 0.0297 |
| sensor_15_mean_5 | 0.0232 |
| sensor_15_mean_15 | 0.0199 |
| sensor_12_slope_30 | 0.0186 |
| sensor_15_ewm_04 | 0.0147 |

## 5. Diagnostics

Stationarity checks were run on first-differenced representative sensor channels. The resulting ADF diagnostics are shown below.

| series | adf_stat | p_value | crit_5pct |
| --- | --- | --- | --- |
| sensor_7_first_difference | -40.8928 | 0.0000 | -2.8617 |
| sensor_12_first_difference | -48.5081 | 0.0000 | -2.8617 |
| sensor_21_first_difference | -20.7355 | 0.0000 | -2.8617 |

Calibration and residual-autocorrelation outputs are also saved in the `outputs/` directory for the presentation and reproducibility package.

## 6. Maintenance Policy Comparison

Policies are evaluated in a continuing-operation environment that repeatedly draws held-out engine trajectories from the FD001 test pool. Costs are set to 18 for preventive replacement and 95 for failure-driven replacement.

The risk-threshold rule is tuned on the validation split by sweeping thresholds and choosing the one with the lowest expected cost. The full sweep is saved in `outputs/threshold_sweep.csv`.

| policy | avg_reward | avg_cost | std_cost | avg_replacements | avg_failures |
| --- | --- | --- | --- | --- | --- |
| risk_threshold_0.30 | -414.0000 | 414.0000 | 15.2128 | 23.0000 | 0.0000 |
| age_threshold_196 | -1108.2143 | 1108.2143 | 152.2292 | 20.9286 | 9.5000 |
| q_learning | -1210.9286 | 1210.9286 | 103.4377 | 22.3571 | 10.5000 |
| reactive | -1784.6429 | 1784.6429 | 81.7108 | 18.7857 | 18.7857 |
| dqn | -1784.6429 | 1784.6429 | 73.3987 | 18.7857 | 18.7857 |

The tuned risk-threshold policy uses threshold 0.30 and is driven by the best predictive model, `nonlinear_ts_forest`.

The policy layer is included to show operational consequences of better prediction. In this project, the main insight is that once the risk estimate is highly accurate, a simple threshold controller is more valuable than adding policy complexity.

## 7. Robustness Extension

To test whether the main predictive story depends too heavily on FD001, I reran the predictive comparison on FD002, FD003, and FD004 using the same train-validation-test protocol. This extension is limited to the forecasting layer, which keeps runtime reasonable while still checking the stability of the central time-series claim.

| subset | model | auc | pr_auc | brier | log_loss |
| --- | --- | --- | --- | --- | --- |
| FD001 | baseline_glm | 0.9808 | 0.9178 | 0.0371 | 0.1234 |
| FD001 | advanced_glm | 0.9936 | 0.9680 | 0.0227 | 0.0739 |
| FD001 | nonlinear_ts_forest | 0.9962 | 0.9797 | 0.0193 | 0.0635 |
| FD002 | baseline_glm | 0.9380 | 0.7986 | 0.0598 | 0.2062 |
| FD002 | advanced_glm | 0.9823 | 0.9111 | 0.0368 | 0.1246 |
| FD002 | nonlinear_ts_forest | 0.9833 | 0.9181 | 0.0583 | 0.2144 |
| FD003 | baseline_glm | 0.9872 | 0.9133 | 0.0282 | 0.0938 |
| FD003 | advanced_glm | 0.9962 | 0.9741 | 0.0164 | 0.0520 |
| FD003 | nonlinear_ts_forest | 0.9966 | 0.9780 | 0.0148 | 0.0501 |
| FD004 | baseline_glm | 0.9161 | 0.7321 | 0.0503 | 0.1868 |
| FD004 | advanced_glm | 0.9799 | 0.8566 | 0.0370 | 0.1185 |
| FD004 | nonlinear_ts_forest | 0.9489 | 0.7213 | 0.0611 | 0.2194 |

The robustness table and `outputs/robustness_auc_by_subset.png` are designed for appendix use in the final report and slides. They show whether the temporal models remain competitive when operating conditions become more complex.

## 8. Discussion

The experiment shows that once the comparison is framed as snapshot-only versus temporal-context models, richer time-series features improve predictive quality on FD001. The biggest practical result is then operational: accurate risk estimation makes a simple threshold rule highly competitive.

## 9. Limitations and Future Work

The main limitations are that the operational comparison is still lightweight and that the robustness extension focuses on predictive metrics rather than full policy evaluation on every subset. Even after strengthening the RL state and reward, the threshold policy remains easier to tune and more effective in this setup. With more time, I would extend the policy environment to support richer partial observability, recurrent state representations, and subset-specific operating regimes.
