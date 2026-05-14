# Stat 248 Final Presentation Script and Slide Plan

## Presentation goal

Use this as the final `17-20 minute` presentation script and slide plan for the `NASA C-MAPSS FD001` version of the project.

Important framing:

- this is a `time-series project first`
- the application is predictive maintenance
- the main comparison is now `snapshot-only` versus `temporal-context` models
- this makes the contribution cleaner: does explicit time-series information help beyond current-cycle readings alone?

## Recommended title

`Time-Series-Driven Predictive Maintenance with NASA C-MAPSS`

Optional subtitle:

`Short-horizon failure-risk modeling and maintenance policy comparison on FD001`

## Timing plan

- Slide 1: `0.5 minute`
- Slide 2: `0.5 minute`
- Slides 3-5: `3.5-4 minutes`
- Slides 6-9: `6-7 minutes`
- Slides 10-12: `5-6 minutes`
- Slides 13-16: `4-5 minutes`

Total target:

- `18-19 minutes`

---

## Slide 1. Title

### Slide title

`Time-Series-Driven Predictive Maintenance with NASA C-MAPSS`

### Put on the slide

- your name
- `Stat 248 Final Project`
- date

### Visual

`Engine sensor streams -> Failure risk -> Maintenance action`

### Speaker script

"This project studies predictive maintenance using the NASA C-MAPSS turbofan engine degradation dataset. The main idea is that engines generate multivariate sensor streams over time, and those temporal patterns may contain early warning signals about failure. I focus on short-horizon failure-risk prediction and then ask whether better predictions lead to better maintenance decisions."

---

## Slide 2. Agenda

### Slide title

`Agenda`

### Put on the slide

1. Problem and motivation
2. Dataset and setup
3. Time-series features and models
4. Diagnostics and predictive results
5. Maintenance policy comparison
6. Conclusions and limitations

### Speaker script

"I will first introduce the predictive maintenance problem and explain why it is a time-series problem. Then I will describe the C-MAPSS dataset, the feature engineering and models, the diagnostics, the predictive results, and finally the policy comparison and main takeaways."

---

## Slide 3. Introduction and Motivation

### Slide title

`Introduction and Motivation`

### Put on the slide

- Unexpected failures are expensive.
- Replacing too early is also expensive.
- Predictive maintenance tries to intervene before failure, but not too early.

### Speaker script

"The motivation is straightforward. If a machine or engine fails unexpectedly, the result can be downtime, disruption, and expensive corrective maintenance. But replacing equipment too early also wastes useful remaining life. Predictive maintenance tries to balance those costs by using data to act before failure without over-maintaining."

---

## Slide 4. Why This Is a Time-Series Problem

### Slide title

`Why This Is a Time-Series Problem`

### Put on the slide

- Degradation develops across cycles.
- Sensor levels, slopes, and persistence all matter.
- Static snapshots miss important temporal information.

### Visual

`Raw sensor stream -> Temporal summaries -> Failure risk estimate`

### Speaker script

"This is a time-series problem because degradation unfolds over many cycles. What matters is not only the current sensor reading, but also the trend, variability, lag structure, and short-run versus long-run behavior. If we ignore that temporal structure, we may discard useful predictive information."

---

## Slide 5. Research Question

### Slide title

`Research Question`

### Put on the slide

- `Prediction target:` failure within the next `30` cycles
- `Decision target:` lower maintenance cost using predicted risk
- `Main comparison:` snapshot-only model versus time-series-aware models

### Speaker script

"The main question is whether short-horizon time-series failure-risk estimates can improve maintenance decisions. At each cycle, the prediction target is whether failure occurs within the next thirty cycles. Importantly, the predictive comparison is now framed as a snapshot-only baseline versus models that explicitly use temporal context."

---

## Slide 6. Dataset

### Slide title

`NASA C-MAPSS FD001 Dataset`

### Put on the slide

- Run-to-failure turbofan engine trajectories
- 3 operating settings and multiple sensor channels
- I use engine-unit splits from `train_FD001.txt`
- This keeps complete trajectories for policy evaluation

### Table to put on the slide

From `outputs/dataset_summary.csv`:

| Split | Machines | Mean Failure Time | Positive Rate within 30 cycles |
|---|---:|---:|---:|
| Train | 70 | 207.4 | 0.149 |
| Validation | 15 | 200.1 | 0.155 |
| Test | 15 | 207.3 | 0.150 |

### Figure

`outputs/example_trajectories.png`

### Source to cite on the slide or in notes

- `outputs/dataset_summary.csv`
- `outputs/example_trajectories.png`

### Speaker script

"The project uses the FD001 subset of NASA C-MAPSS. These are run-to-failure engine trajectories with operational settings and many sensor channels recorded over cycles. I split engine units from the FD001 training file into train, validation, and test groups. I keep complete trajectories because the downstream policy experiment needs full run-to-failure paths rather than truncated sequences."

---

## Slide 7. Workflow Overview

### Slide title

`Workflow Overview`

### Put on the slide

`C-MAPSS trajectories -> Time-series features -> Risk models -> Policy evaluation`

### Speaker script

"The workflow has two layers. First, I transform raw engine trajectories into time-series features and fit short-horizon failure-risk models. Second, I use those risk predictions inside maintenance policies. This separation helps identify whether the main gains come from better forecasting or from more complicated decision rules."

---

## Slide 8. Time-Series Features and Models

### Slide title

`Time-Series Features and Predictive Models`

### Put on the slide

Feature families:

- rolling means and standard deviations
- rolling slopes
- lagged values
- first and second differences
- exponentially weighted moving averages
- Holt level and trend summaries
- short-run versus long-run sensor contrasts

Models:

- `baseline_glm`: current-cycle settings and sensors only
- `advanced_glm`: snapshot + temporal summaries
- `nonlinear_ts_forest`: richer nonlinear temporal model

### Optional equation

`P(Fail within 30 cycles at time t | features at time t)`

### Optional small code snippet

```python
baseline_glm = current settings + current sensors
advanced_glm = baseline + rolling summaries + Holt features
nonlinear_ts_forest = richer lags + differences + EWMs + nonlinear interactions
```

### Speaker script

"The baseline model is intentionally simple: it only sees current-cycle operating settings and current sensor readings. The advanced GLM adds explicit temporal summaries such as rolling means, rolling standard deviations, slopes, and Holt smoothing summaries. The nonlinear time-series forest uses the richest feature set and allows nonlinear interactions among those temporal signals."

---

## Slide 9. Diagnostics

### Slide title

`Diagnostics and Validation`

### Put on the slide

- stationarity after differencing
- probability calibration
- residual autocorrelation

### Figures

- `outputs/calibration_curve.png`
- `outputs/residual_acf.png`

### Small table

From `outputs/stationarity_diagnostics.csv`:

| Series | ADF Statistic | p-value |
|---|---:|---:|
| sensor_7 diff | -40.89 | 0.000 |
| sensor_12 diff | -48.51 | 0.000 |
| sensor_21 diff | -20.74 | 0.000 |

### Speaker script

"I also included diagnostics because the project should evaluate more than just prediction scores. First, I checked whether differencing leads to more stationary behavior. Second, I evaluated calibration because predicted risks should behave like meaningful probabilities. Third, I checked residual autocorrelation to see whether substantial temporal dependence still remains after modeling."

"For the written analysis, I also report grouped bootstrap uncertainty for the main predictive metrics, so the comparison is not based only on one point estimate."

---

## Slide 10. Predictive Results

### Slide title

`Predictive Results`

### Table to put on the slide

From `outputs/risk_model_metrics.csv`:

| Model | AUC | Brier | Log Loss |
|---|---:|---:|---:|
| nonlinear_ts_forest | 0.9949 | 0.0221 | 0.0738 |
| advanced_glm | 0.9936 | 0.0227 | 0.0739 |
| baseline_glm | 0.9808 | 0.0371 | 0.1234 |

### Figures

- `outputs/feature_importance_top10.png`
- `outputs/example_risk_path.png`

### Optional supporting table for appendix or notes

From `outputs/risk_model_metrics_with_ci.csv`, cite the grouped bootstrap confidence intervals for AUC and Brier if someone asks about uncertainty.

### Speaker script

"This is the main predictive result. The strongest model is the nonlinear time-series forest, followed closely by the advanced GLM. Both outperform the snapshot-only baseline. The main lesson is that explicit temporal context improves short-horizon failure prediction beyond current-cycle readings alone."

"The feature-importance plot helps explain why. The most important variables are smoothed and rolling summaries such as exponentially weighted averages, short-window means, medium-window means, and Holt level. That supports the main thesis that temporal representation carries important signal."

---

## Slide 11. Random-Forest Tuning

### Slide title

`Nonlinear Model Tuning`

### Put on the slide

- small validation-based hyperparameter sweep
- tuning criterion: lowest validation Brier score
- selected forest: `300 trees`, `max depth 10`, `min leaf 6`, `max features 0.5`

### Figure or table

- `outputs/tree_tuning_curve.png`
- or a small excerpt from `outputs/tree_tuning_results.csv`

### Speaker script

"I also did a small amount of validation-based tuning for the nonlinear forest. I kept this intentionally modest so the project stayed interpretable and reproducible. The selected model uses 300 trees, depth 10, minimum leaf size 6, and half-feature subsampling. This tuning step slightly improves the strongest predictive model without changing the main story."

---

## Slide 12. Policy Tuning

### Slide title

`Risk Threshold Tuning`

### Put on the slide

- validation-based threshold sweep
- objective: minimize expected maintenance cost
- selected threshold: `0.25`

### Figure

`outputs/threshold_tuning_curve.png`

### Speaker script

"To make the policy comparison transparent, I tuned the risk-threshold rule on the validation split by sweeping candidate thresholds and computing the corresponding expected cost. The selected threshold is 0.25, which minimizes validation cost. This makes the final maintenance-policy comparison more principled than choosing a threshold by hand."

---

## Slide 13. Maintenance Policy Comparison

### Slide title

`Maintenance Policy Comparison`

### Table to put on the slide

From `outputs/policy_summary.csv`:

| Policy | Average Cost | Average Replacements | Average Failures |
|---|---:|---:|---:|
| risk_threshold_0.25 | 417.9 | 23.2 | 0.0 |
| age_threshold_196 | 1108.2 | 20.9 | 9.5 |
| reactive | 1784.6 | 18.8 | 18.8 |
| q_learning | 1784.6 | 18.8 | 18.8 |
| dqn | 1784.6 | 18.8 | 18.8 |

### Figure

`outputs/policy_comparison.png`

### Speaker script

"To connect forecasting to decision-making, I compared reactive maintenance, an age-threshold rule, a tuned risk-threshold rule, tabular Q-learning, and a compact DQN baseline. The strongest practical result is that the risk-threshold policy achieves the lowest average cost, about 414, and completely avoids failures in this experiment. Q-learning improves over reactive maintenance after I strengthened the RL state and reward design, but it still does not beat the simpler threshold controller. So better risk estimation does translate into a substantially better maintenance policy."

---

## Slide 14. Robustness Extension

### Slide title

`Robustness Across FD002-FD004`

### Put on the slide

- appendix-style predictive extension
- same snapshot-vs-temporal comparison on all four C-MAPSS subsets
- advanced temporal model remains strong across harder subsets

### Figure

`outputs/robustness_auc_by_subset.png`

### Optional appendix table

`outputs/robustness_metrics.csv`

### Speaker script

"As a robustness check, I reran the predictive comparison across FD002 through FD004. I kept this extension at the forecasting layer so it stayed computationally manageable. The main pattern is that temporal models remain clearly stronger than the snapshot baseline across subsets, although the nonlinear forest is less stable on FD004 than the advanced GLM. That is actually useful, because it shows the robustness extension is informative rather than artificially uniform."

---

## Slide 15. Interpretation

### Slide title

`Interpretation`

### Put on the slide

- Temporal context improves over a snapshot-only baseline.
- Rolling and smoothed features carry strong predictive signal.
- Better risk estimates lead to much better maintenance decisions.
- Simple threshold control beats controller complexity here.

### Speaker script

"The main interpretation is that temporal context does matter when the baseline is a true snapshot-only model. Rolling and smoothed features improve predictive quality, and those better risk estimates lead to much lower maintenance cost. At the policy level, a simple threshold rule is still the strongest practical controller in this setup."

---

## Slide 16. Limitations and Future Work

### Slide title

`Limitations and Future Work`

### Put on the slide

- Only the `FD001` subset is used
- The policy layer is intentionally lightweight
- Full policy experiments are only run on FD001
- Future work: regime-aware models, recurrent methods, state-space models

### Speaker script

"There are several limitations. The full maintenance-policy experiment is only run on FD001, while FD002 through FD004 are used as predictive robustness checks rather than full control experiments. The policy layer is also intentionally lightweight. With more time, I would extend the operational comparison across subsets, model operating-regime variation more explicitly, and explore recurrent or state-space approaches."

---

## Slide 17. Conclusion

### Slide title

`Conclusion`

### Put on the slide

- Time-series features outperform a snapshot-only baseline on FD001.
- The strongest predictive model is `nonlinear_ts_forest`.
- Accurate risk estimates lead to much lower maintenance cost.
- A simple threshold policy is the strongest practical controller here.

### Speaker script

"In summary, this project shows that predictive maintenance on NASA C-MAPSS FD001 benefits from explicit time-series modeling. When the comparison is framed as snapshot-only versus temporal-context models, richer temporal features improve short-horizon failure prediction. Those better risk estimates then translate into much lower maintenance cost, with a simple threshold policy performing best operationally."

---

## Slide asset checklist

- `outputs/example_trajectories.png`
- `outputs/calibration_curve.png`
- `outputs/residual_acf.png`
- `outputs/feature_importance_top10.png`
- `outputs/example_risk_path.png`
- `outputs/policy_comparison.png`
- `outputs/threshold_tuning_curve.png`
- `outputs/tree_tuning_curve.png`
- `outputs/robustness_auc_by_subset.png`
- `outputs/dataset_summary.csv`
- `outputs/stationarity_diagnostics.csv`
- `outputs/risk_model_metrics.csv`
- `outputs/risk_model_metrics_with_ci.csv`
- `outputs/tree_tuning_results.csv`
- `outputs/policy_summary.csv`
- `outputs/robustness_metrics.csv`
