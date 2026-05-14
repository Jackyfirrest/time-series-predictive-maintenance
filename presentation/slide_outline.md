# Final Presentation Outline

## 1. Title and motivation

- Project title: Time-Series-Driven Predictive Maintenance
- Core question: can time-series failure risk estimation improve maintenance decisions?
- Why it matters: unplanned machine failures are costly, but premature maintenance is also expensive

## 2. Predictive maintenance as a time-series problem

- Failure is driven by degradation over time, not one isolated reading
- Sensor trajectories carry signal through level, slope, volatility, and short-vs-long-run divergence
- This motivates explicit time-series feature engineering and diagnostics

## 3. Data-generating process

- Synthetic fleet of machines with latent health state
- Stressed operating regime increases degradation rate
- Observed sensors: vibration, temperature, pressure
- Label: whether failure occurs within the next 8 time steps

Visuals:
- `outputs/example_trajectories.png`

## 4. Methods overview

- Baseline GLM with age + raw sensors
- Advanced GLM with rolling summaries and Holt state features
- Nonlinear time-series forest with richer temporal features
- Policy layer: reactive, age-threshold, risk-threshold, tabular Q-learning
- Mention DQN only briefly as an optional extension if time allows

Visuals:
- one simple flow diagram: sensors -> risk model -> maintenance policy

## 5. Time-series features

- Rolling means, rolling standard deviations, rolling slopes
- Lag features
- First and second differences
- Exponentially weighted moving averages
- Short-run versus long-run stress features

Talking point:
- emphasize that the strongest gains come from better temporal representation

## 6. Diagnostics

- ADF checks on first-differenced sensor series
- Calibration curves
- Residual autocorrelation

Visuals:
- `outputs/calibration_curve.png`
- `outputs/residual_acf.png`

## 7. Predictive results

- Show the risk model comparison table
- Highlight that `nonlinear_ts_forest` is best on AUC, Brier, and log loss

Visuals:
- `outputs/risk_model_metrics.csv` as a slide table
- `outputs/example_risk_path.png`

## 8. Policy results

- Show the policy comparison table
- Main message: risk-threshold policy dramatically lowers cost relative to reactive maintenance
- Q-learning helps, but stronger time-series modeling matters more than a more complicated controller
- Keep DQN off the main slide unless someone asks about extensions

Visuals:
- `outputs/policy_comparison.png`
- `outputs/policy_summary.csv` as a slide table

## 9. Interpretation

- Time-series structure is the key contribution
- Better risk estimates translate into better maintenance decisions
- RL is useful, but not the main performance bottleneck in this project

## 10. Limitations

- Synthetic data rather than a real industrial benchmark
- RL state is intentionally simple because the project prioritizes explainability
- No explicit latent-state Bayesian model yet

## 11. Future work

- Apply to NASA C-MAPSS or another real benchmark
- Try HMM or state-space models
- Upgrade the decision layer only after the time-series state becomes richer
- Move DQN or other DRL methods here if you want a simpler presentation

## 12. Closing slide

- One-sentence takeaway:
- In predictive maintenance, the biggest gains came from modeling how sensor behavior evolves over time, not just from using a more complex control algorithm.

## Suggested timing

- Slides 1-3: 4 minutes
- Slides 4-6: 5 minutes
- Slides 7-8: 5 minutes
- Slides 9-12: 4 to 5 minutes
