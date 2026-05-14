# Reproducibility Guide

This project is designed to be reproducible from a clean environment with only the Python dependencies listed in `requirements.txt`.

## Environment

- Python `3.11+` is recommended
- all required packages are listed in `requirements.txt`
- the experiment uses fixed random seeds inside the pipeline

## Install

```bash
pip install -r requirements.txt
```

Optional editable install:

```bash
pip install -e .
```

## Run the full project

```bash
python run_project.py
```

This command will regenerate:

- `outputs/*.csv`
- `outputs/*.png`
- `report/final_report.md`
- `report/final_report.tex`

## Main files to inspect

- `outputs/risk_model_metrics.csv`
- `outputs/policy_summary.csv`
- `outputs/stationarity_diagnostics.csv`
- `outputs/calibration_summary.csv`
- `outputs/residual_acf.csv`
- `report/final_report.md`
- `report/final_report.tex`

## Expected interpretation

- the strongest predictive model should be `nonlinear_ts_forest`
- the strongest simple policy should be the tuned `risk_threshold` policy
- the main conclusion should remain that time-series feature quality matters more than controller complexity

## Optional LaTeX compilation

If a LaTeX distribution is installed locally, compile the paper-style report from the `report/` directory:

```bash
pdflatex final_report.tex
```

The report references figures from `../outputs/`, so keep the folder structure unchanged.
