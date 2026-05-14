#!/usr/bin/env bash
set -euo pipefail

echo "Installing dependencies..."
python -m pip install -r requirements.txt

echo "Running full project pipeline..."
python run_project.py

echo "Done."
echo "Main outputs:"
echo "  - outputs/risk_model_metrics.csv"
echo "  - outputs/policy_summary.csv"
echo "  - outputs/dataset_summary.csv"
echo "  - outputs/feature_importance_top10.csv"
echo "  - report/final_report.pdf"
