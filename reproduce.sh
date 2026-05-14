#!/usr/bin/env bash
set -euo pipefail

echo "Installing dependencies..."
python -m pip install -r requirements.txt

echo "Checking for NASA C-MAPSS FD001 data..."
if [ ! -f data/CMAPSSData/train_FD001.txt ]; then
  echo "Missing data/CMAPSSData/train_FD001.txt"
  echo "See data/README.md for the expected file layout."
  exit 1
fi

echo "Running full project pipeline..."
python run_project.py

echo "Compiling LaTeX report..."
(
  cd report
  pdflatex -interaction=nonstopmode final_report.tex >/dev/null
  pdflatex -interaction=nonstopmode final_report.tex >/dev/null
)

echo "Done."
echo "Main outputs:"
echo "  - outputs/risk_model_metrics.csv"
echo "  - outputs/policy_summary.csv"
echo "  - outputs/dataset_summary.csv"
echo "  - outputs/feature_importance_top10.csv"
echo "  - report/final_report.pdf"
