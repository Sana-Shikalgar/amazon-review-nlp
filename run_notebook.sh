#!/bin/bash
set -e
cd "$(dirname "$0")"
git pull origin feat/validation
echo "[run_notebook.sh] Starting nbconvert execution..."
nohup jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=7200 \
  notebook/Feature_Validation_And_Generation.ipynb \
  > /tmp/nbconvert_validation.log 2>&1 &
echo "[run_notebook.sh] Launched PID $!"
echo "Tail log with: tail -f /tmp/nbconvert_validation.log"
