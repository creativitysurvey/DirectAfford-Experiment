#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Runs one real-data method. Baselines need verified official training pipelines.
python src/train.py --dataset laso --data-dir "${DATA_DIR:-./data/laso_dataset}" \
  --epochs "${EPOCHS:-300}" --output-dir "${OUTPUT_DIR:-./outputs/directafford_laso}" "$@"
