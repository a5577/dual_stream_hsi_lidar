#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-/home/temp/miniforge3/envs/cnn/bin/python}"
CHECKPOINT="${CHECKPOINT:-dual_stream_hsi_lidar/runs/attention_fusion/best_model.pt}"

"${PYTHON_BIN}" -m dual_stream_hsi_lidar.test_attention \
  --checkpoint "${CHECKPOINT}" \
  --dataset-dir Houston2013 \
  --workers 0 \
  --save-attention \
  "$@"
