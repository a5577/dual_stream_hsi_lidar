#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-/home/temp/miniforge3/envs/cnn/bin/python}"
"${PYTHON_BIN}" -m dual_stream_hsi_lidar.train_attention \
  --dataset-dir Houston2013 \
  --output-dir dual_stream_hsi_lidar/runs/attention_fusion \
  --workers 0 \
  "$@"
