#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-python}"
"${PYTHON_BIN}" -m dual_stream_hsi_lidar.visualize_dataset \
  --dataset-dir Houston2013 \
  --output-dir dual_stream_hsi_lidar/visualizations \
  "$@"

