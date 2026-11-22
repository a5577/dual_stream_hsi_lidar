#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-python}"

run_experiment() {
  local mode="$1"
  local output_dir="$2"
  "${PYTHON_BIN}" -m dual_stream_hsi_lidar.train \
    --dataset-dir Houston2013 \
    --output-dir "${output_dir}" \
    --fusion-mode "${mode}" \
    --workers 0
}

run_experiment hsi dual_stream_hsi_lidar/runs/hsi_only
run_experiment lidar dual_stream_hsi_lidar/runs/lidar_only
run_experiment fusion dual_stream_hsi_lidar/runs/fusion
