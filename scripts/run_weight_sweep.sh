#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-/home/temp/miniforge3/envs/cnn/bin/python}"
RUN_ROOT="${RUN_ROOT:-dual_stream_hsi_lidar/runs/weight_sweep}"
COMPARISON_DIR="${COMPARISON_DIR:-dual_stream_hsi_lidar/runs/weight_sweep_comparison}"

run_weight() {
  local hsi_weight="$1"
  local lidar_weight="$2"
  local run_name="$3"
  local output_dir="${RUN_ROOT}/${run_name}"

  mkdir -p "${output_dir}"
  echo "Running HSI:LiDAR = ${hsi_weight}:${lidar_weight} -> ${output_dir}"
  "${PYTHON_BIN}" -m dual_stream_hsi_lidar.train \
    --dataset-dir Houston2013 \
    --output-dir "${output_dir}" \
    --fusion-mode fusion \
    --hsi-weight "${hsi_weight}" \
    --lidar-weight "${lidar_weight}" \
    --epochs 40 \
    --patience 12 \
    --workers 0 \
    --threads 16 \
    --device cpu \
    2>&1 | tee "${output_dir}/train.log"
}

run_weight 0.25 0.75 hsi025_lidar075
run_weight 0.375 0.625 hsi0375_lidar0625
run_weight 0.5 0.5 hsi050_lidar050
run_weight 0.625 0.375 hsi0625_lidar0375
run_weight 0.75 0.25 hsi075_lidar025
run_weight 1.0 1.0 hsi100_lidar100

"${PYTHON_BIN}" -m dual_stream_hsi_lidar.compare_weight_sweep \
  --runs-root "${RUN_ROOT}" \
  --output-dir "${COMPARISON_DIR}" \
  --baseline-run hsi100_lidar100
