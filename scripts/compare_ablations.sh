#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-/home/temp/miniforge3/envs/cnn/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/dual_stream_mpl}"

"${PYTHON_BIN}" -m dual_stream_hsi_lidar.compare_ablations \
  --runs-root dual_stream_hsi_lidar/runs \
  --main-run fusion \
  --ablation-runs hsi_only lidar_only \
  --output-dir dual_stream_hsi_lidar/runs/comparison \
  "$@"
