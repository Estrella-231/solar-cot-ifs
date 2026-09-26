#!/usr/bin/env bash
set -euo pipefail
RUN=/home/Data_Pool/zjnu/cpp_v2_20260918_fd201
BASE="$RUN/package/FY4B_CPP_V2_Portable"
ROOT=/home/Data_Pool/zjnu/solar_cot_regional_20260926
cd "$BASE"
set -a
. config/runtime.env
set +a
export CUDA_VISIBLE_DEVICES=5
export GLOBAL_CPP_TORCH_THREADS=4
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec "$RUN/venv/bin/python" "$ROOT/code_v6/produce_hunan_cpp_region_v1.py" \
  --base "$BASE" --grid /tmp/solar_cot_grid_static_20260925.npz \
  --times-json "$ROOT/p2_selection1000_v2/times1000.json" --resume \
  --output "$ROOT/p2_continuous1000_two_v6" --batch 64 --pad-batch 64 --networks two \
  --input-mode region
