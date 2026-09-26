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
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec "$RUN/venv/bin/python" "$ROOT/code_gate_v1/run_cpp_region_fullscene_gate_v1.py" \
  --base "$BASE" --grid /tmp/solar_cot_grid_static_20260925.npz \
  --times "$ROOT/p2_selection_v1/references24.json" \
  --region "$ROOT/p2_stratified48_two_v6" --output "$ROOT/p2_fullscene24_gate_v1"
