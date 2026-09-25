#!/usr/bin/env bash
set -euo pipefail

# One missing Hunan-train UTC frame. Keep V2 output separate from historical CPP.
RUN=/home/Data_Pool/zjnu/cpp_v2_20260918_fd201
BASE="$RUN/package/FY4B_CPP_V2_Portable"
OUT=/home/Data_Pool/zjnu/solar_cot_regional_20260925/cpp_v2_missing_pilot
TARGET_UTC=${CPP_TARGET_UTC:-2024-04-02T04:15:00}
TARGET_TAG=${TARGET_UTC//[-:T]/}
RUN_ID="hunan_cpp_v2_${TARGET_TAG}_b64_w0"
mkdir -p "$OUT/logs" "$OUT/output"
cd "$BASE"
set -a
. config/runtime.env
set +a
export GLOBAL_CPP_TEST_ROOT="$OUT/output"
export GLOBAL_CPP_RESULT_ROOT="$OUT/output/Result"
export GLOBAL_CPP_LOG_DIR="$OUT/logs"
export GLOBAL_CPP_WRITE_EMPTY_ON_ERROR=0
export GLOBAL_CPP_BATCH_SIZE=64
export GLOBAL_CPP_DATALOADER_WORKERS=0
export GLOBAL_CPP_TORCH_THREADS=4
export CUDA_VISIBLE_DEVICES=${CPP_GPU:-5}
exec "$RUN/venv/bin/python" Code/Run_Pipeline.py satellite \
  --satellite FY4B_105E --time "$TARGET_UTC" \
  --device cuda:0 --time-workers 1 --satellite-workers 1 \
  --retrieval-policy skip --run-id "$RUN_ID" \
  --log-dir "$OUT/logs" \
  --summary "$OUT/output/${RUN_ID}_summary.json"
