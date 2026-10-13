#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/Data_Pool_3/zjnu/cot_value_first_20261013/monthly_R_station16_20261013_v1
DATA=/home/Data_Pool_3/chenyi/Auxiliary_data/HuNan_AGRI_Preprocessed
OUT="$ROOT/data/full_trainval_pack_v1"
export PYTHONPATH="$ROOT/runtime_packages${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
test ! -e "$OUT"
python3 -u "$ROOT/scripts/pack_station16_cpp_agri.py" \
  --candidates "$ROOT/input/target_candidates_train_val.csv" \
  --cpp-inventory "$ROOT/input/cpp_file_support_by_frame.csv" \
  --data-root "$DATA" --grid-static "$DATA/grid_static.npz" \
  --output "$OUT"
