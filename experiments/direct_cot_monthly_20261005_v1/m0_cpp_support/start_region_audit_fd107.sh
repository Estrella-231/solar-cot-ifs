#!/bin/bash
set -euo pipefail
BASE=/tmp/direct_cot_m0_20261006
INV=$BASE/cpp_file_support_inventory.csv
GRID=/home/Data_Pool_3/chenyi/Auxiliary_data/HuNan_AGRI_Preprocessed/grid_static.npz
AUX=/home/Data_Pool_3/chenyi/Auxiliary_data/HuNan_AGRI_Preprocessed/aux_256
PY=/usr/bin/python3
CHUNKS=(0 3919 7838 11757)
for i in 0 1 2 3; do
  off=${CHUNKS[$i]}
  nohup $PY $BASE/audit_direct_cot_region_supervision_20261006.py \
    --inventory $INV --grid $GRID --aux-root $AUX \
    --output $BASE/part${i}.csv --offset $off --limit 3919 \
    --skip-utc 202410260600 --skip-utc 202411161400 \
    --skip-utc 202508220215 --skip-utc 202508220300 \
    --skip-utc 202509040100 --skip-utc 202511050800 \
    --skip-utc 202511061400 --skip-utc 202511061500 \
    --skip-utc 202512020500 >$BASE/part${i}.log 2>&1 &
  echo $! >>$BASE/audit_pids.txt
done
