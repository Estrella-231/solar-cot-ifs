#!/usr/bin/env bash
# Dependent final stage of the already-running repair, not a recurring job.
set -euo pipefail
ROOT=/home/Data_Pool_3/wangyc/irradiance_paper/solar_cot_ifs_20260905
PID=3798830
while [ -r "/proc/$PID/cmdline" ] && tr '\0' ' ' < "/proc/$PID/cmdline" | grep -q 'code/repair_cpp_cache.py'; do
  sleep 30
done
PY=/home/Data_Pool/zjnu/.conda/envs/swc/bin/python
# Resume only failed entries; existing ok/missing_source records stay untouched.
# Two bounded serial retries reduce transient shared-storage/HDF read failures.
for attempt in 1 2; do
  echo "RETRY_ERRORS attempt=$attempt $(date -Is)"
  if "$PY" -u "$ROOT/code/repair_cpp_cache.py" \
    --combined-root /home/Data_Pool_3/chenyi/Auxiliary_data/HuNan_Station16_Combined \
    --cpp-root /home/Data_Pool_3/data/Global_CPP/FY/FY4B/Result/CPP \
    --contract "$ROOT/configs/cot_contract.json" \
    --output "$ROOT/data/cpp_aligned_20260905_v2" --workers 1; then
    break
  fi
done
/home/Data_Pool/zjnu/.conda/envs/swc/bin/python /tmp/verify_cpp_repair.py "$ROOT/data/cpp_aligned_20260905_v2"
cp /tmp/cpp_repair_20260905_v2.log "$ROOT/audits/cpp_repair_20260905_v2.log"
echo "ACCEPTED $(date -Is)"
