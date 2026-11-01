#!/bin/bash
set -euo pipefail

PBS_SCRIPT=/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_cot_station_patch_forecast_20261022/oof_source_cv_attempt15/monthly_val_score_attempt16/retry02/run_attempt16_monthly_val_inference.pbs
RECEIPT=/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_cot_station_patch_forecast_20261022/oof_source_cv_attempt15/monthly_val_score_attempt16/retry02/submission_receipt.txt
PY=/public/home/slfu/miniconda3/envs/swc/bin/python
test ! -e "$RECEIPT"

"$PY" - <<'PY'
import re
import subprocess
import sys

listing = subprocess.run(["qstat", "-u", "slfu"], text=True, capture_output=True, check=True).stdout
ids = re.findall(r"(?m)^(\d+\.tc6000)\s+", listing)
unfinished = []
for job_id in ids:
    detail = subprocess.run(["qstat", "-f", job_id], text=True, capture_output=True, check=True).stdout
    match = re.search(r"(?m)^\s*job_state\s*=\s*(\w+)", detail)
    if match and match.group(1) not in {"C", "E"}:
        unfinished.append((job_id, match.group(1)))
print("PRE_SUBMIT_UNFINISHED_SLFU_JOBS=" + repr(unfinished), flush=True)
if len(unfinished) >= 3:
    sys.exit("PBS limit reached; refusing to submit a fourth unfinished job")
PY

job_id=$(qsub "$PBS_SCRIPT")
printf 'job_id=%s\nsubmitted_at=%s\n' "$job_id" "$(date '+%Y-%m-%d %H:%M:%S %Z')" | tee "$RECEIPT"
qstat -u slfu | tee -a "$RECEIPT"
echo "SUBMITTED_ATTEMPT16_JOB=$job_id"
