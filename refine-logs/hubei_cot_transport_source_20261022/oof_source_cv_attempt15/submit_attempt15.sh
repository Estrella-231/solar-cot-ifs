#!/usr/bin/env bash
set -euo pipefail

ROOT=/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_cot_station_patch_forecast_20261022/oof_source_cv_attempt15
PBS_SCRIPT=$ROOT/scripts/run_oof_source_cv_attempt15.pbs
RECEIPT=$ROOT/submit_receipt_$(date +%Y%m%dT%H%M%S).txt

{
  echo "PRE_SUBMIT_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "PRE_SUBMIT_QUEUE_BEGIN"
  qstat -u slfu
  echo "PRE_SUBMIT_QUEUE_END"
  # qstat's state is column 10 (column 5 is SessID); count every non-C/F job.
  unfinished=$(qstat -u slfu | awk '$1 ~ /^[0-9]+[.][[:alnum:]_-]+$/ && $10 !~ /^[CF]$/ {n++} END {print n+0}')
  echo "PRE_SUBMIT_UNFINISHED_COUNT=$unfinished"
  test "$unfinished" -lt 3
  echo "NODE22_SCHEDULER_STATE_BEGIN"
  pbsnodes node22 | sed -n '1,24p'
  echo "NODE22_SCHEDULER_STATE_END"
  test -f "$PBS_SCRIPT"
  test -f "$ROOT/cache_build_v1/CACHE_AUDIT.json"
  test ! -e "$ROOT/pair_complete.json"
  jobid=$(qsub "$PBS_SCRIPT")
  echo "SUBMITTED_JOB=$jobid"
  echo "POST_SUBMIT_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "POST_SUBMIT_QUEUE_BEGIN"
  qstat -u slfu
  echo "POST_SUBMIT_QUEUE_END"
} 2>&1 | tee "$RECEIPT"
