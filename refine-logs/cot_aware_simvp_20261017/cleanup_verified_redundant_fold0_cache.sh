#!/usr/bin/env bash
set -euo pipefail

EXP=/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/refine-logs/cot_aware_simvp_20261017
COMPLETE=/tmp/slfu_future_ghi_oof_fold0_214933_tc6000
CACHE=/tmp/slfu_future_ghi_oof_fold0_resume_214946_tc6000
TRAIN_FRAMES=$EXP/results/future_ghi_oof_manifests_v1/fold_0/train_frames.txt
PYTHON=/public/home/slfu/miniconda3/envs/swc/bin/python

[[ "$CACHE" == /tmp/slfu_future_ghi_oof_fold0_resume_214946_tc6000 ]]
[[ "$(readlink -f -- "$CACHE")" == "$CACHE" ]]
[[ -d "$CACHE" && ! -L "$CACHE" ]]
if ps -eo pid=,args= | grep -E '[p]ython.*--cache[ =]/tmp/slfu_future_ghi_oof_fold0_resume_214946_tc6000'; then
  echo 'REFUSE: a Python process still references the cache' >&2
  exit 42
fi

"$PYTHON" "$EXP/verify_redundant_oof_cache.py" \
  --complete-cache "$COMPLETE" --partial-cache "$CACHE" --train-frames "$TRAIN_FRAMES"
rm -rf -- "$CACHE"
[[ ! -e "$CACHE" ]]
"$PYTHON" "$EXP/verify_future_ghi_oof_cache.py" "$COMPLETE" "$TRAIN_FRAMES"
df -h /tmp
echo 'REDUNDANT_FOLD0_PARTIAL_CACHE_CLEANUP_PASS'
