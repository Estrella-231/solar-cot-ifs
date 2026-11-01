#!/bin/bash
set -euo pipefail

BASE=/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/monthly_R_station16_20261013_v1
C1=$BASE/c1_conservative_gate_20261015
for p in \
  "$C1/scripts/run_conservative_gate_20261015.pbs" \
  "$C1/scripts/run_conservative_gate.py" \
  "$C1/scripts/conservative_gate_metrics.py" \
  "$C1/scripts/train_cloud_gate.py" \
  "$C1/data/clp_trainval.npz" \
  "$C1/data/oof_train_folds_20261015_v2.csv" \
  "$BASE/data/pack_trainval_v1/pack_status.json" \
  "$BASE/results/seed42_v1/best.pt" \
  "$BASE/results/seed42_v1/normalization.json" \
  "$BASE/results/seed42_v1/validation_predictions.npz" \
  "$BASE/results/cloud_gate_seed42_20261014_retry1/validation_predictions.npz"; do
  test -r "$p"
done
test ! -e "$C1/output_attempt3"

QUEUE=$(/opt/gridview/pbs/dispatcher/bin/qstat -u slfu)
printf 'PBS_QUEUE_BEFORE_SUBMIT\n%s\n' "$QUEUE"
COUNT=$(printf '%s\n' "$QUEUE" | awk '$1 ~ /^[0-9]+\./ && $10 != "C" && $10 != "F" {n++} END {print n+0}')
printf 'UNFINISHED_COUNT=%s\n' "$COUNT"
if [ "$COUNT" -ge 3 ]; then
  printf 'QUEUE_GATE_BLOCKED\n'
  exit 2
fi

qsub "$C1/scripts/run_conservative_gate_20261015.pbs"
