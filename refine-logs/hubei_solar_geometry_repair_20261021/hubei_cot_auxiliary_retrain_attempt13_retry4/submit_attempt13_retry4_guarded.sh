#!/bin/bash
set -euo pipefail
ATTEMPT=/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_solar_geometry_repair_20261021/hubei_cot_auxiliary_retrain_attempt13_retry4
QSTAT=/opt/gridview/pbs/dispatcher/bin/qstat
QSUB=/opt/gridview/pbs/dispatcher/bin/qsub
QOUT=$("$QSTAT" -u slfu)
printf 'PRE_SUBMIT_QSTAT_BEGIN\n%s\nPRE_SUBMIT_QSTAT_END\n' "$QOUT"
COUNT=$(printf '%s\n' "$QOUT" | awk '$1 ~ /^[0-9]+\./ && $10 != "C" && $10 != "F" {n++} END {print n+0}')
if [ "$COUNT" -ge 3 ]; then echo "QUEUE_LIMIT_BLOCK unfinished=$COUNT limit=3"; exit 43; fi
test ! -e "$ATTEMPT/PAIR_STARTED_attempt13.json"
test ! -e "$ATTEMPT/PAIR_COMPLETE_attempt13.json"
test ! -d "$ATTEMPT/runs"
JOB=$("$QSUB" "$ATTEMPT/attempt13_pair_retry4.pbs")
echo "SUBMITTED_JOB=$JOB unfinished_before=$COUNT"
printf 'POST_SUBMIT_QSTAT_BEGIN\n'
"$QSTAT" -u slfu
printf 'POST_SUBMIT_QSTAT_END\n'
"$QSTAT" -f "$JOB" | grep -E 'Job Id:|job_state =|queue =|exec_host =|Resource_List\.(nodes|walltime) ='
