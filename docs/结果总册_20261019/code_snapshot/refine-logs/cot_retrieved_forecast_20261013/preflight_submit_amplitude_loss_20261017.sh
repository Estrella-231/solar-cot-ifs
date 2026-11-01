#!/bin/bash
set -euo pipefail

ROOT=/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907
BRANCH=$ROOT/refine-logs/cot_retrieved_forecast_20261013
PBS=$BRANCH/run_amplitude_loss_fulltrain_seed42_20261017.pbs
OUT=$BRANCH/results/amplitude_loss_20261017_seed42_b1024_e12
QSTAT=/opt/gridview/pbs/dispatcher/bin/qstat
QSUB=/opt/gridview/pbs/dispatcher/bin/qsub

echo "=== live slfu PBS queue immediately before qsub ==="
QUEUE=$("$QSTAT" -u slfu)
printf '%s\n' "$QUEUE"
N=$(printf '%s\n' "$QUEUE" | awk '$1 ~ /^[0-9]+\.tc6000$/ {n++} END {print n+0}')
echo "unfinished_jobs_count=$N"
if (( N >= 3 )); then
  echo "PBS limit reached; no job submitted" >&2
  exit 20
fi

test -f "$PBS"
test -f "$BRANCH/train_retrieved_cot_forecaster_20261017.py"
test ! -e "$OUT"
test -w "$(dirname "$OUT")"
test -w "$BRANCH/logs"
test "$(sha256sum "$BRANCH/train_retrieved_cot_forecaster_20261017.py" | awk '{print $1}')" = b7b39c5e4ca4b1698d320b88d1106fdfa507ba85736ec19391c220b31c40b905
test "$(sha256sum "$PBS" | awk '{print $1}')" = 9b449697e0643abfaccfe6fc3f8b8ec2ee7648e540a9c4f7f64a469dda493abd

/public/home/slfu/miniconda3/envs/swc/bin/python - "$BRANCH" "$ROOT" <<'PY'
import json, pathlib, sys
b, root = map(pathlib.Path, sys.argv[1:])
bank = root / "experiments/monthly_R_station16_20261013_v1/results/new_monthly_R_seed42_cot_bank_v1/bank_manifest.json"
pack = b / "data/sequence_cpp_pack_20261017/pack_manifest.json"
frame = b / "data/frame_sidecars_20261017_v1/pack_manifest.json"
audit = b / "results/data_gate/retrieved_cot_full_bank_audit_20261017.json"
items = [json.loads(p.read_text()) for p in (bank, pack, frame, audit)]
assert [x.get("split_id") for x in items] == ["hunan_monthly_block_2d_buffer1d_seed42_v1"] * 4
assert all(x.get("test_payload_opened") is False for x in items)
assert items[0].get("state") == "COMPLETE"
assert items[1].get("state") == "PASS_TRAIN_VAL_SEQUENCE_CPP_PACK"
assert items[2].get("state") == "COMPLETE"
assert items[3].get("state") == "PASS_FULL_BANK_FINITE_AND_PIXEL_SUPPORT_AUDIT"
assert items[3].get("failure_count") == 0
assert items[3].get("R_checkpoint_sha256") == items[0].get("R_checkpoint_sha256")
print("data_gate=PASS split/source/test closed")
PY

echo "=== submitting exactly one GPU task ==="
JOB=$("$QSUB" "$PBS")
echo "submitted_job=$JOB"
"$QSTAT" -u slfu
