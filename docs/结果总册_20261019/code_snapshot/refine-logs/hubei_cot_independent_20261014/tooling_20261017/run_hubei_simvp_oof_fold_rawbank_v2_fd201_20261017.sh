#!/usr/bin/env bash
set -euo pipefail

# Rebuild one immutable, single-fold train-date OOF forecast bank on FD-201.
# Use only fold-specific OOF validation rows; test/buffer and GHI labels remain closed.
FOLD="${1:?fold id 3 or 4 required}"
GPU="${2:?GPU id required}"
case "$FOLD" in
  3)
    COUNT=809
    INDEX_SHA=ade3c0930c4fd8787207c85e5855ae36fc61e778601cdb741957661f2390eab5
    NORM_SHA=f8ff7379b4695b246150565e71651d05267e65be1d08c69f6fd6788082f2b1f2
    CKPT_SHA=50f184270207695b1d69c7f2db47c33e1571395320304d4862560ff0e4430577
    CKPT=/home/Data_Pool/zjnu/hubei_cot_independent_20261016/oof_training/checkpoint_imports_tc6000_20261017/fold_3/best.ckpt
    TRAIN_JOB=214526.tc6000
    TRAIN_EXIT=0
    ;;
  4)
    COUNT=807
    INDEX_SHA=6291a33e95199b13e93fdd696ad0f793af0d273efa6adaf71cb7adc9401644f6
    NORM_SHA=128a92c62bbe14c242837b62d730294be26aa92a276cd334d67a67747b6b60b5
    CKPT_SHA=5637925fd0d4bc092c6f5f8996b20cf356f8bcf0b4cb358768e8856aec3f03a3
    CKPT=/home/Data_Pool/zjnu/hubei_cot_independent_20261016/oof_training/checkpoint_imports_tc6000_20261017/fold_4/best.ckpt
    TRAIN_JOB=214527.tc6000
    TRAIN_EXIT=0
    ;;
  *) echo "Only folds 3-4 are permitted by this runner" >&2; exit 2 ;;
esac

BASE=/home/Data_Pool/zjnu/hubei_cot_independent_20261016
PY=/home/Data_Pool/zjnu/.conda/envs/swc/bin/python
REPO=/home/Data_Pool_3/chenyi/GHI/SimVPv2
STAGE=/tmp/zjnu_hubei_simvp_stage_20261016
FOLD_ROOT="$BASE/manifests/oof_fold_manifests_20261016_v5/folds/fold_$FOLD"
INDEX="$FOLD_ROOT/simvp_oof_index.csv"
NORM="$FOLD_ROOT/norm/simvp_train_only_norm.npz"
TOOL="$BASE/oof_training/tooling_oof_merge_20261017_v5"
INFER="$TOOL/infer_hubei_simvp_trainval_patches_20261016.py"
AUDIT="$TOOL/audit_hubei_simvp_oof_bank_20261017.py"
STAGE_SHA=73e0f91955a7e8c07d69a1e6d79c900275ff000f350a826fb2fa04e79e47571a
TRAIN_RECEIPT="$BASE/oof_training/checkpoint_imports_tc6000_20261017/fold_${FOLD}/training_exit_receipt.json"
OUT="$BASE/oof_training/S_oof_fold${FOLD}_oof_bank_fd201_20261017_raw_v5_tc6000_import_gpu${GPU}"
RUN_MANIFEST="$OUT.launch_manifest.txt"
INFER_LOG="$OUT.infer.log"
AUDIT_LOG="$OUT.audit.log"

test -x "$PY" && test -d "$REPO/openstl"
test -f "$STAGE/stage_report.json" && test -f "$STAGE/stage_manifest.csv"
test -f "$INDEX" && test -f "$NORM" && test -f "$CKPT" && test -f "$TRAIN_RECEIPT"
test -f "$INFER" && test -f "$AUDIT"
test ! -e "$OUT" && test ! -e "$RUN_MANIFEST" && test ! -e "$INFER_LOG" && test ! -e "$AUDIT_LOG"
test "$(sha256sum "$INDEX" | awk '{print $1}')" = "$INDEX_SHA"
test "$(sha256sum "$NORM" | awk '{print $1}')" = "$NORM_SHA"
test "$(sha256sum "$CKPT" | awk '{print $1}')" = "$CKPT_SHA"
test "$(sha256sum "$STAGE/stage_manifest.csv" | awk '{print $1}')" = "$STAGE_SHA"
"$PY" - "$TRAIN_RECEIPT" "$TRAIN_JOB" "$TRAIN_EXIT" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
assert r["status"] == "PBS_TRAINING_EXIT_CONFIRMED"
assert r["job_id"] == sys.argv[2]
assert r["exit_status"] == int(sys.argv[3]) == 0
assert "Exit_status=0" in r["exit_record_line"]
PY

"$PY" - "$STAGE/stage_report.json" "$INDEX" "$COUNT" <<'PY'
import csv, json, sys
stage = json.load(open(sys.argv[1], encoding="utf-8"))
assert stage["status"] == "LOCAL_STAGE_COMPLETE_TRAINVAL_ONLY_TEST_CLOSED"
assert stage["test_sequences"] == 0 and stage["test_or_buffer_payloads_staged"] == 0
with open(sys.argv[2], newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
assert len(rows) == int(sys.argv[3]), (len(rows), sys.argv[3])
assert all(row["split"] == "val" for row in rows)
PY

test "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i "$GPU" | sed 's/ MiB//')" -lt 500
{
  echo "host=$(hostname)"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%S%z)"
  echo "fold=$FOLD"
  echo "gpu_id=$GPU"
  echo "stage_manifest_sha256=$STAGE_SHA"
  echo "index_sha256=$INDEX_SHA"
  echo "normalization_sha256=$NORM_SHA"
  echo "checkpoint_sha256=$CKPT_SHA"
  echo "training_job_id=$TRAIN_JOB"
  echo "training_exit_receipt_sha256=$(sha256sum "$TRAIN_RECEIPT" | awk '{print $1}')"
  echo "oof_validation_rows=$COUNT"
  echo "test_payloads_read=0"
  echo "buffer_payloads_read=0"
  echo "test_or_buffer_rows=0"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv -i "$GPU"
} > "$RUN_MANIFEST"

cd "$REPO"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$INFER" \
  --repo "$REPO" --index "$INDEX" --norm "$NORM" --checkpoint "$CKPT" \
  --output "$OUT" --stage-root "$STAGE" --batch-size 4 --workers 4 \
  --device cuda --shard-index 0 --shard-count 1 --splits val \
  > "$INFER_LOG" 2>&1

"$PY" "$AUDIT" --bank "$OUT" --index "$INDEX" --expected-count "$COUNT" \
  --output "$OUT/OOF_BANK_AUDIT_20261017_V5.json" \
  > "$AUDIT_LOG" 2>&1
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%S%z)" >> "$RUN_MANIFEST"
echo "HUBEI_FOLD${FOLD}_RAW_OOF_BANK_AND_V5_AUDIT_PASS"
