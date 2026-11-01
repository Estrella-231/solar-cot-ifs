#!/usr/bin/env bash
set -euo pipefail

# Rebuild one canonical, unmerged fold-2 train-date OOF bank from the
# independently selected epoch-6 checkpoint. Train-date OOF keys are the
# fold's held-out rows; no test/buffer/GHI payload is read.
PROJECT=/home/Data_Pool/zjnu/hubei_cot_independent_20261016
STAGE=/tmp/zjnu_hubei_simvp_stage_20261016
REPO=/home/Data_Pool_3/chenyi/GHI/SimVPv2
PY=/home/Data_Pool/zjnu/.conda/envs/swc/bin/python
TOOLING="$PROJECT/oof_training/tooling_oof_merge_20261017_v4"
INFER="$PROJECT/oof_training/scripts/infer_hubei_simvp_trainval_patches_20261016.py"
AUDIT="$TOOLING/audit_hubei_simvp_oof_bank_20261017.py"
FOLD="$PROJECT/manifests/oof_fold_manifests_20261016_v5/folds/fold_2"
INDEX="$FOLD/simvp_oof_index.csv"
NORM="$FOLD/norm/simvp_train_only_norm.npz"
CHECKPOINT="$PROJECT/oof_training/S_oof_fold2_seed42_20261016_fd201_20261017/runs/S_oof_fold2_seed42_20261016_fd201_20261017/checkpoints/best.ckpt"
OUT="$PROJECT/oof_training/S_oof_fold2_oof_bank_fd201_20261017_raw_v4_retry1"
RUN_MANIFEST="$PROJECT/oof_training/S_oof_fold2_oof_bank_fd201_20261017_raw_v4_retry1.launch_manifest.txt"
INFER_LOG="$PROJECT/oof_training/S_oof_fold2_oof_bank_fd201_20261017_raw_v4_retry1.infer.log"
AUDIT_LOG="$PROJECT/oof_training/S_oof_fold2_oof_bank_fd201_20261017_raw_v4_retry1.audit.log"
EXPECTED_INDEX_SHA=261495bba4aff307fb32d9e980c54412dcdb94f5bffc305b690f63d45effcd8b
EXPECTED_NORM_SHA=c43077b59720925cb81fe5cb91268e57a0e095d8c4c0dd75b27f483d92e554b3
EXPECTED_CHECKPOINT_SHA=7a3309bad154a5bd1383b4ddc23e95889adddb57ca4379f73014b7f226f1d0b6
EXPECTED_STAGE_SHA=73e0f91955a7e8c07d69a1e6d79c900275ff000f350a826fb2fa04e79e47571a
EXPECTED_COUNT=850
GPU=0

test -x "$PY" && test -d "$REPO/openstl"
test -f "$STAGE/stage_report.json" && test -f "$STAGE/stage_manifest.csv"
test -f "$INDEX" && test -f "$NORM" && test -f "$CHECKPOINT"
test -f "$INFER" && test -f "$AUDIT"
test ! -e "$OUT"
test ! -e "$RUN_MANIFEST" && test ! -e "$INFER_LOG" && test ! -e "$AUDIT_LOG"
test "$(sha256sum "$INDEX" | awk '{print $1}')" = "$EXPECTED_INDEX_SHA"
test "$(sha256sum "$NORM" | awk '{print $1}')" = "$EXPECTED_NORM_SHA"
test "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = "$EXPECTED_CHECKPOINT_SHA"
test "$(sha256sum "$STAGE/stage_manifest.csv" | awk '{print $1}')" = "$EXPECTED_STAGE_SHA"

"$PY" - "$STAGE/stage_report.json" "$INDEX" "$EXPECTED_COUNT" <<'PY'
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
  echo "gpu_id=$GPU"
  echo "stage_manifest_sha256=$EXPECTED_STAGE_SHA"
  echo "index_sha256=$EXPECTED_INDEX_SHA"
  echo "normalization_sha256=$EXPECTED_NORM_SHA"
  echo "checkpoint_sha256=$EXPECTED_CHECKPOINT_SHA"
  echo "oof_validation_rows=$EXPECTED_COUNT"
  echo "test_payloads_read=0"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv -i "$GPU"
} > "$RUN_MANIFEST"

cd "$REPO"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$INFER" \
  --repo "$REPO" --index "$INDEX" --norm "$NORM" --checkpoint "$CHECKPOINT" \
  --output "$OUT" --stage-root "$STAGE" --batch-size 4 --workers 4 \
  --device cuda --shard-index 0 --shard-count 1 --splits val \
  > "$INFER_LOG" 2>&1

"$PY" "$AUDIT" --bank "$OUT" --index "$INDEX" --expected-count "$EXPECTED_COUNT" \
  --output "$OUT/OOF_BANK_AUDIT_20261017_V5.json" \
  > "$AUDIT_LOG" 2>&1
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%S%z)" >> "$RUN_MANIFEST"
echo "HUBEI_FOLD2_RAW_OOF_BANK_AND_V4_AUDIT_PASS"
