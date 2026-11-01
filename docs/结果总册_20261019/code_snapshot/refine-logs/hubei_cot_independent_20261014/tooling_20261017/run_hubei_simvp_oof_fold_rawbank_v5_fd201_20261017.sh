#!/usr/bin/env bash
set -euo pipefail

# Rebuild one immutable, single-fold train-date OOF forecast bank on FD-201.
# Use only fold-specific OOF validation rows; test/buffer and GHI labels remain closed.
FOLD="${1:?fold id 0, 1, or 2 required}"
GPU="${2:?GPU id required}"
case "$FOLD" in
  0)
    COUNT=823
    INDEX_SHA=47afce18a11b7d0f0e43369d069747bece0347097a39561dcad1eb282677ac73
    NORM_SHA=1dbc3be2391cb530a6b438bcf9bd8c4093252a88cbbef295a74eec97cccfa2b2
    CKPT_SHA=2e9e6d50dd5693f7a856a3acb57aea96d5fe4dade67e9d5252f4f28b683ef86c
    CKPT=/home/Data_Pool/zjnu/hubei_cot_independent_20261016/oof_training/runs/S_oof_fold0_seed42_20261016_attempt3/checkpoints/best.ckpt
    ;;
  1)
    COUNT=797
    INDEX_SHA=b83386e2811991b4919d63f5f221f346d3acc82e1671da08eed91e14bf3d8f4a
    NORM_SHA=84c215afec383b4b85326c9283d64c02ae56e707129064df1dc2e80f8f74ac61
    CKPT_SHA=e95a54f44c8d6e5b14935270f616e7a66c9d00921ef2c2ae4d9d2ebeac53871a
    CKPT=/home/Data_Pool/zjnu/hubei_cot_independent_20261016/oof_training/runs/S_oof_fold1_seed42_20261016_attempt1/checkpoints/best.ckpt
    ;;
  2)
    COUNT=850
    INDEX_SHA=261495bba4aff307fb32d9e980c54412dcdb94f5bffc305b690f63d45effcd8b
    NORM_SHA=c43077b59720925cb81fe5cb91268e57a0e095d8c4c0dd75b27f483d92e554b3
    CKPT_SHA=7a3309bad154a5bd1383b4ddc23e95889adddb57ca4379f73014b7f226f1d0b6
    CKPT=/home/Data_Pool/zjnu/hubei_cot_independent_20261016/oof_training/S_oof_fold2_seed42_20261016_fd201_20261017/runs/S_oof_fold2_seed42_20261016_fd201_20261017/checkpoints/best.ckpt
    ;;
  *) echo "Only completed folds 0-2 are permitted by this runner" >&2; exit 2 ;;
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
OUT="$BASE/oof_training/S_oof_fold${FOLD}_oof_bank_fd201_20261017_raw_v5_attempt2_gpu${GPU}"
RUN_MANIFEST="$OUT.launch_manifest.txt"
INFER_LOG="$OUT.infer.log"
AUDIT_LOG="$OUT.audit.log"

test -x "$PY" && test -d "$REPO/openstl"
test -f "$STAGE/stage_report.json" && test -f "$STAGE/stage_manifest.csv"
test -f "$INDEX" && test -f "$NORM" && test -f "$CKPT"
test -f "$INFER" && test -f "$AUDIT"
test ! -e "$OUT" && test ! -e "$RUN_MANIFEST" && test ! -e "$INFER_LOG" && test ! -e "$AUDIT_LOG"
test "$(sha256sum "$INDEX" | awk '{print $1}')" = "$INDEX_SHA"
test "$(sha256sum "$NORM" | awk '{print $1}')" = "$NORM_SHA"
test "$(sha256sum "$CKPT" | awk '{print $1}')" = "$CKPT_SHA"
test "$(sha256sum "$STAGE/stage_manifest.csv" | awk '{print $1}')" = "$STAGE_SHA"

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
