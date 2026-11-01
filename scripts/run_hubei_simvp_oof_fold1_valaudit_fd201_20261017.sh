#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/Data_Pool/zjnu/hubei_cot_independent_20261016
STAGE=/tmp/zjnu_hubei_simvp_stage_20261016
REPO=/home/Data_Pool_3/chenyi/GHI/SimVPv2
PY=/home/Data_Pool/zjnu/.conda/envs/swc/bin/python
INFER="$PROJECT/oof_training/scripts/infer_hubei_simvp_trainval_patches_20261016.py"
AUDIT="$PROJECT/oof_training/scripts/audit_hubei_simvp_oof_fold_val_shards_20261017.py"
FOLD="$PROJECT/manifests/oof_fold_manifests_20261016_v5/folds/fold_1"
INDEX="$FOLD/simvp_trainval_index.csv"
NORM="$FOLD/norm/simvp_train_only_norm.npz"
CHECKPOINT="$PROJECT/oof_training/runs/S_oof_fold1_seed42_20261016_attempt1/checkpoints/best.ckpt"
OUT="$PROJECT/oof_training/S_oof_fold1_val_reload_audit_fd201_20261017"
EXPECTED_INDEX_SHA=5d436d6ebc4c3a6d622d2288afb84e1770f8977451d80c3f7fc5e1f2922a5d17
EXPECTED_NORM_SHA=84c215afec383b4b85326c9283d64c02ae56e707129064df1dc2e80f8f74ac61
EXPECTED_CHECKPOINT_SHA=e95a54f44c8d6e5b14935270f616e7a66c9d00921ef2c2ae4d9d2ebeac53871a
EXPECTED_STAGE_MANIFEST_SHA=73e0f91955a7e8c07d69a1e6d79c900275ff000f350a826fb2fa04e79e47571a
GPU_IDS=(0 1 2 3)

test -x "$PY" && test -d "$REPO/openstl"
test -f "$STAGE/stage_report.json" && test -f "$STAGE/stage_manifest.csv"
test -f "$INDEX" && test -f "$NORM" && test -f "$CHECKPOINT"
test -f "$INFER" && test -f "$AUDIT"
test ! -e "$OUT"

"$PY" - "$STAGE/stage_report.json" "$EXPECTED_STAGE_MANIFEST_SHA" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
assert r["status"] == "LOCAL_STAGE_COMPLETE_TRAINVAL_ONLY_TEST_CLOSED"
assert r["sequences"] == {"train": 5153, "val": 509}
assert r["test_sequences"] == 0 and r["test_or_buffer_payloads_staged"] == 0
assert r["staged_files"] == 23468 and r["staged_bytes"] == 66508197403
assert r["stage_manifest_sha256"] == sys.argv[2]
print("FD201_SOURCE_STAGE_PREFLIGHT_PASS test_closed=1")
PY

test "$(sha256sum "$INDEX" | awk '{print $1}')" = "$EXPECTED_INDEX_SHA"
test "$(sha256sum "$NORM" | awk '{print $1}')" = "$EXPECTED_NORM_SHA"
test "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = "$EXPECTED_CHECKPOINT_SHA"
test "$(sha256sum "$STAGE/stage_manifest.csv" | awk '{print $1}')" = "$EXPECTED_STAGE_MANIFEST_SHA"

for gpu in "${GPU_IDS[@]}"; do
  used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  test "$used" -lt 500
done

export PYTHONPATH="$REPO" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="$(dirname "$PY")/../lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
{
  echo "host=$(hostname)"
  echo "started_utc=$(date -u -Is)"
  echo "gpu_ids=${GPU_IDS[*]}"
  echo "source_stage=$STAGE"
  echo "stage_manifest_sha256=$EXPECTED_STAGE_MANIFEST_SHA"
  echo "index_sha256=$EXPECTED_INDEX_SHA"
  echo "norm_sha256=$EXPECTED_NORM_SHA"
  echo "checkpoint_sha256=$EXPECTED_CHECKPOINT_SHA"
  echo "validation_sequences=407"
  echo "test_payloads_read=0"
  nvidia-smi --id="$(IFS=,; echo "${GPU_IDS[*]}")" --query-gpu=index,name,memory.used,memory.total --format=csv
} > "$OUT/launch_manifest.txt"

pids=()
for gpu in "${GPU_IDS[@]}"; do
  shard="$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$INFER" \
    --repo "$REPO" --index "$INDEX" --norm "$NORM" --checkpoint "$CHECKPOINT" \
    --output "$OUT/shard_$shard" --stage-root "$STAGE" \
    --batch-size 4 --workers 2 --device cuda:0 \
    --shard-index "$shard" --shard-count "${#GPU_IDS[@]}" --splits val \
    > "$OUT/shard_$shard.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
test "$failed" -eq 0

shard_dirs=()
for gpu in "${GPU_IDS[@]}"; do shard_dirs+=("$OUT/shard_$gpu"); done
"$PY" "$AUDIT" --index "$INDEX" --shard-dir "${shard_dirs[@]}" \
  --fold 1 --output "$OUT/FOLD1_VALIDATION_RELOAD_AUDIT.json"
echo "finished_utc=$(date -u -Is)" >> "$OUT/launch_manifest.txt"
