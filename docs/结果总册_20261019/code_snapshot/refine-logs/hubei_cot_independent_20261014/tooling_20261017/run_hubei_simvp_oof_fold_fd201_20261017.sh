#!/usr/bin/env bash
set -euo pipefail

FOLD="${1:?usage: run_hubei_simvp_oof_fold_fd201_20261017.sh FOLD}"
case "$FOLD" in 2|3|4) ;; *) echo "invalid fold: $FOLD" >&2; exit 2 ;; esac

BASE=/home/Data_Pool/zjnu/hubei_cot_independent_20261016
STAGE=/tmp/zjnu_hubei_simvp_stage_20261016
MANIFESTS="$BASE/manifests/oof_fold_manifests_20261016_v5/folds/fold_${FOLD}"
CONFIG="$BASE/oof_training/configs/hbagri_simvp_oof_fold${FOLD}_seed42_20261016.py"
NORM="$MANIFESTS/norm/simvp_train_only_norm.npz"
INDEX="$MANIFESTS/simvp_trainval_index.csv"
TRAINER="$BASE/scripts/run_hubei_simvp_frozen.py"
REPO=/home/Data_Pool_3/chenyi/GHI/SimVPv2
PY=/home/Data_Pool/zjnu/.conda/envs/swc/bin
RUNROOT="$BASE/oof_training/S_oof_fold${FOLD}_seed42_20261016_fd201_20261017"
RUNNAME="S_oof_fold${FOLD}_seed42_20261016_fd201_20261017"

test -f "$STAGE/stage_verification_receipt.json"
test -f "$STAGE/stage_manifest.csv"
test -f "$CONFIG" && test -f "$NORM" && test -f "$INDEX" && test -f "$TRAINER"
test -f "$REPO/tools/train.py"
test ! -e "$RUNROOT"

"$PY/python" - "$STAGE/stage_verification_receipt.json" "$FOLD" "$INDEX" <<'PY'
import csv, json, sys
receipt = json.load(open(sys.argv[1]))
assert receipt["status"] == "PBS_TRAINVAL_STAGE_SHA256_AND_INDEX_REBASE_PASSED_TEST_CLOSED"
assert receipt["test_sequences"] == 0 and receipt["test_or_buffer_payloads_staged"] == 0
assert receipt["payload_files_sha256_verified"] == 23468
assert receipt["payload_bytes_verified"] == 66508197403
fold = int(sys.argv[2])
with open(sys.argv[3], newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
assert rows and all(row["split"] in {"train", "val"} for row in rows)
print(f"HUBEI_FD201_PREFLIGHT_PASS fold={fold} rows={len(rows)} test_closed=1")
PY

GPU_IDS=0,1,2
GPU_MEMORY=$(nvidia-smi --id="$GPU_IDS" --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 >= 500 {bad=1} END {print bad+0}')
test "$GPU_MEMORY" -eq 0
mkdir -p "$RUNROOT/cwd"
{
  echo "host=$(hostname)"
  echo "fold=$FOLD"
  echo "visible_physical_gpu_ids=$GPU_IDS"
  echo "torchrun_local_world_size=3"
  echo "per_rank_batch=4"
  echo "seed=42"
  echo "stage_manifest_sha256=$(sha256sum "$STAGE/stage_manifest.csv" | awk '{print $1}')"
  echo "index_sha256=$(sha256sum "$INDEX" | awk '{print $1}')"
  echo "norm_sha256=$(sha256sum "$NORM" | awk '{print $1}')"
  echo "config_sha256=$(sha256sum "$CONFIG" | awk '{print $1}')"
  echo "trainer_sha256=$(sha256sum "$TRAINER" | awk '{print $1}')"
  echo "train_entry_sha256=$(sha256sum "$REPO/tools/train.py" | awk '{print $1}')"
  echo "repo_openstl_exp_sha256=$(sha256sum "$REPO/openstl/api/exp.py" | awk '{print $1}')"
} > "$RUNROOT/launch_manifest.txt"

export SIMVP_REPO="$REPO"
export HUBEI_SIMVP_CONFIG="$CONFIG"
export HUBEI_SIMVP_NORM="$NORM"
export HUBEI_SIMVP_RUN_CWD="$RUNROOT/cwd"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "$RUNROOT/cwd"
"$PY/torchrun" --standalone --nnodes=1 --nproc_per_node=3 \
  "$TRAINER" --dist --gpus 0 1 2 --res_dir "$RUNROOT/runs" \
  --ex_name "$RUNNAME" 2>&1 | tee "$RUNROOT/train.log"
