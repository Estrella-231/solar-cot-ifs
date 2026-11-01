#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/Data_Pool/zjnu/hubei_cot_independent_20261016
MANIFESTS="$ROOT/manifests/oof_fold_manifests_20261016_v5/folds"
RUNS="$ROOT/oof_training"
STAGE=/tmp/zjnu_hubei_simvp_stage_20261016
PY=/home/Data_Pool/zjnu/anaconda3/envs/tab/bin/python
REPO=/home/Data_Pool_3/chenyi/GHI/SimVPv2
INFER="$ROOT/oof_training/tooling_oof_merge_20261017_v5/infer_hubei_simvp_trainval_patches_20261016.py"
AUDIT="$ROOT/oof_training/tooling_oof_merge_20261017_v5/audit_hubei_simvp_oof_fold_val_shards_20261017_v2_keys_hash.py"
EXPECTED_STAGE_SHA=73e0f91955a7e8c07d69a1e6d79c900275ff000f350a826fb2fa04e79e47571a

test -x "$PY" && test -d "$REPO/openstl" && test -f "$INFER" && test -f "$AUDIT"
test "$(sha256sum "$STAGE/stage_manifest.csv" | awk '{print $1}')" = "$EXPECTED_STAGE_SHA"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

for fold in 0 1 2 3 4; do
  FOLD_DIR="$MANIFESTS/fold_$fold"
  INDEX="$FOLD_DIR/simvp_trainval_index.csv"
  NORM="$FOLD_DIR/norm/simvp_train_only_norm.npz"
  case "$fold" in
    0)
      CHECKPOINT="$RUNS/runs/S_oof_fold0_seed42_20261016_attempt3/checkpoints/best.ckpt"
      EXPECTED_INDEX_SHA=4886e60f93a4490862ba8e9bbd5e8b302c35a599508bc03315a8ded530de3cc4
      EXPECTED_NORM_SHA=1dbc3be2391cb530a6b438bcf9bd8c4093252a88cbbef295a74eec97cccfa2b2
      EXPECTED_CHECKPOINT_SHA=2e9e6d50dd5693f7a856a3acb57aea96d5fe4dade67e9d5252f4f28b683ef86c
      ;;
    1)
      CHECKPOINT="$RUNS/runs/S_oof_fold1_seed42_20261016_attempt1/checkpoints/best.ckpt"
      EXPECTED_INDEX_SHA=5d436d6ebc4c3a6d622d2288afb84e1770f8977451d80c3f7fc5e1f2922a5d17
      EXPECTED_NORM_SHA=84c215afec383b4b85326c9283d64c02ae56e707129064df1dc2e80f8f74ac61
      EXPECTED_CHECKPOINT_SHA=e95a54f44c8d6e5b14935270f616e7a66c9d00921ef2c2ae4d9d2ebeac53871a
      ;;
    2)
      CHECKPOINT="$RUNS/S_oof_fold2_seed42_20261016_fd201_20261017/runs/S_oof_fold2_seed42_20261016_fd201_20261017/checkpoints/best.ckpt"
      EXPECTED_INDEX_SHA=d055cb12739ee3ae2976d22a200050e62d5419210d4b9e31112aca4c34bce4a8
      EXPECTED_NORM_SHA=c43077b59720925cb81fe5cb91268e57a0e095d8c4c0dd75b27f483d92e554b3
      EXPECTED_CHECKPOINT_SHA=7a3309bad154a5bd1383b4ddc23e95889adddb57ca4379f73014b7f226f1d0b6
      ;;
    3)
      CHECKPOINT="$RUNS/checkpoint_imports_tc6000_20261017/fold_3/best.ckpt"
      EXPECTED_INDEX_SHA=3b3d8c0e3b962a8101d96204dfd75b572fac74a2576b127ae941b20b5e450b8f
      EXPECTED_NORM_SHA=f8ff7379b4695b246150565e71651d05267e65be1d08c69f6fd6788082f2b1f2
      EXPECTED_CHECKPOINT_SHA=50f184270207695b1d69c7f2db47c33e1571395320304d4862560ff0e4430577
      ;;
    4)
      CHECKPOINT="$RUNS/checkpoint_imports_tc6000_20261017/fold_4/best.ckpt"
      EXPECTED_INDEX_SHA=7106f8aa6b3073c9ac9edb95b8c4bade6d43074a8b162eb6f57f82b53887c14d
      EXPECTED_NORM_SHA=128a92c62bbe14c242837b62d730294be26aa92a276cd334d67a67747b6b60b5
      EXPECTED_CHECKPOINT_SHA=5637925fd0d4bc092c6f5f8996b20cf356f8bcf0b4cb358768e8856aec3f03a3
      ;;
  esac
  test "$(sha256sum "$INDEX" | awk '{print $1}')" = "$EXPECTED_INDEX_SHA"
  test "$(sha256sum "$NORM" | awk '{print $1}')" = "$EXPECTED_NORM_SHA"
  test "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = "$EXPECTED_CHECKPOINT_SHA"

  OUT="$RUNS/S_oof_fold${fold}_fullval_closed_v1_20261017"
  test ! -e "$OUT"
  mkdir "$OUT"
  printf 'fold=%s\nindex_sha256=%s\nnorm_sha256=%s\ncheckpoint_sha256=%s\nstage_manifest_sha256=%s\ntest_payloads_read=0\nbuffer_payloads_read=0\ntest_or_buffer_rows=0\n' \
    "$fold" "$EXPECTED_INDEX_SHA" "$EXPECTED_NORM_SHA" "$EXPECTED_CHECKPOINT_SHA" "$EXPECTED_STAGE_SHA" > "$OUT/launch_manifest.txt"
  nvidia-smi --id=0 --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv >> "$OUT/launch_manifest.txt"

  "$PY" "$INFER" --repo "$REPO" --index "$INDEX" --norm "$NORM" \
    --checkpoint "$CHECKPOINT" --output "$OUT/bank" --stage-root "$STAGE" \
    --batch-size 4 --workers 2 --device cuda:0 --splits val
  "$PY" "$AUDIT" --index "$INDEX" --fold "$fold" --shard-dir "$OUT/bank" \
    --output "$OUT/FOLD${fold}_VALIDATION_RELOAD_AUDIT_CLOSED_V1.json"
  sha256sum "$OUT/bank/BANK_REPORT.json" "$OUT/bank/val/pred_agri_patch.npy" \
    "$OUT/bank/val/true_agri_patch.npy" "$OUT/bank/val/future_geometry_patch.npy" \
    "$OUT/bank/val/keys.csv" "$OUT/FOLD${fold}_VALIDATION_RELOAD_AUDIT_CLOSED_V1.json" \
    >> "$OUT/launch_manifest.txt"
done

echo "ALL_FULLVAL_AUDITS_CLOSED_V1_COMPLETE"
