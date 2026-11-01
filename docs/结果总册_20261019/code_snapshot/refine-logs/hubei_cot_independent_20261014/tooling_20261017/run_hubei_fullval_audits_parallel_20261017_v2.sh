#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/Data_Pool/zjnu/hubei_cot_independent_20261016
MANIFESTS="$ROOT/manifests/oof_fold_manifests_20261016_v5/folds"
RUNS="$ROOT/oof_training"
STAGE=/tmp/zjnu_hubei_simvp_stage_20261016
PY=/home/Data_Pool/zjnu/.conda/envs/swc/bin/python
REPO=/home/Data_Pool_3/chenyi/GHI/SimVPv2
TOOL="$RUNS/tooling_oof_merge_20261017_v5"
INFER="$TOOL/infer_hubei_simvp_trainval_patches_20261016.py"
AUDIT="$TOOL/audit_hubei_simvp_oof_fold_val_shards_20261017_v2_keys_hash.py"
EXPECTED_STAGE_SHA=73e0f91955a7e8c07d69a1e6d79c900275ff000f350a826fb2fa04e79e47571a
OUTROOT="$RUNS/fullval_audits_parallel_v2_20261017"

test -x "$PY" && test -d "$REPO/openstl" && test -f "$INFER" && test -f "$AUDIT"
test "$(sha256sum "$STAGE/stage_manifest.csv" | awk '{print $1}')" = "$EXPECTED_STAGE_SHA"
test ! -e "$OUTROOT"
mkdir "$OUTROOT"

run_fold() {
  local fold="$1" gpu="$2" fold_dir index norm checkpoint expected_index expected_norm expected_checkpoint
  fold_dir="$MANIFESTS/fold_$fold"
  index="$fold_dir/simvp_trainval_index.csv"
  norm="$fold_dir/norm/simvp_train_only_norm.npz"
  case "$fold" in
    0)
      checkpoint="$RUNS/runs/S_oof_fold0_seed42_20261016_attempt3/checkpoints/best.ckpt"
      expected_index=4886e60f93a4490862ba8e9bbd5e8b302c35a599508bc03315a8ded530de3cc4
      expected_norm=1dbc3be2391cb530a6b438bcf9bd8c4093252a88cbbef295a74eec97cccfa2b2
      expected_checkpoint=2e9e6d50dd5693f7a856a3acb57aea96d5fe4dade67e9d5252f4f28b683ef86c
      ;;
    1)
      checkpoint="$RUNS/runs/S_oof_fold1_seed42_20261016_attempt1/checkpoints/best.ckpt"
      expected_index=5d436d6ebc4c3a6d622d2288afb84e1770f8977451d80c3f7fc5e1f2922a5d17
      expected_norm=84c215afec383b4b85326c9283d64c02ae56e707129064df1dc2e80f8f74ac61
      expected_checkpoint=e95a54f44c8d6e5b14935270f616e7a66c9d00921ef2c2ae4d9d2ebeac53871a
      ;;
    2)
      checkpoint="$RUNS/S_oof_fold2_seed42_20261016_fd201_20261017/runs/S_oof_fold2_seed42_20261016_fd201_20261017/checkpoints/best.ckpt"
      expected_index=d055cb12739ee3ae2976d22a200050e62d5419210d4b9e31112aca4c34bce4a8
      expected_norm=c43077b59720925cb81fe5cb91268e57a0e095d8c4c0dd75b27f483d92e554b3
      expected_checkpoint=7a3309bad154a5bd1383b4ddc23e95889adddb57ca4379f73014b7f226f1d0b6
      ;;
    3)
      checkpoint="$RUNS/checkpoint_imports_tc6000_20261017/fold_3/best.ckpt"
      expected_index=3b3d8c0e3b962a8101d96204dfd75b572fac74a2576b127ae941b20b5e450b8f
      expected_norm=f8ff7379b4695b246150565e71651d05267e65be1d08c69f6fd6788082f2b1f2
      expected_checkpoint=50f184270207695b1d69c7f2db47c33e1571395320304d4862560ff0e4430577
      ;;
    4)
      checkpoint="$RUNS/checkpoint_imports_tc6000_20261017/fold_4/best.ckpt"
      expected_index=7106f8aa6b3073c9ac9edb95b8c4bade6d43074a8b162eb6f57f82b53887c14d
      expected_norm=128a92c62bbe14c242837b62d730294be26aa92a276cd334d67a67747b6b60b5
      expected_checkpoint=5637925fd0d4bc092c6f5f8996b20cf356f8bcf0b4cb358768e8856aec3f03a3
      ;;
    *) return 2 ;;
  esac
  test "$(sha256sum "$index" | awk '{print $1}')" = "$expected_index"
  test "$(sha256sum "$norm" | awk '{print $1}')" = "$expected_norm"
  test "$(sha256sum "$checkpoint" | awk '{print $1}')" = "$expected_checkpoint"

  local out="$OUTROOT/fold_$fold"
  mkdir "$out"
  printf 'fold=%s\ngpu=%s\nindex_sha256=%s\nnorm_sha256=%s\ncheckpoint_sha256=%s\nstage_manifest_sha256=%s\ntest_payloads_read=0\nbuffer_payloads_read=0\ntest_or_buffer_rows=0\npython=%s\n' \
    "$fold" "$gpu" "$expected_index" "$expected_norm" "$expected_checkpoint" "$EXPECTED_STAGE_SHA" "$PY" > "$out/launch_manifest.txt"
  nvidia-smi --id="$gpu" --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv >> "$out/launch_manifest.txt"
  export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
  "$PY" "$INFER" --repo "$REPO" --index "$index" --norm "$norm" \
    --checkpoint "$checkpoint" --output "$out/bank" --stage-root "$STAGE" \
    --batch-size 4 --workers 2 --device cuda:0 --splits val
  "$PY" "$AUDIT" --index "$index" --fold "$fold" --shard-dir "$out/bank" \
    --output "$out/FOLD${fold}_VALIDATION_RELOAD_AUDIT_CLOSED_V2.json"
  "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d.get("unique_validation_key_coverage")==407; assert d.get("test_payloads_read")==0 and d.get("buffer_payloads_read")==0 and d.get("test_or_buffer_rows")==0; assert d.get("status")==f"HUBEI_SIMVP_FOLD{sys.argv[2]}_CHECKPOINT_RELOAD_VALIDATION_AUDIT_PASSED_TEST_CLOSED"' \
    "$out/FOLD${fold}_VALIDATION_RELOAD_AUDIT_CLOSED_V2.json" "$fold"
  sha256sum "$out/bank/BANK_REPORT.json" "$out/bank/val/pred_agri_patch.npy" \
    "$out/bank/val/true_agri_patch.npy" "$out/bank/val/future_geometry_patch.npy" \
    "$out/bank/val/keys.csv" "$out/FOLD${fold}_VALIDATION_RELOAD_AUDIT_CLOSED_V2.json" \
    >> "$out/launch_manifest.txt"
  printf 'FOLD_%s_FULLVAL_AUDIT_PASS_TEST_CLOSED\n' "$fold" >> "$out/launch_manifest.txt"
}

pids=()
for fold in 0 1 2; do
  gpu="$fold"
  run_fold "$fold" "$gpu" > "$OUTROOT/fold_${fold}.log" 2>&1 &
  pids+=("$!")
done
p_failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then p_failed=1; fi
done
test "$p_failed" -eq 0
pids=()
run_fold 3 0 > "$OUTROOT/fold_3.log" 2>&1 &
 pids+=("$!")
run_fold 4 1 > "$OUTROOT/fold_4.log" 2>&1 &
 pids+=("$!")
p_failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then p_failed=1; fi
done
test "$p_failed" -eq 0
printf 'ALL_FIVE_FULLVAL_AUDITS_CLOSED_V2_COMPLETE\n' > "$OUTROOT/COMPLETE.txt"
