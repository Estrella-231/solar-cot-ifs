#!/usr/bin/env bash
set -euo pipefail

# One fold per invocation. Launch only after checking the target GPUs are idle.
fold="${1:?usage: run_hubei_simvp_oof_fold_20261016.sh FOLD ATTEMPT [GPU_CSV]}"
attempt="${2:?usage: run_hubei_simvp_oof_fold_20261016.sh FOLD ATTEMPT [GPU_CSV]}"
gpu_csv="${3:-1,2,3}"
[[ "$fold" =~ ^[0-4]$ ]] || { echo "invalid fold: $fold" >&2; exit 2; }
[[ "$attempt" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "invalid attempt tag" >&2; exit 2; }
[[ "$gpu_csv" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "invalid GPU list" >&2; exit 2; }

ROOT=/home/Data_Pool/zjnu/hubei_cot_independent_20261016
FOLDROOT="$ROOT/manifests/oof_fold_manifests_20261016_v5/folds/fold_${fold}"
CONFIG="$ROOT/oof_training/configs/hbagri_simvp_oof_fold${fold}_seed42_20261016.py"
NORM="$FOLDROOT/norm/simvp_train_only_norm.npz"
INDEX="$FOLDROOT/simvp_trainval_index.csv"
RUN="$ROOT/oof_training/S_fold${fold}_seed42_20261016_${attempt}"
REPO=/home/Data_Pool_3/chenyi/GHI/SimVPv2
PY=/home/Data_Pool/zjnu/.conda/envs/swc/bin
TRAINER="$ROOT/scripts/run_hubei_simvp_frozen.py"
STAGE=/tmp/zjnu_hubei_simvp_stage_20261016

for path in "$CONFIG" "$NORM" "$INDEX" "$TRAINER" "$REPO/tools/train.py" "$STAGE"; do
    [[ -e "$path" ]] || { echo "missing required input: $path" >&2; exit 3; }
done
[[ ! -e "$RUN" ]] || { echo "refusing to overwrite existing run: $RUN" >&2; exit 4; }

IFS=, read -r -a gpu_ids <<< "$gpu_csv"
local_gpu_ids=()
for ((i = 0; i < ${#gpu_ids[@]}; i++)); do
    local_gpu_ids+=("$i")
done
mkdir -p "$RUN/cwd"
{
    printf 'fold=%s\nattempt=%s\ngpus=%s\n' "$fold" "$attempt" "$gpu_csv"
    printf 'trainer_visible_gpu_ids=%s\n' "$(IFS=,; echo "${local_gpu_ids[*]}")"
    printf 'index_sha256=%s\nnorm_sha256=%s\nconfig_sha256=%s\ntrainer_sha256=%s\n' \
        "$(sha256sum "$INDEX" | awk '{print $1}')" \
        "$(sha256sum "$NORM" | awk '{print $1}')" \
        "$(sha256sum "$CONFIG" | awk '{print $1}')" \
        "$(sha256sum "$TRAINER" | awk '{print $1}')"
    printf 'repo=%s\nstage=%s\n' "$REPO" "$STAGE"
} > "$RUN/launch_manifest.txt"

export SIMVP_REPO="$REPO"
export HUBEI_SIMVP_CONFIG="$CONFIG"
export HUBEI_SIMVP_NORM="$NORM"
export HUBEI_SIMVP_RUN_CWD="$RUN/cwd"
export CUDA_VISIBLE_DEVICES="$gpu_csv"
export OMP_NUM_THREADS=1
cd "$RUN/cwd"

"$PY/torchrun" --standalone --nnodes=1 --nproc_per_node="${#gpu_ids[@]}" \
    "$TRAINER" --dist --gpus "${local_gpu_ids[@]}" \
    --res_dir "$ROOT/oof_training/runs" \
    --ex_name "S_oof_fold${fold}_seed42_20261016_${attempt}" 2>&1 | tee "$RUN/train.log"
