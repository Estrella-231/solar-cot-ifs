#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/Data_Pool_3/wangyc/solar-cot-ifs-fd201-20260918
ENV="$ROOT/env/simvp-cu121"
RAW=/home/Data_Pool_3/chenyi/Auxiliary_data/HuNan_AGRI_Preprocessed
ASSET="$ROOT/assets/simvp_hunan_reference_20260918"
RUN="$ROOT/runs/simvp16_hunan_fd201_continue_seed42_v1"
LOG="$ROOT/logs/simvp16_hunan_fd201_continue_seed42_v1.log"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
[ -x "$ENV/bin/python" ] || { echo "environment unavailable" >&2; exit 2; }
[ -d "$RAW/data" ] && [ -d "$RAW/aux_256" ] || { echo "Hunan raw-data contract missing" >&2; exit 2; }
[ -f "$ASSET/manifests/best.pt" ] || { echo "initial checkpoint missing" >&2; exit 2; }
mkdir -p "$ROOT/runs" "$ROOT/logs"
exec "$ENV/bin/python" "$ROOT/scripts/train_simvp_hunan_fd201_continue.py" \
  --root "$RAW" --manifests "$ASSET/manifests" --output "$RUN" --epochs 100 --workers 4 \
  --init-checkpoint "$ASSET/manifests/best.pt" \
  --init-sha256 30f528189f363fd23b0bb1aab52eb8d5c149e330c2fb9e368db3622cf3a5efc6 \
  >> "$LOG" 2>&1