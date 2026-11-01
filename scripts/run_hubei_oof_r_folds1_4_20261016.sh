#!/usr/bin/env bash
set -euo pipefail

BASE=/home/Data_Pool/zjnu/hubei_cot_independent_20261016/manifests/oof_fold_manifests_20261016_v5
ROOT=/home/Data_Pool/zjnu/hubei_cot_independent_20261016/oof_training
PYTHON=/home/Data_Pool/zjnu/.conda/envs/swc/bin/python
TRAINER="$BASE/input/train_hubei_cot_fixedv2.py"
MODEL_SOURCE="$BASE/input/train_cot_repaired.py"

for fold in 1 2 3 4; do
  pack="$BASE/folds/fold_${fold}/r_pack"
  out="$ROOT/R_fold${fold}_seed42_20261016"
  test -f "$pack/PACK_QA.json"
  test ! -e "$out"
  mkdir -p "$ROOT"
  CUDA_VISIBLE_DEVICES=1 "$PYTHON" "$TRAINER" \
    --pack "$pack" \
    --model-source "$MODEL_SOURCE" \
    --output "$out" \
    --seed 42 --batch 128 --epochs 50 --patience 8 \
    --allow-variable-support > "${out}.log" 2>&1
  "$PYTHON" - "$out/status.json" <<'PY'
import json, sys
from pathlib import Path
status = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if status.get("state") != "COMPLETE_VALIDATION_ONLY_TEST_CLOSED":
    raise SystemExit(f"unexpected fold completion state: {status}")
if status.get("test_payloads_read") != 0:
    raise SystemExit("test payload access detected")
print(json.dumps(status, ensure_ascii=False), flush=True)
PY
done
