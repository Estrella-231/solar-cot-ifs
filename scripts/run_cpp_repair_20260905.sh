#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/Data_Pool_3/wangyc/irradiance_paper/solar_cot_ifs_20260905
PY=/home/Data_Pool/zjnu/.conda/envs/swc/bin/python
COMBINED=/home/Data_Pool_3/chenyi/Auxiliary_data/HuNan_Station16_Combined
CPP=/home/Data_Pool_3/data/Global_CPP/FY/FY4B/Result/CPP
echo "START $(date -Is)"
mkdir -p "$ROOT/code" "$ROOT/configs" "$ROOT/data"
for file in repair_cpp_cache.py cpp_grid_mapping.py cpp_quality_mask.py repartition_cot_manifest.py load_repaired_cpp.py; do
  cp "/tmp/$file" "$ROOT/code/$file"
done
cp /tmp/cot_contract.json "$ROOT/configs/cot_contract.json"
cp /tmp/cpp_repair_smoke_inventory.txt "$ROOT/data/cpp_repair_smoke_inventory.txt"
echo "SMOKE $(date -Is)"
"$PY" -u "$ROOT/code/repair_cpp_cache.py" --combined-root "$COMBINED" --cpp-root "$CPP" --contract "$ROOT/configs/cot_contract.json" --output "$ROOT/data/cpp_repair_smoke_20260905_v2" --workers 2 --inventory-source "$ROOT/data/cpp_repair_smoke_inventory.txt"
"$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["processed"]==18 and r["counts"]=={"ok":18}, r' "$ROOT/data/cpp_repair_smoke_20260905_v2/status.json"
echo "FULL $(date -Is)"
"$PY" -u "$ROOT/code/repair_cpp_cache.py" --combined-root "$COMBINED" --cpp-root "$CPP" --contract "$ROOT/configs/cot_contract.json" --output "$ROOT/data/cpp_aligned_20260905_v2" --workers 2
echo "DONE $(date -Is)"
