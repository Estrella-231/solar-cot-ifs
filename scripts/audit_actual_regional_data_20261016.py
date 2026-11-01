"""Read current train/val manifests and metadata; never open test arrays."""
import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path
import sys

region = sys.argv[1]
if region == 'hunan':
    root = Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/monthly_R_station16_20261013_v1')
    docs = ['data/pack_trainval_v1/pack_status.json', 'data/pack_trainval_v1/pack_audit.json', 'data/ghi_head_pack_new_r_v1/pack_manifest.json', 'results/new_monthly_R_seed42_cot_bank_v1/bank_manifest.json', 'results/seed42_v1/run_metadata.json', 'results/ghi_heads_new_r_seed42_v1/sili/run_metadata.json']
    tables = ['input/target_candidates_train_val.csv', 'data/ghi_head_pack_new_r_v1/rows.csv', 'data/pack_trainval_v1/rows.csv', 'results/new_monthly_R_seed42_cot_bank_v1/frame_keys.csv']
    docs += ['../cot_value_first_20261013/data/ghi_kt_trainval_v1/ghi_kt_join_audit.json']
elif region == 'hubei_r':
    root = Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_r_runtime_gate_20261015_v1')
    docs = ['r_pack_trainval/pack_status.json', 'r_pack_trainval/PACK_QA.json']
    tables = ['r_pack_trainval/rows.csv']
else:
    root = Path('/home/Data_Pool/zjnu/hubei_cot_independent_20261016')
    docs = ['pack/pack_report.json', 'runs/seed42/run_metadata.json', 'COMBINED_MANIFEST_REPORT.json', 'R_BANK_REPORT.json', 'FEATURE_REPORT.json', 'assets/frozen_R_20261016/run_metadata.json', 'assets/frozen_R_20261016/status.json', 'manifests/strict24frame_same_day_20261016_v2/report.json', 'norm/simvp_train_only_norm_strict24_same_day_20261016.json']
    tables = ['pack/rows.csv', 'labels/ghi15_train_val_daylight_clear.csv', 'manifests/agri_paths_trainval.csv']

out = {'region': region, 'root': str(root), 'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'documents': {}, 'tables': {}, 'test_arrays_opened': 0}
for rel in docs:
    p = root / rel
    if not p.is_file():
        out['documents'][rel] = {'missing': True}
        continue
    b = p.read_bytes()
    out['documents'][rel] = {'sha256': hashlib.sha256(b).hexdigest(), 'value': json.loads(b)}
for rel in tables:
    p = root / rel
    if not p.is_file():
        out['tables'][rel] = {'missing': True}
        continue
    counts = collections.Counter()
    bounds = {}
    months = collections.Counter()
    state_bounds = {}
    state_counts = collections.Counter()
    source_examples = {}
    with p.open(encoding='utf-8-sig', newline='') as f:
        rd = csv.DictReader(f)
        fields = rd.fieldnames
        for row in rd:
            split = row.get('split', row.get('partition', 'unspecified'))
            station = row.get('station', 'all')
            key = split + '/' + station
            counts[key] += 1
            t = next((row[k] for k in ('target_time_bjt', 'target_time_BJT', 'time_bjt', 'bjt', 'target_time_utc', 'frame_time_utc', 'time_utc', 'timestamp', 'datetime', 'date') if row.get(k)), '')
            if t:
                if key not in bounds:
                    bounds[key] = [t,t]
                bounds[key] = [min(bounds[key][0],t),max(bounds[key][1],t)]
                month = t[:7] if '-' in t[:7] else t[:4] + '-' + t[4:6]
                months[split + '/' + month] += 1
                state = row.get('state', row.get('has_valid_cpp', 'all'))
                sk = key + '/' + state
                state_counts[sk] += 1
                if sk not in state_bounds:
                    state_bounds[sk] = [t,t]
                state_bounds[sk] = [min(state_bounds[sk][0],t),max(state_bounds[sk][1],t)]
            for k in fields:
                if ('path' in k or 'source' in k) and row.get(k):
                    source_examples.setdefault(k, row[k])
    out['tables'][rel] = {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'columns': fields, 'counts': dict(counts), 'time_bounds_as_stored': bounds, 'months_as_stored': dict(sorted(months.items())), 'state_counts': dict(state_counts), 'state_time_bounds_as_stored': state_bounds, 'source_examples': source_examples}
print(json.dumps(out, ensure_ascii=True, indent=2))
