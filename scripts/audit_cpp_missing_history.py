"""Read-only comparison of repaired-source availability and old cached CPP."""
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

root = Path('/home/Data_Pool_3/wangyc/irradiance_paper/solar_cot_ifs_20260905')
records = {}
with (root/'data/cpp_aligned_20260905_v2/ledger.jsonl').open() as f:
    for line in f:
        r = json.loads(line)
        records[r['relative_path']] = r
counts = Counter()
months = defaultdict(Counter)
minutes = defaultdict(Counter)
hours = defaultdict(Counter)
sources = defaultdict(set)
groups = defaultdict(list)
for r in records.values():
    status = r['status']
    counts[status] += 1
    stamp = r['timestamp_utc']
    months[stamp[:7]][status] += 1
    minutes[stamp[14:16]][status] += 1
    hours[r['timestamp_bjt'][11:13]][status] += 1
    sources[status].add(r['source_nc'])
    groups[(status, r['station_id'], stamp[:4])].append(r)
report = dict(counts=dict(counts), unique_source_paths={k:len(v) for k,v in sources.items()},
              by_month=dict(months), by_UTC_minute=dict(minutes), by_BJT_hour=dict(hours), old_samples=[])
print(json.dumps({k:v for k,v in report.items() if k != 'old_samples'}), flush=True)
for key, rows in sorted(groups.items()):
    rows.sort(key=lambda r:r['timestamp_utc'])
    # bounded three points per status/station/year; not an exhaustive old-cache audit
    for i in sorted({0, len(rows)//2, len(rows)-1}):
        r = rows[i]
        with np.load(r['original_path'], allow_pickle=False) as z:
            item = dict(path=r['original_path'], timestamp=r['timestamp_utc'],
                        new_status=r['status'], old_cpp_present=bool(z['cpp_present']),
                        old_finite_cot_pixels=int(np.isfinite(z['cpp'][0]).sum()),
                        old_mask_pixels=int(z['cpp_valid_mask'][0].sum()),
                        source_now_exists=Path(r['source_nc']).exists())
        report['old_samples'].append(item)
print('OLD_SAMPLE_COUNTS', dict(Counter((r['new_status'],r['old_cpp_present']) for r in report['old_samples'])), flush=True)
output = root/'audits/cpp_missing_history_20260907.json'
with output.open('x') as f:
    json.dump(report, f, indent=2)
print('SAVED', output, flush=True)
