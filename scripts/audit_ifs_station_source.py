"""Bounded audit of the actual station-cache IFS source, without inventing release times."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import xarray as xr

root = Path('/home/Data_Pool_3/chenyi/Auxiliary_data/HuNan_IFS_Station16')
summary = json.loads((root/'summary.json').read_text())
with (root/'index.csv').open(newline='') as f:
    reader = csv.DictReader(f)
    columns = reader.fieldnames
    rows = list(reader)
report = dict(summary=summary, index_columns=columns, index_rows=len(rows),
              lead_hour_counts=dict(Counter(r['lead_hour'] for r in rows)),
              index_sha256=hashlib.sha256((root/'index.csv').read_bytes()).hexdigest(),
              release_fields=[c for c in columns if any(s in c.lower() for s in ('release','publish','available'))],
              source_examples=[])
assert report['index_sha256'] == summary['manifest_sha256']
for position in (0, len(rows)//2, len(rows)-1):
    path = Path(rows[position]['source_path'])
    with xr.open_dataset(path, decode_times=False) as ds:
        example = dict(path=str(path), bytes=path.stat().st_size, attrs=dict(ds.attrs),
            dimensions=dict(ds.sizes), variables={k:dict(dims=list(v.dims),attrs=dict(v.attrs)) for k,v in ds.variables.items()},
            coordinates={k:ds[k].values.tolist() for k in ds.coords if ds[k].size < 200})
    report['source_examples'].append(example)
report['causal_gate'] = 'BLOCKED_NO_AUDITED_RELEASE_RECORDS'
report['reason'] = 'Station cache has only steps1-4; no index release records. Raw-file conversion/mtime is not forecast release time.'
output = Path('/home/Data_Pool_3/wangyc/irradiance_paper/solar_cot_ifs_20260905/audits/ifs_station_source_20260907.json')
output.parent.mkdir(exist_ok=True, parents=True)
with output.open('x') as f:
    json.dump(report, f, indent=2, default=str)
print(output, flush=True)
print(json.dumps({k:report[k] for k in ('index_rows','lead_hour_counts','release_fields','causal_gate')}), flush=True)
