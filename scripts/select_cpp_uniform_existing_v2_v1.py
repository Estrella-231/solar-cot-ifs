"""Select already-labelled train/val times disjoint from gapfill and audit pilots."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--manifest',type=Path,required=True)
p.add_argument('--gap-times',type=Path,required=True)
p.add_argument('--pilot-times',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
all_times=set()
with a.manifest.open(encoding='utf-8-sig',newline='') as f:
    for row in csv.DictReader(f):
        if row['split'] in ('train','val'):
            for rel in row['data_relpaths'].split('|'):
                all_times.add(Path(rel).stem)
def canonical(x):
    value=x.replace('-','').replace(':','').replace('T','')
    return value[:12]
gap={canonical(x) for x in json.loads(a.gap_times.read_text())}
pilot={canonical(x) for x in json.loads(a.pilot_times.read_text())}
if not gap <= all_times or not pilot <= all_times:
    raise RuntimeError('gap/pilot timestamp outside train/val manifest')
remaining=sorted(all_times-gap-pilot)
if len(all_times)!=45940 or len(gap)!=27270 or len(pilot)!=1000:
    raise RuntimeError(f'unexpected frame counts: all={len(all_times)} gap={len(gap)} pilot={len(pilot)}')
a.output.parent.mkdir(parents=True,exist_ok=True)
times=[f'{t[:4]}-{t[4:6]}-{t[6:8]}T{t[8:10]}:{t[10:12]}:00' for t in remaining]
a.output.write_text(json.dumps(times,indent=2)+'\n')
identity={'state':'FROZEN_DISJOINT_EXISTING_LABEL_TIMES','frames':len(times),
          'all_train_val_unique':len(all_times),'gapfill_excluded':len(gap),'pilot_excluded':len(pilot),
          'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
          'gap_times_sha256':hashlib.sha256(a.gap_times.read_bytes()).hexdigest(),
          'pilot_times_sha256':hashlib.sha256(a.pilot_times.read_bytes()).hexdigest(),
          'time_source':'UTC matched filenames; train and validation only','test_used':False}
a.output.with_name('identity_existing_v2.json').write_text(json.dumps(identity,indent=2)+'\n')
print(json.dumps(identity),flush=True)
