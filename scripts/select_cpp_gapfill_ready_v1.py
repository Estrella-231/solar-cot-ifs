"""Freeze ready train CPP-gap frames, excluding already produced pilot frames."""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--input-audit',type=Path,required=True)
p.add_argument('--existing-report',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
aud=json.loads(a.input_audit.read_text())
report=json.loads(a.existing_report.read_text())
assert aud['test_used'] is False and aud['manifest_sha256']
already={x['utc'].replace('-','').replace(':','').replace('T','') for x in report['frames']}
ready=[x for x in aud['items'] if x['split']=='train' and x['status']=='ready' and x['era_ok']]
times=sorted({x['utc'] for x in ready if x['utc'] not in already})
if len(times)>27270:
    raise RuntimeError('unexpected count')
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps([f'{t[:4]}-{t[4:6]}-{t[6:8]}T{t[8:10]}:{t[10:12]}:00' for t in times],indent=2)+'\n')
identity={'frames':len(times),'manifest_sha256':aud['manifest_sha256'],
          'input_audit_sha256':hashlib.sha256(a.input_audit.read_bytes()).hexdigest(),
          'excluded_existing_pilot_frames':len(already),'test_used':False,
          'time_source':'missing-input audit UTC stamps'}
a.output.with_name('gapfill_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
print(json.dumps(identity),flush=True)
