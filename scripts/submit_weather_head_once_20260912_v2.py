"""Single idempotent submission; account-wide PBS and GPU resource gate."""
import fcntl
import re
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907')
PBS='/opt/gridview/pbs/dispatcher/bin/'
lock=(ROOT/'submit.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
receipt=ROOT/'audits/weather_head_submission_20260912_v2.json'
intent=ROOT/'audits/weather_head_submission_20260912_v2.intent'
if receipt.exists():
    print(receipt.read_text());raise SystemExit(0)
assert not intent.exists(), 'submission outcome uncertain: inspect rather than retry'
assert not (ROOT/'runs/weather_coverage_head_seed42_20260912_v2').exists()
assert not (ROOT/'weather_head_20260912_v2_pbs.log').exists()
manifest=json.loads((ROOT/'audits/weather_head_launch_sha256_20260912_v2.json').read_text())
for rel,expected in manifest.items():
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==expected,rel
collection=json.loads((ROOT/'data/weather_clp_trainval_20260912_v1/collection.json').read_text())
assert collection['state']=='COMPLETE_CLP_LABEL_READ' and collection['test_used'] is False
query=subprocess.run([PBS+'qstat','-u','slfu'],capture_output=True,text=True,check=True)
jobs=[]
if query.stdout.strip():
    ids=re.findall(r'^\s*(\d+\.[\w.-]+)\s+',query.stdout,re.M)
    assert ids, 'nonempty queue response not understood'
    assert len(set(ids))==len(ids)
    for job_id in ids:
        detail=subprocess.run([PBS+'qstat','-f',job_id],capture_output=True,text=True,check=True).stdout
        fields={};key=None
        for line in detail.splitlines():
            match=re.match(r'^\s{4}(\S+) = (.*)$',line)
            if match:key=match[1];fields[key]=match[2]
            elif key and line[:1].isspace():fields[key]+=line.strip()
        assert fields.get('Job_Owner','').split('@')[0]=='slfu'
        state=fields['job_state']
        if state in ('C','F'):continue
        nodes=fields.get('Resource_List.nodes');ngpus=fields.get('Resource_List.ngpus')
        count=0
        if ngpus is not None:count=int(ngpus)
        elif nodes is not None:
            for chunk in nodes.split('+'):
                parts=chunk.split(':');n=int(parts[0]) if parts[0].isdigit() else 1
                gpu=[x for x in parts if x.startswith('gpus=')]
                count+=n*int(gpu[0].split('=')[1]) if gpu else 0
        else:raise RuntimeError('cannot establish unfinished job GPU request')
        jobs.append(dict(job_id=job_id,state=state,gpus=count,queue_detail=detail))
assert len(jobs)<3, f'account PBS cap: {jobs}'
assert sum(j['gpus'] for j in jobs)+1<=8, f'account GPU cap: {jobs}'
with intent.open('x') as stream:stream.write('One submission attempted; do not remove or blindly retry.\n')
result=subprocess.run([PBS+'qsub','scripts/run_weather_head_20260912_v2.pbs'],cwd=ROOT,capture_output=True,text=True)
record=dict(server_utc=datetime.now(timezone.utc).isoformat(),queue_before=query.stdout,
            counted_jobs=jobs,code_sha256=manifest,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
receipt.write_text(json.dumps(record,indent=2))
assert result.returncode==0, record
job_id=result.stdout.strip()
assert job_id.endswith('.tc6000') and job_id.split('.')[0].isdigit(),record
record['job_id']=job_id
after=subprocess.run([PBS+'qstat','-u','slfu'],capture_output=True,text=True)
record.update(queue_after=after.stdout,queue_after_returncode=after.returncode)
receipt.write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
