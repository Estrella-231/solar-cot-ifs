"""Verify the transferred pack and submit at most one PBS job for this run."""
import fcntl
import hashlib
import json
import subprocess
from pathlib import Path

root = Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907')
lock = (root/'submit.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
result = root/'submitted_job.json'
if result.exists():
    print(result.read_text())
    raise SystemExit(0)
intent = root/'submit.intent'
if intent.exists():
    raise RuntimeError('previous submission outcome uncertain; inspect qstat before any retry')
if (root/'runs/cot_repaired_seed42_v1/run_metadata.json').exists():
    raise RuntimeError('training output exists without submission record; inspect before retry')
pack = root/'data/cot_repaired_pack_20260907_v1'
assert json.loads((pack/'pack_status.json').read_text())['state'] == 'COMPLETE'
for name, expected in json.loads((pack/'SHA256.json').read_text()).items():
    h = hashlib.sha256()
    with (pack/name).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):
            h.update(b)
    assert h.hexdigest() == expected, name
for name, expected in json.loads((root/'launch_code_sha256.json').read_text()).items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
with intent.open('x') as f:
    f.write('Submission attempted once. Never remove without checking scheduler.\n')
completed = subprocess.run(['/opt/gridview/pbs/dispatcher/bin/qsub', 'scripts/run_cot_repaired.pbs'],
                           cwd=root, capture_output=True, text=True, check=True)
job = completed.stdout.strip()
assert job.endswith('.tc6000') and job.split('.')[0].isdigit(), completed
record = dict(job_id=job, state='SUBMITTED', pack_verified=True, code_verified=True)
result.write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)
