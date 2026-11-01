import hashlib,json,re,subprocess
from pathlib import Path
from datetime import datetime,timezone
base=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/refine-logs/agri_direct_cot_20261020')
expected={'run_dense_observed_reference_cpu_20261021.pbs': '2aa60c61ffd6b1164c0fa8ecdd93f777230586d42d7f579f241eee8407cb65f6', 'extract_dense_observed_reference.py': 'd2df2410ff8ac042aa6b66c2659afc0ef8bc2fe97b62f991eacb2eea997feca2', 'data/dense_observed_reference_gate_20261021.json': '2006810985b814cfc0593242cca1d0cd4769004a6505decb3440dceb9493c059'}
for name,digest in expected.items():assert hashlib.sha256((base/name).read_bytes()).hexdigest()==digest,name
receipt=base/'results/dense_observed_reference_submission_20261021.json'
assert not receipt.exists(),'submission receipt exists; inspect actual queue, do not resubmit'
assert not (base/'results/dense_observed_v_reference_patches_v1').exists()
qstat='/opt/gridview/pbs/dispatcher/bin/qstat'
qsub='/opt/gridview/pbs/dispatcher/bin/qsub'
q=subprocess.check_output([qstat,'-u','slfu'],text=True)
jobs=[line for line in q.splitlines() if re.match(r'^\s*\d+\.',line)]
print('QUEUE_BEFORE\n'+q,flush=True)
assert len(jobs)<3, 'three unfinished account jobs already present'
record={'state':'SUBMITTING','utc':datetime.now(timezone.utc).isoformat(),'queue_before':q,'source_sha256':expected,'scope':'CPU-only offline V observed-reference extraction; no training/scoring'}
receipt.write_text(json.dumps(record,indent=2)+'\n')
r=subprocess.check_output([qsub,str(base/'run_dense_observed_reference_cpu_20261021.pbs')],text=True).strip()
assert re.fullmatch(r'\d+\.[A-Za-z0-9_.-]+',r),r
record.update(state='SUBMITTED',pbs_job_id=r)
receipt.write_text(json.dumps(record,indent=2)+'\n')
print('SUBMITTED '+r,flush=True)
print('QUEUE_AFTER\n'+subprocess.check_output([qstat,'-u','slfu'],text=True),flush=True)
