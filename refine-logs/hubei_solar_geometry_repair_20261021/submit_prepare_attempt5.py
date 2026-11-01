import subprocess,sys,json
from pathlib import Path
BASE=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_solar_geometry_repair_20261021')
sys.path.insert(0,str(BASE/'code_pipeline_attempt5'))
import queue_controller as q
assert (BASE/'prepare_data_attempt5.pbs').is_file()
assert not (BASE/'prepared_attempt5').exists()
pre=q.queue()
print('QUEUE_BEFORE',json.dumps(pre,sort_keys=True),flush=True)
if len(pre)>=3: raise RuntimeError('three unfinished account jobs; refusing submission')
if any(v.get('name')=='hb_solar_data5' for v in pre.values()): raise RuntimeError('attempt5 data job already active')
p=subprocess.run([q.QSUB,str(BASE/'prepare_data_attempt5.pbs')],capture_output=True,text=True)
print('QSUB_RESULT',p.returncode,p.stdout.strip(),p.stderr.strip(),flush=True)
if p.returncode: raise SystemExit(p.returncode)
jid=p.stdout.strip()
if not jid or not jid[0].isdigit(): raise RuntimeError('unrecognized qsub result; inspect queue before any retry')
after=q.queue()
print('SUBMITTED',jid,'QUEUE_AFTER',json.dumps(after,sort_keys=True),flush=True)
if jid not in after: print('JOB_NOT_IN_IMMEDIATE_QSTAT; inspect qstat -f before any retry',flush=True)
