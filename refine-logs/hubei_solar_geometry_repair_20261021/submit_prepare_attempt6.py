import json,subprocess,sys
from pathlib import Path
base=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_solar_geometry_repair_20261021')
sys.path.insert(0,str(base/'code_pipeline_attempt6'))
import queue_controller as q
pbs=base/'prepare_data_attempt6.pbs'
assert pbs.is_file() and not (base/'prepared_attempt6').exists()
src=json.loads((base/'MISSING_R_SOURCE_AUDIT_attempt6.json').read_text())
transfer=json.loads((base/'TRANSFER_AUDIT_attempt6.json').read_text())
assert src['state']=='PASS_SOURCE_R_ONLY_TRAIN_VAL' and src['count']==202
assert transfer['state']=='PASS_TRANSFERRED_R_ONLY_TRAIN_VAL_SHA256' and transfer['count']==202 and transfer['all_hashes_match']
pre=q.queue()
print('QUEUE_BEFORE',json.dumps(pre,sort_keys=True),flush=True)
if len(pre)>=3: raise RuntimeError('three unfinished account jobs; refusing submission')
if any(v.get('name')=='hb_solar_data6' for v in pre.values()): raise RuntimeError('attempt6 data job already active')
p=subprocess.run([q.QSUB,str(pbs)],capture_output=True,text=True)
print('QSUB_RESULT',p.returncode,p.stdout.strip(),p.stderr.strip(),flush=True)
if p.returncode: raise SystemExit(p.returncode)
jid=p.stdout.strip()
if not jid or not jid[0].isdigit(): raise RuntimeError('unrecognized qsub result; inspect queue before retry')
after=q.queue()
print('SUBMITTED',jid,'QUEUE_AFTER',json.dumps(after,sort_keys=True),flush=True)
if jid not in after: print('JOB_NOT_IN_IMMEDIATE_QSTAT; inspect qstat -f before any retry',flush=True)
