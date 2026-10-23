"""Reuse approved PBS resource check; no scheduler submission on a full queue."""
import hashlib,importlib.util,json,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent;BASE=P.parent/'hubei_cot_regional_input_20261023'
spec=importlib.util.spec_from_file_location('regional_queue',BASE/'submit_profile.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
receipt=P/'SUBMISSION_attempt1.json';intent=P/'SUBMISSION_INTENT_attempt1.json';assert not receipt.exists() and not intent.exists()
assert json.loads((P/'ENGINEERING_CHECK.json').read_text())['state']=='CPU_ENGINEERING_PASS'
before=m.queue();assert len(before)<3 and sum(j['requested_gpus'] for j in before)+3<=8
physical=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','node23',
    'nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits'],text=True)
assert sum(int(r.split(',')[-2])<500 and int(r.split(',')[-1])<5 for r in physical.strip().splitlines())>=3
old=P.parent/'hubei_ghi_dual_domain_20261023/replay.pbs'
header=old.read_text().split('python -u ')[0].replace('ghidualdomain','cottaskaligned').replace(str(old.parent/'replay.log'),str(P/'replay.log'))
script=P/'replay.pbs';script.write_text(header+f'python -u "{P}/run.py"\n')
subprocess.run(['bash','-n',str(script)],check=True)
record=dict(queue_before=before,physical_before=physical,server='ZJNU college',node='node23',gpus=3,
    purpose='81 fixed-budget train-date OOF fits, independent station tasks; anticipated minutes, not hours',
    hashes={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['PLAN.md','model.py','train_station.py','run.py','ENGINEERING_CHECK.json','replay.pbs']})
intent.write_text(json.dumps(record,indent=2));job=subprocess.check_output(['qsub',str(script)],text=True).strip()
record.update(job_id=job,queue_after=m.queue(),test_payloads_read=0);receipt.write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
