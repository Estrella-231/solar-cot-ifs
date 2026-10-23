import hashlib,importlib.util,json,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('queue_rules',P.parent/'hubei_cot_regional_input_20261023/submit_profile.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
receipt=P/'TRAIN_SUBMISSION_attempt1.json';intent=P/'TRAIN_SUBMISSION_INTENT_attempt1.json'
assert not receipt.exists() and not intent.exists()
assert json.loads((P/'PACK_AUDIT.json').read_text())['state']=='PASS_CAUSAL_PACK_RECONSTRUCTION'
before=m.queue();assert len(before)<3 and sum(r['requested_gpus'] for r in before)+3<=8
physical=subprocess.check_output(['ssh','-o','BatchMode=yes','node23','nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits'],text=True)
assert sum(int(r.split(',')[-2])<500 and int(r.split(',')[-1])<5 for r in physical.strip().splitlines())>=3
script=P/'train.pbs';header=(P/'replay.pbs').read_text().split('python -u ')[0].replace('cotdomain','cotdomainfit').replace('walltime=00:10:00','walltime=00:20:00').replace(str(P/'replay.log'),str(P/'train.log'))
script.write_text(header+f'python -u "{P}/run_train.py"\n');subprocess.run(['bash','-n',str(script)],check=True)
record=dict(queue_before=before,physical_before=physical,node='node23',server='ZJNU college',gpus=3,purpose='Profile first, then three independent 600-update COT-domain correction seeds; F roles only; no GHI fitting',hashes={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['TRAINING_CONTRACT.md','train_corrector.py','run_train.py','PACK_COMPLETE.json','PACK_AUDIT.json','train.pbs']})
intent.write_text(json.dumps(record,indent=2));job=subprocess.check_output(['qsub',str(script)],text=True).strip();record.update(job_id=job,queue_after=m.queue());receipt.write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
