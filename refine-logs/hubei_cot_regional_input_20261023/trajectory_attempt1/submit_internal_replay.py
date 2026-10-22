"""One new bounded diagnostic; original training and outer validation stay sealed."""
import hashlib,importlib.util,json,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent;BASE=P.parent
spec=importlib.util.spec_from_file_location('regional_queue',BASE/'submit_profile.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
receipt=P/'INTERNAL_REPLAY_SUBMISSION_attempt1.json';intent=P/'INTERNAL_REPLAY_SUBMISSION_INTENT_attempt1.json'
assert not receipt.exists() and not intent.exists()
assert json.loads((P/'FITS_FINAL_AUDIT_attempt1.json').read_text())['state']=='PASS_SIX_FROZEN_FITS_BUDGET_BINDING_AND_INTERNAL_SELECTION'
before=module.queue();assert len(before)<3 and sum(r['requested_gpus'] for r in before)+4<=8
physical=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','node23','nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits'],text=True)
assert sum(int(r.split(',')[-2])<500 and int(r.split(',')[-1])<5 for r in physical.strip().splitlines())>=4
header=(P/'eval_node23_attempt1.pbs').read_text().split('python "'+str(P/'run_eval.py')+'"')[0]
assert 'run_eval.py' not in header
header=header.replace('trajeval','trajinternal').replace('eval_attempt1.log','internal_replay_attempt1.log')
script=P/'internal_replay_node23_attempt1.pbs';script.write_text(header+f'python "{P}/run_internal_replay.py" --root "{P}" --gpus "$CUDA_VISIBLE_DEVICES"\n')
subprocess.run(['bash','-n',str(script)],check=True)
record=dict(queue_before=before,physical_before=physical,node='node23',server='ZJNU college',gpus=4,new_training_runs=0,
    purpose='frozen six-model train-internal thickness and motion attribution only',
    code_sha256={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['internal_replay.py','run_internal_replay.py','internal_replay_node23_attempt1.pbs']})
intent.write_text(json.dumps(record,indent=2));job=subprocess.check_output(['qsub',str(script)],text=True).strip()
record.update(job_id=job,queue_after=module.queue(),test_payloads_read=0);receipt.write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
