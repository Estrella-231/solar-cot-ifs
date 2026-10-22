"""Submit once, only after six frozen fits and live physical/queue verification."""
import hashlib,importlib.util,json,subprocess
import sys
from pathlib import Path
P=Path(__file__).resolve().parent;BASE=P.parent
spec=importlib.util.spec_from_file_location('regional_queue',BASE/'submit_profile.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
receipt=P/'EVAL_SUBMISSION_attempt1.json';intent=P/'EVAL_SUBMISSION_INTENT_attempt1.json'
assert not receipt.exists() and not intent.exists()
arm=json.loads((P/'fits_attempt1/ARM_SELECTION.json').read_text())
assert arm['state']=='SIX_FITS_COMPLETE_ARM_FROZEN_BEFORE_OUTER_VAL' and len(arm['runs'])==6
for r in arm['runs']:
    assert hashlib.sha256((Path(r['run'])/'best.pt').read_bytes()).hexdigest()==r['best_sha256']
subprocess.run([sys.executable,str(P/'audit_fits.py')],check=True)
training_job=json.loads((P/'FITS_SUBMISSION_attempt1.json').read_text())['job_id']
before=module.queue();assert not any(r['id']==training_job for r in before)
status=subprocess.run(['qstat','-f',training_job],text=True,capture_output=True)
if status.returncode:
    assert 'Unknown Job Id' in status.stderr,status.stderr
    terminal='PBS record purged; full active queue absent and six completed fits/checkpoints verified'
else:
    terminal=status.stdout;assert 'job_state = C' in terminal and 'exit_status = 0' in terminal
assert len(before)<3 and sum(r['requested_gpus'] for r in before)+4<=8,before
physical=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','node23',
    'nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits'],text=True)
assert sum(int(r.split(',')[-2])<500 and int(r.split(',')[-1])<5 for r in physical.strip().splitlines())>=4
header=(P/'fits_node23_attempt1.pbs').read_text().split('python "/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_cot_regional_input_20261023/trajectory_attempt1/run_paired_fits.py"')[0]
assert 'run_paired_fits.py' not in header
header=header.replace('trajpaired','trajeval').replace('walltime=03:00:00','walltime=00:40:00').replace('fits_attempt1.log','eval_attempt1.log')
script=P/'eval_node23_attempt1.pbs'
script.write_text(header+f'python "{P}/run_eval.py" --root "{P}" --gpus "$CUDA_VISIBLE_DEVICES"\n'
    +f'python "{P}/score_eval.py" --root "{P}" --baseline "{BASE}/g4_attempt1/scores/paired_pixels.csv" --out "{P}/eval_attempt1/scores"\n')
subprocess.run(['bash','-n',str(script)],check=True)
record=dict(queue_before=before,physical_before=physical,node='node23',server='ZJNU college',gpus=4,
    training_job_terminal=terminal,arm_selection_sha256=hashlib.sha256((P/'fits_attempt1/ARM_SELECTION.json').read_bytes()).hexdigest(),
    code_sha256={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['trajectory_model.py','build_eval_pack.py','infer_eval.py','run_eval.py','score_eval.py','eval_node23_attempt1.pbs']})
intent.write_text(json.dumps(record,indent=2));job=subprocess.check_output(['qsub',str(script)],text=True).strip()
record.update(job_id=job,queue_after=module.queue(),test_payloads_read=0,GHI_training_started=False)
receipt.write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
