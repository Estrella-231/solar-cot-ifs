"""Submit exact paired budget after real GPU profile and live resource check."""
import hashlib,importlib.util,json,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent;BASE=P.parent
spec=importlib.util.spec_from_file_location('regional_queue',BASE/'submit_profile.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
receipt=P/'FITS_SUBMISSION_attempt1.json';intent=P/'FITS_SUBMISSION_INTENT_attempt1.json'
assert not receipt.exists() and not intent.exists()
profile=json.loads((P/'PROFILE_DECISION_attempt1.json').read_text());assert profile['state']=='PASS_TRAJECTORY_2_4_GPU_CORRECTNESS_AND_SPEED'
world=profile['selected_world'];assert world in (2,4)
profile_job=json.loads((P/'PROFILE_SUBMISSION_attempt1.json').read_text())['job_id']
status=subprocess.run(['qstat','-f',profile_job],text=True,capture_output=True)
before=module.queue();assert not any(r['id']==profile_job for r in before)
if status.returncode:
    assert 'Unknown Job Id' in status.stderr,status.stderr
    terminal='PBS record purged; no live account job; completed correctness/profile receipts verified. '+status.stderr
else:
    terminal=status.stdout;assert 'job_state = C' in terminal and 'exit_status = 0' in terminal
assert len(before)<3 and sum(r['requested_gpus'] for r in before)+world<=8,before
physical=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','node23',
    'nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits'],text=True)
assert sum(int(r.split(',')[-2])<500 and int(r.split(',')[-1])<5 for r in physical.strip().splitlines())>=world
header=(BASE/'profile_node23_attempt1.pbs').read_text().split('for N in 2 4; do')[0]
header=header.replace('rregionprof','trajpaired').replace('walltime=00:40:00','walltime=03:00:00')
header=header.replace('gpus=4',f'gpus={world}').replace('-eq 4',f'-eq {world}')
header=header.replace(str(BASE/'profile_attempt1.log'),str(P/'fits_attempt1.log'))
script=P/'fits_node23_attempt1.pbs';script.write_text(header+f'python "{P}/run_paired_fits.py" --root "{P}" --world {world}\n')
subprocess.run(['bash','-n',str(script)],check=True)
record=dict(queue_before=before,physical_before=physical,node='node23',gpus=world,profile_job_terminal=terminal,
    purpose='six fixed-budget seed-matched trajectory fits; internal selection only',global_batch=128,max_updates_per_fit=2000,
    code_sha256={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in
        ['trajectory_model.py','train_trajectory.py','run_paired_fits.py','EXPERIMENT_PLAN_20261023.md','fits_node23_attempt1.pbs']})
intent.write_text(json.dumps(record,indent=2));job=subprocess.check_output(['qsub',str(script)],text=True).strip()
record.update(job_id=job,queue_after=module.queue(),test_payloads_read=0);receipt.write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
