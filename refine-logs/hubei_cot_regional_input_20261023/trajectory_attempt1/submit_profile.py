"""One 2/4 GPU trajectory correctness/throughput allocation, no formal fits."""
import hashlib,json,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent;BASE=P.parent;sys.path.insert(0,str(BASE))
# Avoid importing this file recursively, whose basename is also submit_profile.
import importlib.util
spec=importlib.util.spec_from_file_location('regional_queue',BASE/'submit_profile.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
attempt=2
receipt=P/f'PROFILE_SUBMISSION_attempt{attempt}.json';intent=P/f'PROFILE_SUBMISSION_INTENT_attempt{attempt}.json'
assert not receipt.exists() and not intent.exists()
pack_record=json.loads((P/'PAIR_PACK_COMPLETE.json').read_text())
assert pack_record['state']=='TRAIN_ONLY_TRAJECTORY_PAIR_PACK_COMPLETE_NOT_TRAINED'
pack=pack_record['pack'];assert pack=='/dev/shm/hubei_trajectory_pair_pack_20261023_attempt2'
# Initial GPU caches passed; the failed CPU directory glob was repaired without
# rerunning those caches. Require the separate successful CPU repair receipt.
repair=json.loads((P/'CPU_PACK_REPAIR_attempt2.json').read_text())
assert repair['state']=='PASS_CPU_ONLY_PAIR_PACK_REPAIR' and repair['pack']==pack
cache_job=json.loads((P/'CACHE_SUBMISSION_attempt1.json').read_text())['job_id']
status=subprocess.run(['qstat','-f',cache_job],text=True,capture_output=True)
before=module.queue();assert not any(r['id']==cache_job for r in before)
if status.returncode:
    assert 'Unknown Job Id' in status.stderr,status.stderr
    terminal='PBS record purged; no live account job; verified cache receipts and CPU repair preserved. '+status.stderr
else:
    terminal=status.stdout;assert 'job_state = C' in terminal
assert len(before)<3 and sum(r['requested_gpus'] for r in before)+4<=8,before
physical=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','node23',
    'nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits'],text=True)
assert sum(int(r.split(',')[-2])<500 and int(r.split(',')[-1])<5 for r in physical.strip().splitlines())>=4
header=(BASE/'profile_node23_attempt1.pbs').read_text().split('for N in 2 4; do')[0]
header=header.replace('rregionprof','trajprofile').replace(str(BASE/'profile_attempt1.log'),str(P/f'profile_attempt{attempt}.log'))
body=f'''for N in 2 4; do
 python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$N" "{P}/train_trajectory.py" --pack {pack} --out "{P}/profile${{N}}_fp32_attempt{attempt}" --arm full --seed 42 --witness-only --witness-init "{P}/profile2_fp32_attempt1/checkpoint.pt"
done
for N in 2 4; do
 python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$N" "{P}/train_trajectory.py" --pack {pack} --out "{P}/profile${{N}}_amp_attempt{attempt}" --arm full --seed 42 --profile-only
done
python "{P}/compare_profiles.py" --attempt {attempt}
'''
script=P/f'profile_node23_attempt{attempt}.pbs';script.write_text(header+body);subprocess.run(['bash','-n',str(script)],check=True)
record=dict(queue_before=before,physical_before=physical,node='node23',gpus=4,
    purpose='2/4 GPU trajectory masked-loss, gradient, checkpoint and throughput; no formal fit',cache_job_terminal=terminal,
    code_sha256={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in
        ['trajectory_model.py','train_trajectory.py','compare_profiles.py',script.name]})
intent.write_text(json.dumps(record,indent=2));job=subprocess.check_output(['qsub',str(script)],text=True).strip()
record.update(job_id=job,queue_after=module.queue(),test_payloads_read=0);receipt.write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
