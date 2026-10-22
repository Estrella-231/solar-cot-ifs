"""Queue/physical-resource checked frozen-R history inference, no model training."""
import hashlib,json,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent;BASE=P.parent;sys.path.insert(0,str(BASE))
from submit_profile import queue
receipt=P/'CACHE_SUBMISSION_attempt1.json';intent=P/'CACHE_SUBMISSION_INTENT_attempt1.json'
assert not receipt.exists() and not intent.exists()
assert json.loads((P/'STAGE_AUDIT.json').read_text())['state']=='PASS_TRAIN_HISTORY_LOCAL_HASHES'
before=queue();assert len(before)<3 and sum(r['requested_gpus'] for r in before)+4<=8,before
physical=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','node23',
    'nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits'],text=True)
assert sum(int(r.split(',')[-2])<500 and int(r.split(',')[-1])<5 for r in physical.strip().splitlines())>=4
header=(BASE/'profile_node23_attempt1.pbs').read_text().split('for N in 2 4; do')[0]
header=header.replace('rregionprof','trajRcache').replace('walltime=00:40:00','walltime=01:00:00')
header=header.replace(str(BASE/'profile_attempt1.log'),str(P/'cache_attempt1.log'))
body=f'python "{P}/run_cache.py" --gpus "$CUDA_VISIBLE_DEVICES" --root "{P}"\n'
script=P/'cache_node23_attempt1.pbs';script.write_text(header+body);subprocess.run(['bash','-n',str(script)],check=True)
record=dict(queue_before=before,physical_before=physical,node='node23',gpus=4,
    purpose='frozen R history inference and train-only pair pack; no E/GHI training',
    code_sha256={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in
        ['cache_history_r.py','build_pair_pack.py','run_cache.py','cache_node23_attempt1.pbs']})
intent.write_text(json.dumps(record,indent=2));job=subprocess.check_output(['qsub',str(script)],text=True).strip()
record.update(job_id=job,queue_after=queue(),test_payloads_read=0);receipt.write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
