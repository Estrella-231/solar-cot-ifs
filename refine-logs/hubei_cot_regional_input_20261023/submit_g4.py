"""One queue-checked four-GPU frozen comparison, no training or GHI labels."""
import hashlib,json,subprocess
from pathlib import Path
from submit_profile import queue
P=Path(__file__).resolve().parent
receipt=P/'G4_SUBMISSION_attempt1.json';intent=P/'G4_SUBMISSION_INTENT_attempt1.json'
assert not receipt.exists() and not intent.exists()
assert json.loads((P/'G4_STAGE_AUDIT.json').read_text())['state']=='PASS_G4_HISTORY_LOCAL_HASHES'
assert json.loads((P/'r_region_seed42_attempt1/training_complete.json').read_text())['state']=='R_REGION_TRAIN_COMPLETE_VAL_PREDICTIONS_SAVED'
before=queue();assert len(before)<3 and sum(r['requested_gpus'] for r in before)+4<=8,before
physical=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','node23',
    'nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits'],text=True)
assert sum(int(r.split(',')[-2])<500 and int(r.split(',')[-1])<5 for r in physical.strip().splitlines())>=4,physical
header=(P/'profile_node23_attempt1.pbs').read_text().split('for N in 2 4; do')[0]
header=header.replace('rregionprof','g4cotroutes').replace('ppn=8','ppn=16').replace('profile_attempt1.log','g4_attempt1.log')
body='''export S_PYTHON=/public/home/slfu/ttzhou/swc/irradiance_forecast/hubei_cot_independent_20261017/.venv_simvp/bin/python
python "$BASE/run_g4.py" --gpus "$CUDA_VISIBLE_DEVICES" --root "$BASE" --data /dev/shm/hubei_g4_region_20261023_attempt1 --out "$BASE/g4_attempt1"
'''
script=P/'g4_node23_attempt1.pbs';script.write_text(header+body)
subprocess.run(['bash','-n',str(script)],check=True)
record=dict(queue_before=before,physical_before=physical,node='node23',gpus=4,
    purpose='frozen future COT reference comparison; no training or GHI',
    selection_sha256=hashlib.sha256((P/'G4_SELECTION.json').read_bytes()).hexdigest(),
    code_sha256={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in
     ('g4_inference.py','g4_motion.py','score_g4.py','run_g4.py','g4_node23_attempt1.pbs')})
intent.write_text(json.dumps(record,indent=2));job=subprocess.check_output(['qsub',str(script)],text=True).strip()
record.update(job_id=job,queue_after=queue(),test_payloads_read=0);receipt.write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
