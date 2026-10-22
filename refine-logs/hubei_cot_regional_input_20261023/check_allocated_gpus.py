"""Check only scheduler-assigned GPUs; refuse to train over live occupants."""
import os,subprocess,json
ids=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(ids)==4 and len(set(ids))==4
rows=[]
for i in ids:
    raw=subprocess.check_output(['nvidia-smi','-i',i,'--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    index,name,used,util=[s.strip() for s in raw.strip().split(',')]
    rows.append(dict(index=index,name=name,used_MiB=int(used),utilization=int(util)))
    assert int(used)<500 and int(util)<5, ('Assigned GPU is occupied; do not disturb it',rows[-1])
print(json.dumps(dict(state='ASSIGNED_GPU_FREE_BEFORE_TORCH',pbs_job_id=os.environ.get('PBS_JOBID'),gpus=rows)),flush=True)
