import hashlib,json,os,subprocess,sys,time
from pathlib import Path
import torch

P=Path(__file__).resolve().parent
assert torch.cuda.device_count()==3,os.environ.get('CUDA_VISIBLE_DEVICES')
assert not (P/'training_attempt1').exists()
checks=[]
for gpu in range(3):
    torch.manual_seed(42)
    with torch.cuda.device(gpu):
        a=torch.randn(16,16,device=f'cuda:{gpu}',requires_grad=True)
        loss=(a@a.T).square().mean();loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(a.grad).all()
        checks.append(dict(gpu=gpu,name=torch.cuda.get_device_name(gpu),loss=float(loss)))
(P/'GPU_WITNESS.json').write_text(json.dumps(dict(devices=checks,physical_gpus=os.environ['CUDA_VISIBLE_DEVICES'],
    pbs_job=os.environ.get('PBS_JOBID'),hashes={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['PLAN.md','model.py','train_station.py','run.py']}),indent=2))
start=time.perf_counter(); jobs=[]
for gpu,st in enumerate(['Niushou','Wangzhai','Mizhuang']):
    log=(P/f'{st}.log').open('w')
    job=subprocess.Popen([sys.executable,'-u',str(P/'train_station.py'),'--station',st,'--gpu',str(gpu)],stdout=log,stderr=subprocess.STDOUT)
    jobs.append((st,job,log))
statuses=[]
for st,job,log in jobs:
    code=job.wait();log.close();statuses.append(dict(station=st,exit_code=code))
(P/'RUN_STATUS.json').write_text(json.dumps(dict(workers=statuses,seconds=time.perf_counter()-start),indent=2))
assert all(r['exit_code']==0 for r in statuses),statuses
print('ALL_STATIONS_COMPLETE',time.perf_counter()-start,flush=True)
