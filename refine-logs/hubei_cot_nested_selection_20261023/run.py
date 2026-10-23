"""Four-GPU allocation, profile 2 vs 4, then a hard inner/selection/outer barrier."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
import torch
P=Path(__file__).resolve().parent
assert torch.cuda.device_count()==4
def worker(mode,station,outer,gpu,tag=''):
    return [sys.executable,'-u',str(P/'worker.py'),'--mode',mode,'--station',station,'--outer',str(outer),'--gpu',str(gpu),'--tag',tag]
def wave(tasks,slots,prefix):
    start=time.perf_counter();pending=list(tasks);running={};results=[]
    while pending or running:
        for gpu in range(slots):
            if gpu not in running and pending:
                mode,st,outer,tag=pending.pop(0);log=(P/f'{prefix}_{st}_{outer}_{tag}.log').open('w')
                running[gpu]=(subprocess.Popen(worker(mode,st,outer,gpu,tag),stdout=log,stderr=subprocess.STDOUT),log,(st,outer,tag))
        for gpu,(proc,log,task) in list(running.items()):
            code=proc.poll()
            if code is not None:
                log.close();results.append(dict(task=task,exit_code=code));del running[gpu]
                # Let the remaining workers finish even after a failure; never relaunch.
                if code!=0:pending=[]
        if running:time.sleep(.1)
    assert all(r['exit_code']==0 for r in results),results
    return dict(seconds=time.perf_counter()-start,workers=results,slots=slots)
start=time.perf_counter()
pre=wave([('witness','Niushou',0,'preflight')],1,'witness')
profile={}
for slots in [2,4]:profile[slots]=wave([('profile','Niushou',0,f'slots{slots}_case{i}') for i in range(4)],slots,f'profile{slots}')
slots=min(profile,key=lambda n:profile[n]['seconds'])
(P/'RESOURCE_PROFILE.json').write_text(json.dumps(dict(physical=os.environ['CUDA_VISIBLE_DEVICES'],gpu_names=[torch.cuda.get_device_name(i) for i in range(4)],
    preflight=pre,profile=profile,chosen_slots=slots,pbs_job=os.environ.get('PBS_JOBID')),indent=2))
tasks=[(st,fold) for st in ['Niushou','Wangzhai','Mizhuang'] for fold in range(3)]
inner=wave([('inner',st,fold,'') for st,fold in tasks],slots,'inner')
subprocess.run([sys.executable,str(P/'freeze_selection.py')],check=True)
assert not (P/'outer_attempt1').exists()
outer=wave([('outer',st,fold,'') for st,fold in tasks],slots,'outer')
subprocess.run([sys.executable,str(P/'assemble.py')],check=True)
subprocess.run([sys.executable,str(P/'score.py')],check=True)
(P/'RUN_STATUS.json').write_text(json.dumps(dict(state='COMPLETE',inner=inner,outer=outer,seconds=time.perf_counter()-start,
    selection_sha256=hashlib.sha256((P/'SELECTION.json').read_bytes()).hexdigest()),indent=2))
print('NESTED_COMPLETE',time.perf_counter()-start,flush=True)
