"""Read-only internal attribution in two independent inference waves."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--gpus',required=True);a=ap.parse_args()
gpus=a.gpus.split(',');assert len(gpus)==4
out=a.root/'internal_replay_attempt1';out.mkdir(exist_ok=False)
runs=json.loads((a.root/'fits_attempt1/ARM_SELECTION.json').read_text())['runs'];assert len(runs)==6
for start in [0,4]:
    tasks=[]
    for i,spec in enumerate(runs[start:start+4]):
        name=f"seed{spec['seed']}_{spec['arm']}";log=(out/f'{name}.log').open('w')
        cmd=[sys.executable,str(a.root/'internal_replay.py'),'--run',spec['run'],'--out',str(out/name)]
        tasks.append((subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpus[i]),stdout=log,stderr=subprocess.STDOUT),log,name))
    failures=[]
    for proc,log,name in tasks:
        rc=proc.wait();log.close()
        if rc:failures.append(dict(name=name,exit=rc))
    assert not failures,failures
(out/'COMPLETE.json').write_text(json.dumps(dict(state='SIX_TRAIN_INTERNAL_FROZEN_REPLAYS_COMPLETE',new_training_runs=0,outer_val_read=False,test_buffer_read=False),indent=2))
