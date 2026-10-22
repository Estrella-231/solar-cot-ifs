"""Independent frozen-inference shards in one PBS allocation; no DDP training."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--gpus',required=True);ap.add_argument('--root',type=Path,required=True)
ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
gpus=a.gpus.split(',');assert len(gpus)==4 and len(set(gpus))==4
a.out.mkdir(exist_ok=False);started=time.monotonic()
for mode in ['history','forecast']:
    jobs=[]
    for shard,gpu in enumerate(gpus):
        output=a.out/f'{mode}{shard}';log=(a.out/f'{mode}{shard}.log').open('w')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu)
        python=os.environ['S_PYTHON'] if mode=='forecast' else sys.executable
        cmd=[python,a.root/'g4_inference.py','--mode',mode,'--shard',str(shard),'--world','4',
             '--data',a.data,'--out',output]
        jobs.append((subprocess.Popen(list(map(str,cmd)),stdout=log,stderr=subprocess.STDOUT,env=env),log,str(output)))
    failures=[]
    for p,log,output in jobs:
        rc=p.wait();log.close()
        if rc:failures.append(dict(output=output,exit_code=rc))
    assert not failures,failures
subprocess.run([sys.executable,str(a.root/'score_g4.py'),'--run',str(a.out),'--out',str(a.out/'scores')],check=True)
(a.out/'COMPLETE.json').write_text(json.dumps(dict(state='G4_FROZEN_ROUTES_INFERENCE_AND_SCORING_COMPLETE',
    seconds=time.monotonic()-started,gpus=gpus,training=False,test_payloads_read=0,GHI_labels_read=0),indent=2))
print('G4_FROZEN_ROUTES_COMPLETE',flush=True)
