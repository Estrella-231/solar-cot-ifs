import argparse,json,os,subprocess,sys
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--gpus',required=True);ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
gpus=a.gpus.split(',');assert len(gpus)==4
out=a.root/'cache_attempt1';out.mkdir(exist_ok=False);jobs=[]
for i,gpu in enumerate(gpus):
    log=(out/f'shard{i}.log').open('w');env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu)
    cmd=[sys.executable,a.root/'cache_history_r.py','--shard',i,'--world',4,
         '--data','/dev/shm/hubei_region_trajectory_train_20261023_attempt1','--out',out/f'shard{i}']
    jobs.append((subprocess.Popen(list(map(str,cmd)),env=env,stdout=log,stderr=subprocess.STDOUT),log,i))
fail=[]
for p,log,i in jobs:
    rc=p.wait();log.close()
    if rc:fail.append(dict(shard=i,exit=rc))
assert not fail,fail
subprocess.run([sys.executable,str(a.root/'build_pair_pack.py'),'--cache',str(out),
    '--out','/dev/shm/hubei_trajectory_pair_pack_20261023_attempt1'],check=True)
report=json.loads(Path('/dev/shm/hubei_trajectory_pair_pack_20261023_attempt1/REPORT.json').read_text())
(a.root/'PAIR_PACK_COMPLETE.json').write_text(json.dumps(report,indent=2));print('TRAJECTORY_PAIR_PACK_READY_NOT_TRAINED',flush=True)
