"""Profile 2/4 independent GPU workers and run frozen new-source inference."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--gpus',required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--attempt',type=int,default=1);a=p.parse_args()
    gpu=a.gpus.split(',');assert len(gpu)==4 and len(set(gpu))==4
    code=Path(__file__).resolve().parent;out=a.root/f'inference_attempt{a.attempt}';out.mkdir(exist_ok=False)
    shared=['--data',a.data,'--selection',a.root/'INPUT_SELECTION.json','--grid',a.root/'grid_static256.npz','--freeze',a.root/'MODEL_FREEZE.json']
    def start(mode,slot,n,dest):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu[slot]);log=dest.with_suffix('.log').open('w')
        cmd=[os.environ.get('S_PYTHON',sys.executable),code/'infer_new_period.py','--mode',mode,'--shard',slot,'--world',n,'--out',dest]+shared
        return subprocess.Popen(list(map(str,cmd)),env=env,stdout=log,stderr=subprocess.STDOUT),log,dest
    rates=[]
    for n in [2,4]:
        tasks=[start('profile',i,n,out/f'profile{n}_gpu{i}') for i in range(n)]
        errors=[]
        for proc,log,path in tasks:
            rc=proc.wait();log.close()
            if rc:errors.append((str(path),rc))
        assert not errors,errors
        rate=sum(json.loads((path/'PROFILE.json').read_text())['sequences_per_second'] for _,_,path in tasks)
        rates.append(dict(workers=n,aggregate_sequences_per_second=rate));print(json.dumps(rates[-1]),flush=True)
    peak=max(r['aggregate_sequences_per_second'] for r in rates);n=max(r['workers'] for r in rates if r['aggregate_sequences_per_second']>=.95*peak)
    (out/'PROFILE.json').write_text(json.dumps(dict(results=rates,selected_workers=n),indent=2)+'\n')
    tasks=[start('S',i,n,out/f'shard{i}') for i in range(n)];errors=[]
    for proc,log,path in tasks:
        rc=proc.wait();log.close()
        if rc:errors.append((str(path),rc))
    assert not errors,errors
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu[0]);dest=a.root/f'predictions_attempt{a.attempt}'
    cmd=[sys.executable,code/'infer_new_period.py','--mode','R_GHI','--out',dest,'--s-bank',out,
         '--keys',a.root/'keys.csv','--labels',a.root/'labels_with_clear.csv']+shared
    subprocess.run(list(map(str,cmd)),env=env,check=True)
    print('FROZEN_NEW_PERIOD_INFERENCE_COMPLETE',flush=True)
if __name__=='__main__':main()
