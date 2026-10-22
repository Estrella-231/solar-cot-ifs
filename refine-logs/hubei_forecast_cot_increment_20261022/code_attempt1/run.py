import argparse,json,subprocess,os,time,sys
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepared',required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--gpus',required=True);a=ap.parse_args()
    gpus=a.gpus.split(',');assert len(gpus)==4 and len(set(gpus))==4
    a.out.mkdir(parents=True,exist_ok=False);profiles=a.out/'profile';profiles.mkdir();trainer=Path(__file__).with_name('train.py')
    def launch(st,seed,gpu,out,profile=False):
        env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        cmd=[sys.executable,str(trainer),'--prepared',a.prepared,'--station',st,'--seed',str(seed),'--profile' if profile else '--out',str(out)]
        log=(out.parent/(out.name+'.log')).open('w');p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);return p,log
    rates=[]
    for n in [1,2,4]:
        active=[];t=time.time()
        for i in range(n):
            out=profiles/f'workers{n}_slot{i}.json';p,log=launch(['Niushou','Wangzhai','Mizhuang'][i%3],42+i,gpus[i],out,True);active.append((p,log,out))
        for p,log,out in active:
            rc=p.wait();log.close();assert rc==0,(rc,out)
        rate=sum(json.loads(p.read_text())['examples_per_second'] for _,_,p in active)
        rates.append({'workers':n,'aggregate_examples_per_second':rate,'wall_seconds_with_startup':time.time()-t})
        print(json.dumps({'event':'PROFILE',**rates[-1]}),flush=True)
    best=max(x['aggregate_examples_per_second'] for x in rates);workers=max(x['workers'] for x in rates if x['aggregate_examples_per_second']>=.95*best)
    (a.out/'profile.json').write_text(json.dumps({'results':rates,'selected_workers':workers},indent=2))
    tasks=[(st,seed) for seed in [42,43,44] for st in ['Niushou','Wangzhai','Mizhuang']];running={};done=[];pairs=a.out/'pairs';pairs.mkdir()
    while tasks or running:
        for gpu in gpus[:workers]:
            if gpu in running or not tasks:continue
            st,seed=tasks.pop(0);out=pairs/f'{st}_seed{seed}';p,log=launch(st,seed,gpu,out);running[gpu]=(p,log,out,st,seed)
            print(json.dumps({'event':'START','station':st,'seed':seed,'gpu':gpu}),flush=True)
        for gpu,item in list(running.items()):
            p,log,out,st,seed=item
            if p.poll() is None:continue
            log.close();del running[gpu]
            if p.returncode!=0:
                # Let already-running paired trials preserve their own progress.
                for pp,ll,*_ in running.values():pp.wait();ll.close()
                raise RuntimeError(f'Worker failed: {out}, code {p.returncode}; see sibling log')
            rec=json.loads((out/'COMPLETE.json').read_text());assert rec['state']=='FOUR_ARM_COMPLETE';done.append({'station':st,'seed':seed,'path':str(out)})
            print(json.dumps({'event':'COMPLETE','station':st,'seed':seed,'gpu':gpu}),flush=True)
        (a.out/'progress.json').write_text(json.dumps({'complete':done,'running':[str(v[2]) for v in running.values()],'remaining':tasks},indent=2))
        if tasks or running:time.sleep(2)
    (a.out/'ALL_COMPLETE.json').write_text(json.dumps({'state':'ALL_36_HEADS_COMPLETE','trials':done,'selected_workers':workers,'test_payloads_read':0,'buffer_payloads_read':0},indent=2));print('ALL_36_HEADS_COMPLETE',flush=True)
if __name__=='__main__':main()
