import json,os,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent;gpus=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(gpus)==len(set(gpus))==3
for gpu in gpus:
    used,util=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',');assert int(used)<500 and int(util)<5
assert json.loads((P/'PACK_AUDIT.json').read_text())['state']=='PASS_CAUSAL_PACK_RECONSTRUCTION'
out=P/'training_attempt1';out.mkdir(exist_ok=False)
with (out/'profile.log').open('w') as log:subprocess.run([sys.executable,str(P/'train_corrector.py'),'--mode','profile'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpus[0]),stdout=log,stderr=subprocess.STDOUT,check=True)
assert json.loads((out/'PROFILE.json').read_text())['state'].startswith('PASS_CORRECTOR_')
processes=[]
for seed,gpu in zip([42,43,44],gpus):
    log=(out/f'seed{seed}.log').open('w');p=subprocess.Popen([sys.executable,str(P/'train_corrector.py'),'--mode','train','--seed',str(seed)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT);processes.append((p,log,seed))
failed=[]
for p,log,seed in processes:
    code=p.wait();log.close()
    if code:failed.append(dict(seed=seed,exit=code))
assert not failed,failed
seeds=[json.loads((out/f'seed{seed}/COMPLETE.json').read_text()) for seed in [42,43,44]]
(out/'COMPLETE.json').write_text(json.dumps(dict(state='THREE_SEED_COT_DOMAIN_PILOT_COMPLETE',seeds=seeds,upstream_training=False,outer_payloads_read=0),indent=2));print(json.dumps(seeds),flush=True)
