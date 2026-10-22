import argparse,json,os,subprocess,sys
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--gpus',required=True);a=ap.parse_args()
gpus=a.gpus.split(',');assert len(gpus)==3 and len(set(gpus))==3
for gpu in gpus:
    f=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',');assert int(f[0])<500 and int(f[1])<5
out=a.root/'reference_attempt1';out.mkdir(exist_ok=False);tasks=[]
for roles,gpu in zip(['fold_0,fold_3','fold_1,fold_4','fold_2'],gpus):
    log=(out/f'{roles.replace(",","_")}.log').open('w');p=subprocess.Popen([sys.executable,str(a.root/'infer_roles.py'),'--roles',roles],env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT);tasks.append((p,log,roles))
failed=[]
for p,log,roles in tasks:
    code=p.wait();log.close()
    if code:failed.append(dict(roles=roles,exit=code))
assert not failed,failed
records=[json.loads((out/f'fold_{i}/COMPLETE.json').read_text()) for i in range(5)]
assert all(r['state']=='MATCHED_FOLD_ACTUAL_AGRI_R_REFERENCE_COMPLETE' for r in records)
(out/'COMPLETE.json').write_text(json.dumps(dict(state='MATCHED_FIVE_FOLD_ACTUAL_TRAIN_COT_REFERENCE_COMPLETE',roles=records,frames=sum(r['frames'] for r in records),new_fits=0,official_val_new_period_test_outerbuffer_payloads_read=0),indent=2))
