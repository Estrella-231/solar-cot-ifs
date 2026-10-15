"""Recover missing early validation curve; replay original prefix without tuning."""
import sys,json,hashlib,math,time
from pathlib import Path
import numpy as np
import torch
S=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907')
E=S/'experiments/history_cot_forecast_ghi_20261015_v1'
sys.path.insert(0,str(S/'scripts'))
from train_history_cot_ghi_paired_20261015 import Head,seed,metrics
O=E/'diagnosis_20261015/early_replay_attempt1';assert not O.exists();O.mkdir()
torch.set_num_threads(2)
z=np.load(E/'results_attempt2/paired_pack.npz');norms=json.loads((E/'results_attempt2/source_audit.json').read_text())['cot_norm_train_only']
data={}
for split in ('train','val'):
 data[split]={'x':torch.tensor(z[split+'_x'],device='cuda'),'lead':torch.tensor(z[split+'_lead'],device='cuda'),'support':torch.tensor(z[split+'_support'],device='cuda'),'clear':torch.tensor(z[split+'_clear'],device='cuda',dtype=torch.float32),'ghi':torch.tensor(z[split+'_ghi'],device='cuda',dtype=torch.float32),'cot':{arm:torch.tensor((z[split+'_cot_'+arm]-norms[arm][0])/norms[arm][1],device='cuda') for arm in ('transport','hybrid')}}
tr,va=data['train'],data['val'];true=z['val_ghi'];n=len(z['train_rows']);batch=8192
def forward(m,d,ix,arm):return m(d['x'][ix],d['cot']['transport' if arm=='A' else arm][ix],d['support'][ix],d['lead'][ix],arm!='A')
@torch.no_grad()
def evaluate(m,arm):
 pred=[]
 for lo in range(0,len(true),2048):
  ix=torch.arange(lo,min(lo+2048,len(true)),device='cuda');pred.append((forward(m,va,ix,arm).double()*va['clear'][ix].double()).cpu().numpy())
 return np.concatenate(pred)
records=[];summary=json.loads((E/'results_attempt2/summary.json').read_text());checkpoints=(1,2,5,10,20,30,40,50,75,100)
for sv in (42,43,44):
 rng=np.random.default_rng(sv);stream=np.concatenate([rng.permutation(n) for _ in range(math.ceil(1200*batch/n))])[:1200*batch].reshape(1200,batch)
 for arm in ('A','transport','hybrid'):
  seed(sv);m=Head().cuda();initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in m.state_dict().values())).hexdigest()
  old=next(r for r in summary['runs'] if r['arm']==arm and r['seed']==sv);assert initial==old['initial_sha256']
  opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4);curve=[];run=O/f'{arm}_seed{sv}';run.mkdir()
  for step in range(1,101):
   lr=1e-5+.5*(1e-3-1e-5)*(1+math.cos(math.pi*(step-1)/1200))
   for g in opt.param_groups:g['lr']=lr
   ix=torch.tensor(stream[step-1],device='cuda');opt.zero_grad(set_to_none=True)
   loss=((forward(m,tr,ix,arm)*tr['clear'][ix]-tr['ghi'][ix])/1000).square().mean();assert torch.isfinite(loss)
   loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);opt.step()
   if step in checkpoints:
    pred=evaluate(m,arm);met=metrics(pred,true);curve.append(dict(step=step,train_batch_rmse=float(loss.detach().sqrt()*1000),**met))
    np.savez_compressed(run/f'pred_step{step:03d}.npz',pred_ghi=pred,true_ghi=true,row_ids=z['val_rows'])
  oldcurve=json.loads((E/'results_attempt2'/run.name/'curve.json').read_text());diff=abs(curve[-1]['rmse']-oldcurve[0]['rmse']);assert diff<.001,(run.name,diff)
  (run/'curve.json').write_text(json.dumps(curve,indent=2));records.append(dict(seed=sv,arm=arm,step100_replay_rmse_difference=diff,early_curve=curve))
  print('PREFIX_REPLAY',sv,arm,'step100_difference',diff,flush=True)
  del m,opt;torch.cuda.empty_cache()
(O/'summary.json').write_text(json.dumps(dict(state='COMPLETE_EXACT_PREFIX_REPLAY',new_test_read=False,scope='early checkpoint grid diagnostic; same initialization/data/order/lr/update prefix; no method or hyperparameter tuning',records=records),indent=2));print('EARLY_REPLAY_COMPLETE',flush=True)
