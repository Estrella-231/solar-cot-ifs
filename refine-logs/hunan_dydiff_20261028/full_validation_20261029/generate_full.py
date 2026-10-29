"""Frozen Hubei EMA on deterministic Hunan validation windows; not test evidence."""
import argparse,csv,hashlib,json,os,time
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
import torch
import torch.distributed as dist
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code_hunan_v1"))
from data import BASE,PREPARED,CHANNELS
from dataset import CachedDataset
from model import AGRIDyDiff,ModelConfig
from diffusion import DynamicalDiffusion

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--windows',type=int,default=16);a=p.parse_args()
 rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
 torch.cuda.set_device(local);device=torch.device('cuda',local)
 if world>1:dist.init_process_group('nccl')
 a.output.mkdir(parents=True,exist_ok=True)
 frozen=BASE/'hubei_frozen';ckpath=frozen/'epoch150.pt';normpath=frozen/'norm.json'
 assert sha(normpath)=='817e071799a3ede5b35f66ba578a14f15ec317e59691273497d296458760d2fb'
 # Rank zero checks the immutable transferred checkpoint once, not once per GPU.
 if rank==0:assert sha(ckpath)=='c4362d3d69181659ebd977c0b66319108a67043e15911ddc909d9c06d4434293'
 if world>1:dist.barrier()
 hb=json.loads(normpath.read_text());assert tuple(hb['channels'])==CHANNELS
 mean=torch.tensor(hb['mean'],device=device)[None,None,:,None,None];std=torch.tensor(hb['std'],device=device)[None,None,:,None,None]
 data=CachedDataset('val')
 with (frozen/'hubei_day_assignment.csv').open() as f:hbtrain={r['date_BJT'] for r in csv.DictReader(f) if r['split']=='train'}
 # All frozen validation windows, without hour/month/performance filtering.
 selected=list(range(len(data)));assert len(selected)==3140
 overlap={i:any(str((datetime.strptime(Path(p).stem,'%Y%m%d%H%M')+timedelta(hours=8)).date()) in hbtrain for p in row['data_relpaths'].split('|')) for i,row in enumerate(data.rows)}
 if rank==0:
  record={'scope':'complete frozen Hunan validation, cross-region frozen Hubei DyDiff; not independent test','indices':selected,'rows':data.rows,'selection':'all 3140 val rows, no time of day or performance filter','hubei_training_day_overlap_indices':[i for i in selected if overlap[i]],'reporting':'full frozen val descriptive, plus prespecified no-Hubei-train-day-overlap subset','test_payloads_read':0}
  target=a.output/'selection.json'
  if target.exists():assert json.loads(target.read_text())==record
  else:target.write_text(json.dumps(record,indent=2))
 cfg=ModelConfig();model=AGRIDyDiff(cfg).to(device).eval();model.requires_grad_(False)
 checkpoint=torch.load(ckpath,map_location='cpu',weights_only=False);assert checkpoint['epoch']==150
 model.load_state_dict(checkpoint['ema'],strict=True);del checkpoint
 diffusion=DynamicalDiffusion().to(device)
 hnmean=torch.tensor(data.mean,device=device)[None,None];hnstd=torch.tensor(data.std,device=device)[None,None]
 sums=torch.zeros((5,16,13),device=device,dtype=torch.float64);local_records=[]
 started=time.monotonic()
 for idx in selected[rank::world]:
  if (a.output/f'val_{idx:05d}_complete.json').exists():continue
  sample=data[idx];batch={k:torch.from_numpy(v)[None].to(device) for k,v in sample.items()}
  hphys=batch['history']*hnstd+hnmean;yphys=batch['future']*hnstd+hnmean
  hmask=batch['history_valid'].bool();fmask=batch['future_valid'].bool()
  # The Hubei model expects its original conservative observational support.
  def support(x,valid,geom):
   valid=valid.clone() & torch.isfinite(x) & (x>0)
   valid[:,:,:6]&=(x[:,:,:6]<=2)&(geom[:,:,2:3]>.5)
   valid[:,:,6:]&=(x[:,:,6:]>=50)&(x[:,:,6:]<=400)
   return valid
  hmask=support(hphys,hmask,batch['geometry'][:,:8]);fmask=support(yphys,fmask,batch['geometry'][:,8:])
  history=torch.where(hmask,(hphys-mean)/std,0)
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   pred=diffusion.sample(model,history,batch['geometry'],hmask,steps=25,seed=[2026102800+idx]).float()*std+mean
  assert torch.isfinite(pred).all()
  e=torch.where(fmask,pred-yphys,0);sums[0]+=e.square().sum((0,3,4),dtype=torch.float64);sums[1]+=e.abs().sum((0,3,4),dtype=torch.float64);sums[2]+=fmask.sum((0,3,4))
  common=fmask & hmask[:,-1:];pe=torch.where(common,hphys[:,-1:]-yphys,0);me=torch.where(common,pred-yphys,0)
  sums[3]+=pe.square().sum((0,3,4),dtype=torch.float64);sums[4]+=me.square().sum((0,3,4),dtype=torch.float64)
  name=f'val_{idx:05d}'
  # Preserve only R-required station patches, avoiding ~171GB full-image output.
  coords=((163,148),(73,150))
  patch=np.stack([pred[0,:,:,rr-8:rr+8,cc-8:cc+8].cpu().numpy() for rr,cc in coords],axis=1)
  tmp=a.output/(name+'_pred.tmp')
  with tmp.open('wb') as f:np.save(f,patch)
  tmp.replace(a.output/(name+'_pred.npy'))
  receipt={'index':idx,'seed':2026102800+idx,'shape':list(patch.shape),'hubei_train_day_overlap':overlap[idx],'test_payloads_read':0}
  (a.output/(name+'_complete.json')).write_text(json.dumps(receipt))
  local_records.append(receipt)
  print(json.dumps({'event':'hubei_on_hunan_window_complete','rank':rank,'index':idx,'elapsed_seconds':time.monotonic()-started}),flush=True)
 # No NCCL object gathering: rank-local receipts are the resumable source.
 if world>1:dist.barrier()
 if rank==0:
  records=[json.loads((a.output/f'val_{i:05d}_complete.json').read_text()) for i in selected]
  assert len(records)==3140
  result={'status':'PASS','scope':'complete frozen Hunan validation, no forecast-adapted training; independent test closed','windows':len(selected),'sampling_steps':25,'members':1,'hubei_checkpoint_sha256':sha(ckpath),'hubei_normalization_sha256':sha(normpath),'overlap_windows':sum(overlap.values()),'no_overlap_windows':sum(not x for x in overlap.values()),'patch_shape':[16,2,13,16,16],'seed_rule':'2026102800+frozen_val_index','test_payloads_read':0,'elapsed_seconds':time.monotonic()-started}
  (a.output/'completion.json').write_text(json.dumps(result,indent=2));print('FULL_VAL_FORECAST_COMPLETE',flush=True)
 if world>1:dist.destroy_process_group()
if __name__=='__main__':main()
