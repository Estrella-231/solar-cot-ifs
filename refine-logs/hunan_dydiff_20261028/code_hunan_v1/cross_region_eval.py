"""Frozen Hubei EMA on deterministic Hunan validation windows; not test evidence."""
import argparse,csv,hashlib,json,os,time
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
import torch
import torch.distributed as dist
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
 # Hunan val only. Exclude any full 24-frame window intersecting a Hubei train day.
 bymonth={}
 for i,row in enumerate(data.rows):
  times=[datetime.strptime(Path(r).stem,'%Y%m%d%H%M')+timedelta(hours=8) for r in row['data_relpaths'].split('|')]
  if any(str(t.date()) in hbtrain for t in times):continue
  init=times[7]
  if not (8<=init.hour<11):continue
  bymonth.setdefault(init.strftime('%Y-%m'),[]).append((i,init.isoformat()))
 selected=[];months=sorted(bymonth)
 for m in months:
  candidates=bymonth[m];selected.append(candidates[len(candidates)//2][0])
 if len(selected)>a.windows:selected=[selected[i] for i in np.linspace(0,len(selected)-1,a.windows,dtype=int)]
 assert selected
 if rank==0:
  (a.output/'selection.json').write_text(json.dumps({'scope':'Hunan validation diagnostic, excludes Hubei train dates','indices':selected,'rows':[data.rows[i] for i in selected],'selection':'month-stratified deterministic midday; no performance-based selection','hubei_train_day_exclusion':True,'test_payloads_read':0},indent=2))
 cfg=ModelConfig();model=AGRIDyDiff(cfg).to(device).eval();model.requires_grad_(False)
 checkpoint=torch.load(ckpath,map_location='cpu',weights_only=False);assert checkpoint['epoch']==150
 model.load_state_dict(checkpoint['ema'],strict=True);del checkpoint
 diffusion=DynamicalDiffusion().to(device)
 hnmean=torch.tensor(data.mean,device=device)[None,None];hnstd=torch.tensor(data.std,device=device)[None,None]
 sums=torch.zeros((5,16,13),device=device,dtype=torch.float64);local_records=[]
 started=time.monotonic()
 for idx in selected[rank::world]:
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
  name=f'val_{idx:05d}';np.save(a.output/(name+'_pred.npy'),pred[0].cpu().numpy())
  # Two native physical channels retained for cloud/trajectory visual inspection.
  np.savez_compressed(a.output/(name+'_case.npz'),history=hphys[0,:, [1,9]].cpu().numpy(),truth=yphys[0,:, [1,9]].cpu().numpy(),pred=pred[0,:, [1,9]].cpu().numpy(),valid=fmask[0,:, [1,9]].cpu().numpy())
  local_records.append({'index':idx,'row':data.rows[idx],'seed':2026102800+idx,'path':name+'_pred.npy'})
  print(json.dumps({'event':'hubei_on_hunan_window_complete','rank':rank,'index':idx,'elapsed_seconds':time.monotonic()-started}),flush=True)
 if world>1:
  dist.all_reduce(sums);records=[None]*world if rank==0 else None;dist.gather_object(local_records,records,dst=0)
 else:records=[local_records]
 if rank==0:
  counts=sums[2];rmse=(sums[0]/counts.clamp_min(1)).sqrt().cpu().numpy();mae=(sums[1]/counts.clamp_min(1)).cpu().numpy()
  persistence_skill=(1-sums[4].sum(0)/sums[3].sum(0).clamp_min(1e-12)).cpu().tolist()
  result={'status':'PASS','scope':'Hubei frozen model on selected Hunan validation; exploratory cross-region diagnostic','windows':len(selected),'channels':list(CHANNELS),'lead_minutes':list(range(15,241,15)),
   'rmse_physical_lead_channel':rmse.tolist(),'mae_physical_lead_channel':mae.tolist(),'valid_pixel_counts':counts.cpu().tolist(),'matched_mse_skill_vs_persistence_channel':persistence_skill,
   'hubei_checkpoint_sha256':sha(ckpath),'hubei_normalization_sha256':sha(normpath),'geometry':'Hunan grid; corrected NOAA UTC; known target geometry only','quality':'Hunan original QA AND Hubei physical/daylight support; same support for paired metrics',
   'sampling_steps':25,'members':1,'weights_frozen':True,'retrained_on_hunan':False,'test_payloads_read':0,'records':sum(records,[]),'elapsed_seconds':time.monotonic()-started}
  (a.output/'metrics.json').write_text(json.dumps(result,indent=2));print('HUBEI_ON_HUNAN_DIAGNOSTIC_COMPLETE',flush=True)
 if world>1:dist.destroy_process_group()
if __name__=='__main__':main()
