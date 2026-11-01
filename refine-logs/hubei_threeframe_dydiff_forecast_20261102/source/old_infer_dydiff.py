"""Epoch150 EMA forecasts. Only historical AGRI and known solar geometry inputs."""
from collections import OrderedDict
import json,os,sys,time
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset,DataLoader
from paths import *
sys.path.insert(0,str(BASE/'train_code_v1'))
from model import AGRIDyDiff,ModelConfig
from diffusion import DynamicalDiffusion
from samplers import StridedValidationSampler

class Histories(Dataset):
 def __init__(self):
  self.rows=json.loads((OUT/'selection.json').read_text())['sequences']
  n=json.loads((PREP/'norm.json').read_text());assert tuple(n['channels'])==CHANNELS
  self.mean=np.asarray(n['mean'],np.float32)[:,None,None];self.std=np.asarray(n['std'],np.float32)[:,None,None]
  self.audit=json.loads((OUT/'history_audit.json').read_text());self.cache=OrderedDict()
 def __len__(self):return len(self.rows)
 def history(self,rel):
  if rel in self.cache:self.cache.move_to_end(rel);return self.cache[rel]
  p=Path(os.environ['GHI_RAW_CACHE'])/Path(rel).name;s=p.stat();r=self.audit[rel]
  assert (s.st_size,s.st_mtime_ns)==(r['source_bytes'],r['source_mtime_ns'])
  raw=np.load(p,mmap_mode='r');x=np.asarray(raw[list(CHANNELS)],np.float32)
  with np.load(PREP/'aux'/f'{p.stem}.npz') as z:valid=np.stack([(z['valid_bits']&(1<<k))!=0 for k in range(13)])
  normalized=np.zeros_like(x);np.subtract(x,self.mean,out=normalized,where=valid);np.divide(normalized,self.std,out=normalized,where=valid)
  assert np.isfinite(normalized).all();self.cache[rel]=(normalized,valid)
  while len(self.cache)>64:self.cache.popitem(last=False)
  return normalized,valid
 def __getitem__(self,i):
  r=self.rows[i];paths=r['row']['data_relpaths'].split('|');hist=[self.history(p) for p in paths[:8]]
  geometry=[]
  for rel in paths:
   with np.load(PREP/'aux'/f'{Path(rel).stem}.npz') as z:geometry.append(np.stack([z[k] for k in ['cosSOZ','cosRAA','day_mask']]))
  return dict(history=np.stack([x[0] for x in hist]),history_valid=np.stack([x[1] for x in hist]),
    geometry=np.stack(geometry).astype(np.float32),index=i,seed=22261028+r['original_val_index'])

def main():
 rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
 torch.cuda.set_device(local);dist.init_process_group('nccl');device=torch.device('cuda',local)
 assert world==6
 ready=json.loads((OUT/'PREPARE_COMPLETE.json').read_text());assert ready['status']=='PASS'
 meta=json.loads((BASE/'run_seed42_150ep_v1/run_metadata.json').read_text())
 ds=Histories();sampler=StridedValidationSampler(len(ds),rank,world)
 loader=DataLoader(ds,batch_size=4,sampler=sampler,num_workers=2,pin_memory=True,persistent_workers=True,prefetch_factor=1,multiprocessing_context='spawn')
 model=AGRIDyDiff(ModelConfig(**meta['architecture']['config'])).to(device).eval().requires_grad_(False)
 ckpt=torch.load(OUT/'checkpoint_epoch150.pt',map_location='cpu');assert ckpt['epoch']==150
 assert ckpt['signature']==meta['signature'];model.load_state_dict(ckpt['ema'],strict=True);del ckpt
 diffusion=DynamicalDiffusion().to(device);patches=[];indices=[];start=time.time()
 for bi,b in enumerate(loader,1):
  ix=b.pop('index').tolist();seeds=b.pop('seed').tolist();b={k:v.to(device,non_blocking=True) for k,v in b.items()}
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   p=diffusion.sample(model,b['history'],b['geometry'],b['history_valid'],steps=25,seed=seeds)
  assert torch.isfinite(p).all()
  physical=p[:,:,:,103:119,118:134].float().cpu().numpy()*ds.std[None,None]+ds.mean[None,None]
  patches.append(physical);indices.extend(ix)
  if bi==1 or bi%5==0:
   dump(OUT/f'progress_rank{rank:02d}.json',dict(epoch=150,batch=bi,local_sequences=len(indices),local_total=len(sampler),seconds=time.time()-start))
   if rank==0:print('DYDIFF_GHI_PROGRESS',bi,len(indices),len(sampler),round(time.time()-start,1),flush=True)
 np.savez_compressed(OUT/f'dydiff_rank{rank:02d}.npz',indices=np.asarray(indices),patches=np.concatenate(patches))
 dist.barrier()
 if rank==0:
  full=np.empty((len(ds),16,13,16,16),np.float32);allindices=[]
  for q in range(world):
   with np.load(OUT/f'dydiff_rank{q:02d}.npz') as z:full[z['indices']]=z['patches'];allindices.extend(z['indices'].tolist())
  assert sorted(allindices)==list(range(len(ds)))
  np.save(OUT/'dydiff_agri_patch.npy',full)
  assert sha(OUT/'checkpoint_epoch150.pt')==ready['dydiff_checkpoint_sha256']
  dump(OUT/'INFERENCE_COMPLETE.json',dict(status='PASS',epoch=150,sequences=len(ds),checkpoint_sha256=ready['dydiff_checkpoint_sha256'],
    sampling_steps=25,sampling_members=1,seed_policy='22261028+original_current_validation_sequence_index',
    future_observed_AGRI_used=False,output_patch_role='Internal original frozen COT input context only; GHI readouts and scoring each station own3x3',
    seconds=time.time()-start,prediction_sha256=sha(OUT/'dydiff_agri_patch.npy')))
  print('DYDIFF_INFERENCE_COMPLETE',len(ds),flush=True)
 dist.destroy_process_group()
if __name__=='__main__':main()
