import os,sys,csv,json,hashlib,time,subprocess,argparse,copy
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
import torch
from torch import nn
ROOT=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907')
OUT=ROOT/'refine-logs/hubei_same_time_geometry_20261102';BASE=ROOT/'experiments/hubei_solar_geometry_repair_20261021'
ST={'Niushou':(8,6),'Wangzhai':(9,8),'Mizhuang':(7,10)}
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def dump(p,d):p.write_text(json.dumps(d,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def model():
 sys.path.insert(0,str(ROOT/'experiments/cloud_probability_input_20261020'))
 from train_four_arm import FourArmIntervalHead
 return FourArmIntervalHead()
def build():
 torch.set_num_threads(2);fr=OUT/'frames';assert (fr/'COMPLETE.json').exists()
 raw=np.load(fr/'raw.npy');ks=rows(fr/'keys.csv');lut={r['time_bjt']:i for i,r in enumerate(ks)};assert len(lut)==len(ks)
 sys.path.insert(0,str(BASE/'code_pipeline_attempt12'));from train_cot_repaired import COTUNet
 norm=json.loads((BASE/'prepared_attempt12/R_packs/full/norm.json').read_text());print('NORM_KEYS',list(norm),flush=True)
 mean=np.array(norm['mean'],np.float32);std=np.array(norm['std'],np.float32)
 ck=BASE/'R_runs_attempt12/full/best.pt';net=COTUNet().cuda().eval().requires_grad_(False);net.load_state_dict(torch.load(ck,map_location='cpu')['model'])
 cs=[]
 with torch.no_grad():
  for k in range(0,len(raw),256):
   x=(raw[k:k+256]-mean[None,:,None,None])/std[None,:,None,None]
   c=net(torch.from_numpy(x).cuda()).cpu().numpy()[:,0]
   valid=(raw[k:k+256,15]>.5)&(raw[k:k+256,13]>np.cos(np.deg2rad(80)))
   cs.append(np.where(valid,c,0).astype(np.float32))
 c=np.concatenate(cs);assert np.isfinite(c).all() and (c>=0).all()
 np.save(fr/'cot_log1p.npy',c)
 old=ROOT/'experiments/hubei_cloud_probability_20261020/ghi_fourarm_pack_attempt5';rr=rows(old/'rows.csv')
 xx=[];yy=[];clear=[];kept=[];excluded={};used=[]
 for i,r in enumerate(rr):
  t=datetime.strptime(r['time_bjt'],'%Y%m%d%H%M');kk=[(t-timedelta(minutes=m)).strftime('%Y%m%d%H%M') for m in (30,15,0)]
  if any(q not in lut or ks[lut[q]]['split']!=r['split'] for q in kk):excluded['context_unavailable_or_split_boundary']=excluded.get('context_unavailable_or_split_boundary',0)+1;continue
  ids=[lut[q] for q in kk];y,x=ST[r['station']];a=raw[ids,:,y-1:y+2,x-1:x+2];cot=c[ids,y-1:y+2,x-1:x+2]
  if np.any(np.all(a[:,:13]==0,axis=(1,2,3))):excluded['all_spectral_zero_frame']=excluded.get('all_spectral_zero_frame',0)+1;continue
  feat=np.concatenate((a[:,:15],cot[:,None],np.zeros((3,1,3,3),np.float32)),axis=1)
  assert feat.shape==(3,17,3,3) and np.isfinite(feat).all()
  xx.append(feat);yy.append(float(r['ghi_wm2']));clear.append(float(r['clear_sky_wm2']));kept.append(r);used.append(ids)
 pack=OUT/'pack';pack.mkdir(exist_ok=False);np.save(pack/'x.npy',np.stack(xx));np.save(pack/'ghi.npy',np.array(yy,np.float32));np.save(pack/'clear.npy',np.array(clear,np.float32));np.save(pack/'frame_indices.npy',used)
 with (pack/'rows.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(kept[0]));w.writeheader();w.writerows(kept)
 counts={f'{sp}/{st}':sum(r['split']==sp and r['station']==st for r in kept) for sp in ('train','val') for st in ST}
 # Both arms use one shared row axis; actual geometry and radiance fixed before fit.
 dump(pack/'COMPLETE.json',{'state':'COMPLETE','rows':len(kept),'candidate_pool':len(rr),'counts':counts,'excluded':excluded,'R_checkpoint_sha256':sha(ck),'R_norm_sha256':sha(BASE/'prepared_attempt12/R_packs/full/norm.json'),'source_label_rows_sha256':sha(old/'rows.csv'),'R_frozen':True,'training_domain':'three observed frames only','offset_minutes':[-30,-15,0],'test_payloads_read':0});print('PACK_COMPLETE',counts,excluded,flush=True)
def train(st,seed):
 torch.set_num_threads(1);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.backends.cudnn.benchmark=False
 pack=OUT/'pack';rs=rows(pack/'rows.csv');ids=np.array([i for i,r in enumerate(rs) if r['station']==st]);split=np.array([rs[i]['split'] for i in ids]);tr=np.flatnonzero(split=='train');va=np.flatnonzero(split=='val')
 x=np.array(np.load(pack/'x.npy',mmap_mode='r')[ids]);y=np.load(pack/'ghi.npy')[ids];cl=np.load(pack/'clear.npy')[ids]
 assert len(tr)>0 and len(va)>0 and len({rs[i]['time_bjt'] for i in ids})==len(ids)
 am=x[tr,:,:13].astype(float).mean((0,1,3,4));asd=np.maximum(x[tr,:,:13].astype(float).std((0,1,3,4)),1e-6);cm=float(x[tr,:,15].astype(float).mean());cs=max(float(x[tr,:,15].astype(float).std()),1e-6)
 x[:,:,:13]=(x[:,:,:13]-am[None,None,:,None,None])/asd[None,None,:,None,None];x[:,:,15]=(x[:,:,15]-cm)/cs
 xt=torch.tensor(x,device='cuda');yt=torch.tensor(y,device='cuda');ct=torch.tensor(cl,device='cuda');base=model().cuda();init=copy.deepcopy(base.state_dict());ih=hashlib.sha256(b''.join(t.cpu().numpy().tobytes() for t in init.values())).hexdigest();del base
 models={a:model().cuda() for a in ('A','AC')}
 for m in models.values():m.load_state_dict(init)
 opts={a:torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4) for a,m in models.items()};sched={a:torch.optim.lr_scheduler.CosineAnnealingLR(o,30,eta_min=1e-5) for a,o in opts.items()}
 rng=np.random.default_rng(seed);orders=[rng.permutation(tr) for _ in range(30)];oh=hashlib.sha256(np.concatenate(orders).tobytes()).hexdigest();best={a:float('inf') for a in models};updates=0
 dest=OUT/'training'/f'{st}_seed{seed}';dest.mkdir(parents=True,exist_ok=False);norm={'agri_mean':am.tolist(),'agri_std':asd.tolist(),'cot_log1p_mean':cm,'cot_log1p_std':cs,'fit_split':'train'};dump(dest/'norm.json',norm)
 def inputs(a,ix):
  v=xt[ix].clone()
  if a=='A':v[:,:,15]=0
  return v
 start=time.time()
 for ep,order in enumerate(orders,1):
  loss_tot={a:0. for a in models}
  for k in range(0,len(order),512):
   ix=torch.tensor(order[k:k+512],device='cuda')
   for a,m in models.items():
    m.train();opts[a].zero_grad(set_to_none=True);loss=(((m(inputs(a,ix))*ct[ix]-yt[ix])/1000)**2).mean();assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);opts[a].step();loss_tot[a]+=float(loss)*len(ix)
   updates+=1
  scores={}
  for a,m in models.items():
   m.eval()
   with torch.no_grad():pred=torch.cat([m(inputs(a,ix))*ct[ix] for ix in torch.tensor(va,device='cuda').split(1024)]).cpu().numpy()
   err=pred.astype(float)-y[va].astype(float);rmse=float(np.sqrt(np.mean(err**2)));scores[a]=rmse
   if rmse<best[a]:
    best[a]=rmse;torch.save({'model':m.state_dict(),'epoch':ep,'updates':updates,'normalization':norm,'arm':a,'seed':seed,'station':st},dest/f'{a}_best.pt');np.savez_compressed(dest/f'{a}_validation.npz',prediction=pred,truth=y[va],clear=cl[va],global_indices=ids[va])
   sched[a].step()
  rec={'epoch':ep,'updates':updates,'rmse':scores,'train_loss':{a:v/len(tr) for a,v in loss_tot.items()},'seconds':time.time()-start};print(st,seed,json.dumps(rec),flush=True)
  with (dest/'epochs.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
 for a,m in models.items():torch.save({'model':m.state_dict(),'epoch':30,'updates':updates,'normalization':norm},dest/f'{a}_last.pt')
 dump(dest/'COMPLETE.json',{'state':'COMPLETE','station':st,'seed':seed,'train':len(tr),'val':len(va),'epochs':30,'updates_per_arm':updates,'global_batch':512,'paired_initialization_sha256':ih,'shared_orders_sha256':oh,'parameters':sum(p.numel() for p in models['A'].parameters()),'seconds':time.time()-start,'normalization':'per-station train only','training_input':'real same-time three frames','R_frozen':True,'test_payloads_read':0})
def score():
 rs=rows(OUT/'pack/rows.csv');cells=[];allkeys=None
 for st in ST:
  for seed in [42,43,44]:
   dest=OUT/'training'/f'{st}_seed{seed}';assert (dest/'COMPLETE.json').exists()
   a=np.load(dest/'A_validation.npz');b=np.load(dest/'AC_validation.npz');assert np.array_equal(a['global_indices'],b['global_indices']) and np.array_equal(a['truth'],b['truth'])
   ii=a['global_indices'];truth=a['truth'].astype(float);pred=np.stack([a['prediction'],b['prediction']]).astype(float);dates=np.array([rs[i]['time_bjt'][:8] for i in ii]);cells.append((st,seed,ii,dates,truth,pred))
 dates=sorted(set(np.concatenate([c[3] for c in cells])));rng=np.random.default_rng(20261102);stats=[]
 for st,seed,ii,dt,y,p in cells:
  stats.append({'station':st,'seed':seed,'n':len(y),'A_rmse':float(np.sqrt(np.mean((p[0]-y)**2))),'AC_rmse':float(np.sqrt(np.mean((p[1]-y)**2))),'A_mae':float(np.mean(abs(p[0]-y))),'AC_mae':float(np.mean(abs(p[1]-y))),'A_bias':float(np.mean(p[0]-y)),'AC_bias':float(np.mean(p[1]-y))})
 vals=np.array([[v['A_rmse'],v['AC_rmse']] for v in stats]);point=vals.mean(0);draws=[]
 for _ in range(5000):
  chosen=rng.choice(dates,len(dates),replace=True);weights={d:int(np.sum(chosen==d)) for d in dates};s=[]
  for st,seed,ii,dt,y,p in cells:
   w=np.array([weights[d] for d in dt]);s.append(np.sqrt(((p-y)**2*w).sum(1)/w.sum()))
  s=np.mean(s,0);draws.append(s[1]-s[0])
 ci=np.quantile(draws,[.025,.975]).tolist();by={st:{'A_rmse':float(vals[[i for i,s in enumerate(stats) if s['station']==st],0].mean()),'AC_rmse':float(vals[[i for i,s in enumerate(stats) if s['station']==st],1].mean())} for st in ST}
 report={'state':'COMPLETE','scope':'Hubei corrected solar geometry; observed T-30/T-15/T frames; paired validation','station_seed_equal_RMSE':{'A':float(point[0]),'AC':float(point[1])},'relative_RMSE_reduction_pct':float(100*(point[0]-point[1])/point[0]),'delta_AC_minus_A':float(point[1]-point[0]),'date_block_95CI':ci,'unique_validation_station_times':sum(v['n'] for v in stats)//3,'target_dates':dates,'by_station':by,'cells':stats,'test_payloads_read':0}
 dump(OUT/'RESULTS.json',report)
 txt='# 湖北修正太阳几何：真实三帧同期GHI验证\n\n'+json.dumps({k:v for k,v in report.items() if k not in ('cells','target_dates')},ensure_ascii=False,indent=2)+'\n\n日期块配对区间为5000次目标BJT日期重采样。逐站、逐种子先算RMSE后等权，不混池。非独立test。\n'
 (OUT/'RESULTS.md').write_text(txt);print(txt,flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['build','train','score','run']);p.add_argument('--station');p.add_argument('--seed',type=int);a=p.parse_args()
 if a.mode=='build':build()
 elif a.mode=='train':train(a.station,a.seed)
 elif a.mode=='score':score()
 else:
  build();devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',');tasks=[(st,seed) for st in ST for seed in [42,43,44]];running=[]
  while tasks or running:
   busy={g for pr,g,f in running}
   for g in devices:
    if g in busy or not tasks:continue
    st,seed=tasks.pop(0);f=(OUT/f'{st}_{seed}.log').open('w');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=g
    running.append((subprocess.Popen([sys.executable,__file__,'train','--station',st,'--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT,env=env),g,f))
   left=[]
   for pr,g,f in running:
    ret=pr.poll()
    if ret is None:left.append((pr,g,f))
    else:
     f.close()
     if ret:raise RuntimeError(f'train worker exit {ret}')
   running=left;time.sleep(2)
  score()
if __name__=='__main__':main()
