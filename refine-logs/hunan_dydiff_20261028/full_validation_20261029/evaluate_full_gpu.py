"""Frozen corrected Hunan R and matched GHI heads on saved DyDiff validation.

CPU only. All input times are canonical UTC; labels end the target15min interval.
No network is trained, no test numerical payload is opened.
"""
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
import torch

ROOT=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907')
BASE=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/HunanDyDiff_20261028')
RAW=Path('/public/home/slfu/ttzhou/Auxiliary_data/HuNan_AGRI_Preprocessed')
SROOT=ROOT/'experiments/hunan_monthly_block_20261002'
RROOT=ROOT/'experiments/r_solar_geometry_corrected_20261015'
PACK=ROOT/'experiments/cot_interval_history_20261015_v1/data_attempt2'
HEADROOT=ROOT/'experiments/cloud_probability_input_20261020'
HEADS=HEADROOT/'results/four_arm_paired_v1_attempt1'
PRED=BASE/'cross_hubei_on_hunan_20261028/attempt1'
CHANNELS=(0,1,2,3,4,5,8,9,10,11,12,13,14)
STATIONS=('sili','zhujia');SEEDS=(42,43,44)
COORDS=((163,148),(73,150))

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def csvrows(p):
 with Path(p).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--prediction-root',type=Path,required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--rank',type=int,default=0);ap.add_argument('--world',type=int,default=1);a=ap.parse_args();global PRED;PRED=a.prediction_root;device=torch.device('cuda',a.rank);torch.cuda.set_device(device)
 assert not a.output.exists(),'new attempt required';a.output.mkdir(parents=True)
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 sys.path.insert(0,str(SROOT/'code'));sys.path.insert(0,str(RROOT));sys.path.insert(0,str(HEADROOT))
 from triad_models import SimVPBackbone
 from hunan_data import load_frame,resolve_aux
 from train_paired_r import COTUNet
 from train_four_arm import FourArmIntervalHead,EXPECTED_COT_MEAN,EXPECTED_COT_STD
 from solar_geometry_corrected import compute_solar_angles
 selection=json.loads((PRED/'selection.json').read_text())
 assert len(selection['indices'])==len(selection['rows']) and len(selection['indices'])==(16 if a.smoke else 3140)
 selection['indices']=selection['indices'][a.rank::a.world];selection['rows']=selection['rows'][a.rank::a.world]
 rc=RROOT/'results/paired_v1/corrected/best.pt';rn=rc.parent/'normalization.json'
 assert sha(rc)=='7273a21d1d0c44d0aef6002f3e32e66b68a55f9001f1828bd1f285a5c19c680a'
 assert sha(rn)=='a1fcb4666082d70e213ef3cd62d546c1cd162774a8fdf4583483a00019696869'
 rnorm=json.loads(rn.read_text());rmean=np.array(rnorm['mean'],np.float32)[:,None,None];rstd=np.array(rnorm['std'],np.float32)[:,None,None]
 rmodel=COTUNet().eval().requires_grad_(False);rmodel.load_state_dict(torch.load(rc,map_location='cpu',weights_only=False)['model']);rmodel.to(device)
 sp=SROOT/'runs/simvp_seed42_v1/training/best.pt'
 assert sha(sp)=='f255da84cd57f058cdbbcc09204123366055e7b0e94310a79708295e11c50ec4'
 sim=SimVPBackbone(str(SROOT/'code')).eval().requires_grad_(False)
 ck=torch.load(sp,map_location='cpu',weights_only=False);assert ck['metadata']['test_payloads_read']==0
 sim.load_state_dict(ck['model']);sim.to(device);del ck
 norm=json.loads((SROOT/'manifests_v1/hunan_train_norm.json').read_text())
 sm=np.array(norm['mean'],np.float32)[:,None,None];ss=np.array(norm['std'],np.float32)[:,None,None]
 with np.load(BASE/'grid_static_hunan.npz') as g:grid={k:np.array(g[k],np.float32) for k in ('lat','lon','saa')}
 gm=json.loads((PACK/'pack_manifest.json').read_text());assert gm['test_payload_opened'] is False
 am=np.array(gm['AGRI_norm_mean'],np.float32)[None,:,None,None];ast=np.array(gm['AGRI_norm_std'],np.float32)[None,:,None,None]
 grow=csvrows(PACK/'rows.csv');ghi_keys={(r['station'],r['target_time_utc']):(i,r) for i,r in enumerate(grow) if r['split']=='val'}
 exclusions={(v['station'],v['bjt_date']) for v in json.loads((HEADROOT/'GHI_STATION_DAY_EXCLUSIONS_20261019.json').read_text())['exclusions']}
 cp=RROOT/'data/paired_common_v1';cprows=csvrows(cp/'rows_supervised.csv')
 # Array positions index the FULL supervised table; never enumerate a val-filtered table.
 cpp_keys={(r['station'],r['target_time_utc']):i for i,r in enumerate(cprows) if r['split']=='val'}
 cpp=np.load(cp/'cot.npy',mmap_mode='r');cppmask=np.load(cp/'common_mask.npy',mmap_mode='r')
 assert cpp.shape[0]==len(cprows)==cppmask.shape[0]
 provenance={'corrected_R_sha256':sha(rc),'R_normalization_sha256':sha(rn),'simvp_checkpoint_sha256':sha(sp),
  'GHI_pack_sha256':sha(PACK/'pack_manifest.json'),'GHI_rows_sha256':sha(PACK/'rows.csv'),'head_setup_sha256':sha(HEADS/'setup.json'),
  'geometry':'R and GHI use their corrected trained NOAA convention: night SOA0, day_mask SOZ<85; SimVP retains original trained geometry only',
  'CPP_join':'full rows_supervised.csv index joined by station/UTC target, val only; no filtered-local indexing',
  'future_validity':'offline scoring only; predicted AGRI input mask is finite prediction, never actual future QA',
  'heads_training':'frozen same-time observed three-frame inputs; no DyDiff/SimVP-adapted retraining',
  'test_payloads_read':0,'script_sha256':sha(__file__)}
 write(a.output/'provenance.json',provenance)
 observations=[];features=[];meta=[];case_arrays=[];excluded=[]

 def geometry(t,rr,cc,size=16):
  r0,c0=rr-8,cc-8;sl=np.s_[r0:r0+size,c0:c0+size]
  soz,soa=compute_solar_angles(grid['lat'][sl],grid['lon'][sl],t);soa=np.where(soz>=90,0,soa)
  return np.stack([np.cos(np.deg2rad(soz)),np.cos(np.deg2rad(soa-grid['saa'][sl])),(soz<85).astype(np.float32)]).astype(np.float32)
 def infer_r(agri,geo):
  x=np.concatenate([agri,geo],axis=2).reshape(-1,16,16,16)
  x=(x-rmean)/rstd;assert np.isfinite(x).all()
  with torch.inference_mode():out=rmodel(torch.from_numpy(x).to(device)).cpu().numpy()
  assert np.isfinite(out).all()
  return out.reshape(agri.shape[0],2,1,16,16)

 started=time.monotonic()
 for wi,(idx,row) in enumerate(zip(selection['indices'],selection['rows'])):
  assert row['split']=='val';paths=row['data_relpaths'].split('|');ts=[datetime.strptime(Path(r).stem,'%Y%m%d%H%M') for r in paths]
  init=ts[7];assert all(t-init==timedelta(minutes=15*(j-7)) for j,t in enumerate(ts))
  obs=np.empty((24,2,13,16,16),np.float32);valid=np.empty_like(obs,dtype=bool);geom=np.empty((24,2,3,16,16),np.float32)
  hist=[];oldgeo=[]
  for ti,(rel,t) in enumerate(zip(paths,ts)):
   raw=np.load(RAW/'data'/rel,mmap_mode='r',allow_pickle=False)
   with np.load(resolve_aux(RAW,t+timedelta(hours=8))) as z:
    bits=z['agri_valid_bits']
    oldgeo.append(np.stack([z[k] for k in ('cosSOZ','cosRAA','day_mask')]).astype(np.float32))
    for si,(rr,cc) in enumerate(COORDS):
     r0,c0=rr-8,cc-8;patch=np.asarray(raw[np.array(CHANNELS),r0:r0+16,c0:c0+16],np.float32)
     vb=bits[r0:r0+16,c0:c0+16];v=np.stack([(vb&(1<<k))!=0 for k in CHANNELS])&np.isfinite(patch)
     obs[ti,si]=np.where(v,patch,sm);valid[ti,si]=v;geom[ti,si]=geometry(t,rr,cc)
    if ti<8:
     x=np.asarray(raw[list(CHANNELS)],np.float32);v=np.stack([(bits&(1<<k))!=0 for k in CHANNELS])&np.isfinite(x)
     hist.append(np.where(v,(x-sm)/ss,0).astype(np.float32))
  with torch.inference_mode():
   sf=sim(torch.from_numpy(np.stack(hist)[None]).to(device),torch.from_numpy(np.stack(oldgeo)[None]).to(device)).cpu().numpy()[0]*ss+sm
  assert np.isfinite(sf).all()
  df=np.load(PRED/f'val_{idx:05d}_pred.npy',mmap_mode='r');assert df.shape in ((16,13,256,256),(16,2,13,16,16))
  images={'observed':obs,'dydiff':obs.copy(),'simvp':obs.copy()}
  for si,(rr,cc) in enumerate(COORDS):
   images['dydiff'][8:,si]=df[:,si] if df.ndim==5 else df[:,:,rr-8:rr+8,cc-8:cc+8]
   images['simvp'][8:,si]=sf[:,:,rr-8:rr+8,cc-8:cc+8]
  cots={'observed':infer_r(images['observed'],geom)}
  for method in ('dydiff','simvp'):
   cots[method]=np.concatenate([cots['observed'][:8],infer_r(images[method][8:],geom[8:])])
  for k in range(16):
   for si,station in enumerate(STATIONS):
    target=ts[k+8].strftime('%Y-%m-%d %H:%M:%S');day=(ts[k+8]+timedelta(hours=8)).date().isoformat();key=(station,target)
    ci=cpp_keys.get(key);cotref=np.full((16,16),np.nan,np.float32);cmask=np.zeros((16,16),bool)
    if ci is not None:
     assert cprows[ci]['station']==station and cprows[ci]['target_time_utc']==target and cprows[ci]['split']=='val'
     cotref=np.array(cpp[ci,0]);cmask=np.array(cppmask[ci,0],bool)
    observations.append({'window':idx,'station':station,'init_time_utc':str(init),'target_time_utc':target,'target_date_bjt':day,'lead_minutes':15*(k+1),'hubei_train_day_overlap':int(idx in selection.get('hubei_training_day_overlap_indices',[]))})
    case_arrays.append({'cot_observed':cots['observed'][k+8,si,0], 'cot_dydiff':cots['dydiff'][k+8,si,0], 'cot_simvp':cots['simvp'][k+8,si,0],
     'cot_persistence':cots['observed'][7,si,0], 'cpp':cotref,'cpp_mask':cmask,
     'observed_valid_center':bool(valid[k+8,si,:,8,8].all() and geom[k+8,si,0,8,8]>np.cos(np.deg2rad(80)))})
    if key not in ghi_keys or (station,day) in exclusions:
     excluded.append({'window':idx,'station':station,'target_time_utc':target,'reason':'approved_invalid_day' if (station,day) in exclusions else 'no_frozen_valid_GHI_row'});continue
    pi,gr=ghi_keys[key];assert int(gr['row'])==COORDS[si][0] and int(gr['col'])==COORDS[si][1]
    assert gr['earlier_time_utc']==str(ts[k+6]) and gr['previous_time_utc']==str(ts[k+7])
    clear=float(gr['clear']);y=float(gr['ghi']);assert clear>0 and np.isfinite(y)
    entry={**observations[-1],'pack_index':pi,'ghi':y,'clear':clear,'suspect_daylight_long_zero':gr.get('suspect_daylight_long_zero','0')};meta.append(entry)
    fi={};sl=slice(k+6,k+9)
    for method in ('observed','dydiff','simvp'):
     ag=images[method][sl,si,:,7:10,7:10];v=valid[sl,si,:,7:10,7:10].copy()
     # Only actual historical frames use source QA. ALL future prediction slots
     # use their own finite values; reference masks cannot become model inputs.
     if method!='observed':
      for j,frame_index in enumerate(range(k+6,k+9)):
       if frame_index>=8:v[j]=np.isfinite(ag[j])
     ag=np.where(v,(ag-am)/ast,0).astype(np.float32)
     g=geom[sl,si,:2,7:10,7:10]
     c=(cots[method][sl,si,:,7:10,7:10]-EXPECTED_COT_MEAN)/EXPECTED_COT_STD
     z=np.zeros((3,1,3,3),np.float32)
     fi[method]=np.concatenate([ag,g,c,z],axis=1).astype(np.float32)
     if method!='observed':
      oracle=fi[method].copy();oracle[:,15]=(cots['observed'][sl,si,0,7:10,7:10]-EXPECTED_COT_MEAN)/EXPECTED_COT_STD
      histcot=fi[method].copy()
      for j,frame_index in enumerate(range(k+6,k+9)):
       if frame_index>=8:histcot[j,15]=(cots['observed'][7,si,0,7:10,7:10]-EXPECTED_COT_MEAN)/EXPECTED_COT_STD
      fi[method+'_oracle_cot']=oracle;fi[method+'_history_cot']=histcot
    features.append(fi)
  print(json.dumps({'event':'cot_window_complete','window':idx,'completed':wi+1,'elapsed_seconds':time.monotonic()-started,'GHI_rows':len(meta)}),flush=True)
  write(a.output/'progress.json',{'state':'CPU_R_SIMVP_INFERENCE','windows_complete':wi+1,'windows_total':len(selection['indices']),'GHI_rows':len(meta),'test_payloads_read':0})
  del sf,df
 arrays={k:np.stack([r[k] for r in case_arrays]) for k in case_arrays[0]}
 np.savez_compressed(a.output/'cot_predictions.npz',**arrays)
 with (a.output/'cot_rows.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(observations[0]));w.writeheader();w.writerows(observations)
 assert meta and features
 with (a.output/'ghi_rows.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(meta[0]));w.writeheader();w.writerows(meta)
 write(a.output/'excluded_ghi_rows.json',excluded)
 x={method:np.stack([v[method] for v in features]) for method in features[0]}
 for v in x.values():assert v.shape==(len(meta),3,17,3,3) and np.isfinite(v).all()
 y=np.array([r['ghi'] for r in meta],np.float64);clear=np.array([r['clear'] for r in meta],np.float32);sites=np.array([r['station'] for r in meta])
 predictions={};head_hashes={};replay=[]
 # Check already saved real-input validation predictions first. No new training.
 with np.load(HEADS/'validation_inputs.npz') as z:
  lookup={int(v):i for i,v in enumerate(z['pack_index'])};vr=np.array([lookup[r['pack_index']] for r in meta])
  stored=np.concatenate([z['agri'][vr],z['geometry'][vr],z['cot_corrected_log1p_normalized'][vr],np.zeros_like(z['p_cloud_centered'][vr])],axis=2)
  assert np.max(np.abs(z['ghi'][vr]-y))<1e-3 and np.max(np.abs(z['clear'][vr]-clear))<1e-3
 for si,station in enumerate(STATIONS):
  ids=np.flatnonzero(sites==station)
  if not len(ids):continue
  for seed in SEEDS:
   for arm in ('M00','M10'):
    hp=HEADS/f'seed{seed}'/station/arm/'best.pt';hpayload=torch.load(hp,map_location='cpu',weights_only=False)
    assert hpayload['station']==station and hpayload['seed']==seed and hpayload['arm']==arm and hpayload['test_payloads_read']==0
    head=FourArmIntervalHead().eval().requires_grad_(False);head.load_state_dict(hpayload['model']);head.to(device);head_hashes[f'{station}/{seed}/{arm}']=sha(hp)
    xx=stored[ids].copy()
    if arm=='M00':xx[:,:,15]=0
    with torch.inference_mode():rep=(head(torch.from_numpy(xx).to(device))*torch.from_numpy(clear[ids]).to(device)).cpu().numpy().astype(np.float64)
    with np.load(hp.parent/'validation_predictions.npz') as z:
     saved={int(v):float(p) for v,p in zip(z['pack_index'],z['prediction'])}
     expected=np.array([saved[meta[i]['pack_index']] for i in ids])
    diff=float(abs(rep-expected).max());assert diff<.1,('CPU_replay',station,seed,arm,diff)
    replay.append({'station':station,'seed':seed,'arm':arm,'max_abs_Wm2':diff})
    methods=('observed','dydiff','simvp') if arm=='M00' else tuple(x)
    for method in methods:
     if method not in x:continue
     xx=x[method][ids].copy()
     if arm=='M00':xx[:,:,15]=0
     with torch.inference_mode():pred=(head(torch.from_numpy(xx).to(device))*torch.from_numpy(clear[ids]).to(device)).cpu().numpy().astype(np.float64)
     arr=np.full(len(meta),np.nan,np.float64);arr[ids]=pred;name=f'{station}_seed{seed}_{arm}_{method}';predictions[name]=arr
 np.savez_compressed(a.output/'ghi_predictions.npz',truth=y,clear=clear.astype(np.float64),**predictions)
 write(a.output/'cpu_replay.json',{'state':'PASS_FROZEN_GHI_CPU_REPLAY','max_abs_Wm2':max(v['max_abs_Wm2'] for v in replay),'per_run':replay})
 write(a.output/'completion.json',{'status':'PASS','windows':len(selection['indices']),'COT_rows':len(observations),'GHI_rows':len(meta),'stations':{s:int(sum(v['station']==s for v in meta)) for s in STATIONS},'head_sha256':head_hashes,
  'elapsed_seconds':time.monotonic()-started,'R_precision':'GPU_FP32','SimVP_precision':'GPU_FP32, original frozen weights and trained legacy geometry; no fine tuning','scope':'all supplied frozen validation windows; frozen same-time GHI heads replayed on forecast inputs','test_payloads_read':0})
 print('COT_GHI_CPU_DIAGNOSTIC_COMPLETE',flush=True)
if __name__=='__main__':main()
