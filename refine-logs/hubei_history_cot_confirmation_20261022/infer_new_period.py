"""Frozen S/R/GHI inference on NEW dates; never load future real AGRI."""
import argparse,csv,hashlib,json,os,sys,time
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader

EXP=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments')
BASE=EXP/'hubei_solar_geometry_repair_20261021'
HEAD=EXP/'hubei_forecast_cot_increment_20261022'
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'code_pipeline_attempt12'))
sys.path.insert(0,str(HEAD/'code_attempt1'))
from solar_geometry_noaa import compute_solar_angles
from history_base import HistoryCOTGHIHead,CENTERS
AGRI=[0,1,2,3,4,5,8,9,10,11,12,13,14]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def readcsv(p):return list(csv.DictReader(Path(p).open()))
def geom(grid,t):
    z,az,day=compute_solar_angles(grid['lat'],grid['lon'],t-timedelta(hours=8))
    az=np.where(day>.5,az,0.)
    return np.stack((np.cos(np.deg2rad(z)),np.cos(np.deg2rad(az-grid['saa'])),day)).astype(np.float32)
def stamp(t):return (t-timedelta(hours=8)).strftime('%Y%m%d%H%M')

class PastOnly(Dataset):
    def __init__(self,root,selection,grid,norm,shard,world):
        self.root=root;self.grid=grid;self.norm=norm
        self.inits=[datetime.fromisoformat(t) for t in selection['initializations_BJT']]
        self.ids=list(range(len(self.inits)))[shard::world]
        self.im=norm['input_mean'][None,:,None,None];self.sd=norm['input_std'][None,:,None,None]
    def __len__(self):return len(self.ids)
    def __getitem__(self,k):
        ix=self.ids[k];init=self.inits[ix];hist=[];allgeo=[]
        for off in range(-105,241,15):
            t=init+timedelta(minutes=off);assert t>=datetime(2026,7,15)
            if off<=0:
                raw=np.load(self.root/'data'/f'{stamp(t)}.npy',allow_pickle=False)
                stored=np.stack((np.cos(np.deg2rad(raw[18])),np.cos(np.deg2rad(raw[17]-raw[15])),raw[19]))
                # Match corrected training aux: compute trig before float32
                # rounding, rather than taking cosine of stored float32 SOZ.
                g=geom(self.grid,t);assert np.max(abs(stored-g))<1e-6
                assert raw.shape==(20,256,256) and np.isfinite(raw).all()
                hist.append(np.concatenate((raw[AGRI],g),axis=0))
            else:g=geom(self.grid,t)
            allgeo.append(g)
        x=(np.stack(hist)-self.im)/self.sd
        aux=np.stack(allgeo);am=self.norm['input_mean'][13:][None,:,None,None];sd=self.norm['input_std'][13:][None,:,None,None]
        return torch.from_numpy(x.astype(np.float32)),torch.from_numpy(((aux-am)/sd).astype(np.float32)),ix

def s_model(out,freeze,norm,device='cuda'):
    sys.path.insert(0,str(BASE/'simvp_code_snapshot'))
    from openstl.methods.simvp import SimVP
    cfg=dict(method='simvp',dataname='hbagri',data_name='hbagri',pre_seq_length=8,aft_seq_length=16,total_length=24,
        in_shape=[8,16,256,256],out_channels=13,model_type='gSTA',hid_S=32,hid_T=256,N_S=4,N_T=8,
        spatio_kernel_enc=3,spatio_kernel_dec=3,drop_path=.1,mlp_ratio=8.,drop=0.,metrics=['mse','mae'],
        steps_per_epoch=1,test_mean=norm['agri_mean'],test_std=norm['agri_std'],save_dir=str(out),seed=42,
        epoch=50,opt='adamw',lr=1e-3,weight_decay=1e-4,sched='cosine',min_lr=1e-5,warmup_epoch=5)
    ck=freeze['S']['best_checkpoint'];assert sha(ck)==freeze['S']['best_checkpoint_sha256']
    model=SimVP(**cfg);state=torch.load(ck,map_location='cpu',weights_only=False)['state_dict']
    model.load_state_dict(state,strict=True);return model.to(device).eval()

def infer_s(a,freeze,selection,grid):
    assert sha(freeze['S_norm_path'])==freeze['S_norm_sha256']
    with np.load(freeze['S_norm_path'],allow_pickle=False) as z:
        norm={k:np.asarray(z[k],dtype=np.float32) for k in ['agri_mean','agri_std','input_mean','input_std']}
    assert norm['agri_mean'].shape==norm['agri_std'].shape==(13,)
    assert norm['input_mean'].shape==norm['input_std'].shape==(16,)
    ds=PastOnly(a.data,selection,grid,norm,a.shard,a.world)
    if a.mode=='profile':ds.ids=ds.ids[:16]
    loader=DataLoader(ds,batch_size=16,shuffle=False,num_workers=4,pin_memory=True)
    a.out.mkdir(exist_ok=False);model=s_model(a.out,freeze,norm)
    if a.mode=='profile':
        x,aux,ix=next(iter(loader));x=x.cuda();aux=aux.cuda()
        with torch.inference_mode():
            for _ in range(2):q=model(x,batch_aux=aux)
            torch.cuda.synchronize();start=time.monotonic()
            for _ in range(4):q=model(x,batch_aux=aux)
            torch.cuda.synchronize();elapsed=time.monotonic()-start
        assert torch.isfinite(q).all()
        (a.out/'PROFILE.json').write_text(json.dumps(dict(sequences_per_second=len(x)*4/elapsed,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(),batch=len(x),seconds=elapsed,gpu=torch.cuda.get_device_name(0)))+'\n');return
    pred=np.lib.format.open_memmap(a.out/'pred.npy',mode='w+',dtype=np.float32,shape=(len(ds),16,13,16,16))
    geo=np.lib.format.open_memmap(a.out/'geometry.npy',mode='w+',dtype=np.float32,shape=(len(ds),16,3,16,16))
    ids=[];cursor=0;started=time.monotonic()
    with torch.inference_mode():
        for x,aux,ix in loader:
            y=model(x.cuda(non_blocking=True),batch_aux=aux.cuda(non_blocking=True)).float().cpu().numpy()
            assert y.shape[1:]==(16,13,256,256) and np.isfinite(y).all()
            y=y[:,:,:,103:119,118:134]*norm['agri_std'][None,None,:,None,None]+norm['agri_mean'][None,None,:,None,None]
            g=aux.numpy()[:,8:,:,103:119,118:134]*norm['input_std'][None,None,13:,None,None]+norm['input_mean'][None,None,13:,None,None]
            n=len(x);pred[cursor:cursor+n]=y;geo[cursor:cursor+n]=g;cursor+=n;ids.extend(ix.tolist())
            print(json.dumps(dict(shard=a.shard,done=cursor,total=len(ds),seconds=time.monotonic()-started)),flush=True)
    pred.flush();geo.flush();assert cursor==len(ds)
    (a.out/'COMPLETE.json').write_text(json.dumps(dict(state='PAST_ONLY_SIMVP_INFERENCE_COMPLETE',sequence_ids=ids,
        seconds=time.monotonic()-started,pred_sha256=sha(a.out/'pred.npy'),geometry_sha256=sha(a.out/'geometry.npy'),
        checkpoint_sha256=freeze['S']['best_checkpoint_sha256'],future_real_AGRI_payloads_read=0,
        old_test_payloads_read=0,old_buffer_payloads_read=0),indent=2)+'\n')

def r_and_ghi(a,freeze,selection,grid):
    from train_cot_repaired import COTUNet
    rmeta=json.loads((BASE/'R_runs_attempt12/full/run_metadata.json').read_text())
    assert sha(BASE/'code_pipeline_attempt12/train_cot_repaired.py')==rmeta['model_source_sha256']
    a.out.mkdir(exist_ok=False);frames=[datetime.fromisoformat(t) for t in selection['history_times_BJT']]
    features=[]
    for t in frames:
        raw=np.load(a.data/'data'/f'{stamp(t)}.npy',mmap_mode='r',allow_pickle=False)
        g=geom(grid,t)
        features.append(np.concatenate((raw[AGRI,103:119,118:134],g[:,103:119,118:134]),axis=0))
    features=np.stack(features).astype(np.float32);valid=(features[:,15:16]>.5)&(features[:,13:14]>np.cos(np.deg2rad(80.)))
    assert sha(freeze['R_norm_path'])==freeze['R_norm_sha256']
    rn=json.loads(Path(freeze['R_norm_path']).read_text());mu=np.array(rn['mean'],np.float32)[None,:,None,None];sd=np.array(rn['std'],np.float32)[None,:,None,None]
    r=COTUNet().cuda().eval();assert sha(freeze['R_checkpoint'])==freeze['R_checkpoint_sha256']
    r.load_state_dict(torch.load(freeze['R_checkpoint'],map_location='cuda',weights_only=False)['model'])
    cot=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        for j in range(0,len(features),64):
            q=torch.expm1(r(torch.from_numpy((features[j:j+64]-mu)/sd).cuda()).float())
            cot.append(q.masked_fill(~torch.from_numpy(valid[j:j+64]).cuda(),0.).cpu().numpy())
    cot=np.concatenate(cot);assert np.isfinite(cot).all() and np.all(cot>=0)
    np.save(a.out/'history_cot.npy',cot);del r
    frame_index={t:i for i,t in enumerate(frames)}
    allinits=[datetime.fromisoformat(t) for t in selection['initializations_BJT']];n=len(allinits)
    pred=np.empty((n,16,13,16,16),np.float32);geo=np.empty((n,16,3,16,16),np.float32);seen=set()
    for p in sorted(a.s_bank.glob('shard*/COMPLETE.json')):
        m=json.loads(p.read_text());ids=m['sequence_ids'];assert not seen.intersection(ids);seen.update(ids)
        assert sha(p.parent/'pred.npy')==m['pred_sha256'] and sha(p.parent/'geometry.npy')==m['geometry_sha256']
        pred[ids]=np.load(p.parent/'pred.npy');geo[ids]=np.load(p.parent/'geometry.npy')
    assert seen==set(range(n));initindex={t:i for i,t in enumerate(allinits)}
    keys=readcsv(a.keys);labels={(r['station'],r['target_time_BJT']):r for r in readcsv(a.labels)}
    # Use the already frozen positive clear-sky criterion, same for both arms.
    keys=[k for k in keys if labels[(k['station'],k['target_time_BJT'])]['valid_daylight_clear_sky_label']=='1']
    heads=freeze['GHI_heads'];receipts=[]
    for station in ['Niushou','Wangzhai','Mizhuang']:
        rr=[k for k in keys if k['station']==station];cy,cx=CENTERS[station];ys=slice(cy-1,cy+2);xs=slice(cx-1,cx+2)
        si=np.array([initindex[datetime.fromisoformat(k['init_time_BJT'])] for k in rr]);li=np.array([int(k['lead_minutes'])//15-1 for k in rr])
        hi=np.array([[frame_index[datetime.fromisoformat(k['init_time_BJT'])-timedelta(minutes=off)] for off in range(105,-1,-15)] for k in rr])
        future=np.concatenate((pred[si,li,:,ys,xs],geo[si,li,:2,ys,xs]),axis=1)
        history=features[hi,:15,ys,xs];physical_cot=cot[hi,:,ys,xs];vm=valid[hi,:,ys,xs]
        clear=np.array([float(labels[(station,k['target_time_BJT'])]['clear_sky_ghi_15min_wm2']) for k in rr],np.float32)
        truth=np.array([float(labels[(station,k['target_time_BJT'])]['ghi_wm2']) for k in rr],np.float64)
        for seed in [42,43,44]:
            for arm in ['G0','G1']:
                spec=next(s for s in heads if s['station']==station and s['seed']==seed and s['arm']==arm)
                assert sha(spec['path'])==spec['sha256'] and sha(spec['norm_source'])==spec['norm_source_sha256']
                norms=json.loads(Path(spec['norm_source']).read_text())['data_meta']['base_norm']
                f=future.copy();h=history.copy()
                f[:,:13]=(f[:,:13]-np.array(norms['future_agri_mean'],np.float32)[None,:,None,None])/np.array(norms['future_agri_std'],np.float32)[None,:,None,None]
                h[:,:,:13]=(h[:,:,:13]-np.array(norms['history_agri_mean'],np.float32)[None,None,:,None,None])/np.array(norms['history_agri_std'],np.float32)[None,None,:,None,None]
                c=np.where(vm,(np.log1p(physical_cot)-np.float32(norms['history_cot_log1p_mean']))/np.float32(norms['history_cot_log1p_std']),np.float32(0)).astype(np.float32)
                model=HistoryCOTGHIHead().cuda().eval();model.load_state_dict(torch.load(spec['path'],map_location='cuda',weights_only=False)['model'])
                out=[]
                with torch.inference_mode():
                    for j in range(0,len(rr),1536):
                        x=[torch.from_numpy(np.ascontiguousarray(v[j:j+1536])).cuda() for v in [f,h,c]]
                        if arm=='G0':x[2].zero_()
                        out.append(model(*x).cpu().numpy())
                kt=np.concatenate(out);assert np.isfinite(kt).all()
                dest=a.out/f'{station}_seed{seed}_{arm}.npz'
                np.savez_compressed(dest,pred_ghi=kt.astype(np.float64)*clear.astype(np.float64),ghi=truth,clear=clear,
                    init_time_BJT=np.array([k['init_time_BJT'] for k in rr]),target_time_BJT=np.array([k['target_time_BJT'] for k in rr]),
                    lead_minutes=np.array([int(k['lead_minutes']) for k in rr]),analysis=np.array([k['analysis'] for k in rr]))
                receipts.append(dict(station=station,seed=seed,arm=arm,rows=len(rr),checkpoint_sha256=spec['sha256'],file=str(dest),sha256=sha(dest)))
    (a.out/'COMPLETE.json').write_text(json.dumps(dict(state='FROZEN_NEW_PERIOD_PAIRED_GHI_PREDICTIONS_COMPLETE',predictions=receipts,
        freeze_sha256=sha(a.freeze),old_test_payloads_read=0,old_buffer_payloads_read=0,future_real_AGRI_used=False,
        historical_COT_only=True,new_period_used_for_model_selection=False),indent=2)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['S','profile','R_GHI'],required=True)
    for name in ['data','selection','grid','freeze','out']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['keys','labels','s-bank']:p.add_argument('--'+name,type=Path)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--world',type=int,default=1);a=p.parse_args()
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    freeze=json.loads(a.freeze.read_text());selection=json.loads(a.selection.read_text());grid=dict(np.load(a.grid))
    if a.mode in ['S','profile']:infer_s(a,freeze,selection,grid)
    else:r_and_ghi(a,freeze,selection,grid)
if __name__=='__main__':main()
