"""Frozen regional R and frozen SimVP: causal validation inference only."""
import argparse,hashlib,json,os,sys,time
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from regional_model import COTUNet
from solar_geometry_noaa import compute_solar_angles

ROOT=Path(__file__).resolve().parent
REPAIR=ROOT.parent/'hubei_solar_geometry_repair_20261021'
CHANNELS=[0,1,2,3,4,5,8,9,10,11,12,13,14]
CENTERS=[(111,124),(112,126),(110,128)]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def dt(k):return datetime.strptime(k,'%Y%m%d%H%M')
def fmt(t):return t.strftime('%Y%m%d%H%M')
def geometry(grid,t):
    z,az,day=compute_solar_angles(grid['lat'],grid['lon'],t-timedelta(hours=8))
    az=np.where(day>.5,az,0.)
    return np.stack([np.cos(np.deg2rad(z)),np.cos(np.deg2rad(az-grid['saa'])),day]).astype('float32')
def read_frame(row,data):
    raw=np.load(data/'data'/Path(row['data_path']).name,allow_pickle=False)
    with np.load(data/'aux'/Path(row['aux_path']).name,allow_pickle=False) as z:
        g=np.stack([z[k] for k in ('cosSOZ','cosRAA','day_mask')])
    assert raw.shape==(20,256,256) and g.shape==(3,256,256)
    assert np.isfinite(raw).all() and np.isfinite(g).all()
    return raw,g
def slices(x):return np.stack([x[...,r-1:r+2,c-1:c+2] for r,c in CENTERS],axis=-3)
def r_model():
    ck=ROOT/'r_region_seed42_attempt1/best.pt'
    assert sha(ck)=='ed7be2ea2c987557d3753b3eb7d8d0f9ffb4513b63e4afedfcbb54be132600e2'
    checkpoint=torch.load(ck,map_location='cpu')
    pack=Path('/dev/shm/hubei_regional_r_pack_20261023_attempt2')
    assert sha(pack/'SHA256.json')==checkpoint['binding']['pack_hash_file_sha256']
    assert sha(ROOT/'regional_model.py')==checkpoint['binding']['model_source_sha256']
    assert sha(pack/'norm.json')==json.loads((pack/'SHA256.json').read_text())['norm.json']
    r=COTUNet().cuda().eval();r.load_state_dict(checkpoint['model'],strict=True)
    norm=json.loads((pack/'norm.json').read_text())
    mu=np.asarray(norm['mean'],np.float32)[:,None,None];sd=np.asarray(norm['std'],np.float32)[:,None,None]
    return r,mu,sd
class Histories(Dataset):
    def __init__(self,selection,data,grid,norm,shard,world):
        self.rows={r['bjt']:r for r in selection['frames']};self.ids=list(range(len(selection['initializations'])))[shard::world]
        self.inits=selection['initializations'];self.data=data;self.grid=grid;self.norm=norm
    def __len__(self):return len(self.ids)
    def __getitem__(self,idx):
        ix=self.ids[idx];init=dt(self.inits[ix]);hist=[];geo=[]
        for offset in range(-105,241,15):
            t=init+timedelta(minutes=offset)
            if offset<=0:
                row=self.rows[fmt(t)];assert row['split']=='val'
                raw,g=read_frame(row,self.data);assert np.max(abs(g-geometry(self.grid,t)))<1e-6
                hist.append(np.concatenate([raw[CHANNELS],g]))
            else:g=geometry(self.grid,t)
            geo.append(g)
        x=(np.stack(hist)-self.norm['input_mean'][None,:,None,None])/self.norm['input_std'][None,:,None,None]
        g=(np.stack(geo)-self.norm['input_mean'][None,13:,None,None])/self.norm['input_std'][None,13:,None,None]
        return torch.from_numpy(x.astype('float32')),torch.from_numpy(g.astype('float32')),ix
def s_model(freeze,norm,out):
    sys.path.insert(0,str(REPAIR/'simvp_code_snapshot'))
    from openstl.methods.simvp import SimVP
    cfg=dict(method='simvp',dataname='hbagri',data_name='hbagri',pre_seq_length=8,aft_seq_length=16,total_length=24,
      in_shape=[8,16,256,256],out_channels=13,model_type='gSTA',hid_S=32,hid_T=256,N_S=4,N_T=8,
      spatio_kernel_enc=3,spatio_kernel_dec=3,drop_path=.1,mlp_ratio=8.,drop=0.,metrics=['mse','mae'],
      steps_per_epoch=1,test_mean=norm['agri_mean'],test_std=norm['agri_std'],save_dir=str(out),seed=42,
      epoch=50,opt='adamw',lr=1e-3,weight_decay=1e-4,sched='cosine',min_lr=1e-5,warmup_epoch=5)
    ck=freeze['S']['best_checkpoint'];assert sha(ck)==freeze['S']['best_checkpoint_sha256']
    model=SimVP(**cfg);model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=False)['state_dict'],strict=True)
    return model.cuda().eval()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['history','forecast'],required=True)
    ap.add_argument('--shard',type=int,required=True);ap.add_argument('--world',type=int,required=True)
    ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(exist_ok=False,parents=True)
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    selection=json.loads((ROOT/'G4_SELECTION.json').read_text());assert selection['split']=='val'
    audit=json.loads((ROOT/'G4_STAGE_AUDIT.json').read_text())
    assert audit['state']=='PASS_G4_HISTORY_LOCAL_HASHES'
    assert sha(ROOT/'G4_SELECTION.json')==audit['selection_sha256']
    model,mu,sd=r_model();started=time.monotonic()
    with torch.inference_mode():
        if a.mode=='history':
            frames=selection['frames'][a.shard::a.world];pred=[];valid=[];ir=[];keys=[]
            for start in range(0,len(frames),32):
                xx=[];vv=[]
                for row in frames[start:start+32]:
                    raw,g=read_frame(row,a.data);x=np.concatenate([raw[CHANNELS],g])
                    xx.append((x-mu)/sd);vv.append((g[2]>.5)&(g[0]>np.cos(np.deg2rad(80)))&(raw[8:15]>0).all(0))
                    ir.append(raw[12]);keys.append(row['bjt'])
                with torch.autocast('cuda',dtype=torch.float16):q=model(torch.from_numpy(np.stack(xx)).cuda()).float()
                assert torch.isfinite(q).all();pred.append(q.cpu().numpy()[:,0]);valid.extend(vv)
            np.savez(a.out/'history.npz',keys=np.asarray(keys),log_cot=np.concatenate(pred),valid=np.stack(valid),c13=np.stack(ir))
        else:
            freeze=json.loads((ROOT/'G4_MODEL_FREEZE.json').read_text())
            assert sha(freeze['S_norm_path'])==freeze['S_norm_sha256']
            with np.load(freeze['S_norm_path'],allow_pickle=False) as z:
                norm={k:np.asarray(z[k],np.float32) for k in ['agri_mean','agri_std','input_mean','input_std']}
            grid=dict(np.load(ROOT/'grid_static256.npz'))
            ds=Histories(selection,a.data,grid,norm,a.shard,a.world)
            loader=DataLoader(ds,batch_size=4,num_workers=2,pin_memory=True,shuffle=False)
            s=s_model(freeze,norm,a.out);ids=[];cot=[];features=[]
            for x,aux,ix in loader:
                q=s(x.cuda(non_blocking=True),batch_aux=aux.cuda(non_blocking=True)).float()
                assert q.shape[1:]==(16,13,256,256) and torch.isfinite(q).all()
                q=q*torch.as_tensor(norm['agri_std'],device='cuda')[None,None,:,None,None]+torch.as_tensor(norm['agri_mean'],device='cuda')[None,None,:,None,None]
                geo=aux[:,8:].cuda()*torch.as_tensor(norm['input_std'][13:],device='cuda')[None,None,:,None,None]+torch.as_tensor(norm['input_mean'][13:],device='cuda')[None,None,:,None,None]
                v=torch.cat([q,geo],2).reshape(-1,16,256,256)
                tm=torch.as_tensor(mu,device='cuda');ts=torch.as_tensor(sd,device='cuda');out=[]
                for start in range(0,len(v),32):
                    with torch.autocast('cuda',dtype=torch.float16):z=model((v[start:start+32]-tm)/ts).float()
                    assert torch.isfinite(z).all();out.append(z.cpu().numpy()[:,0])
                log=np.concatenate(out).reshape(len(x),16,256,256)
                cot.append(slices(log));features.append(slices(v.reshape(len(x),16,16,256,256).cpu().numpy()))
                ids.extend(ix.tolist());print(json.dumps(dict(mode=a.mode,shard=a.shard,done=len(ids),total=len(ds))),flush=True)
            np.savez(a.out/'forecast.npz',ids=np.asarray(ids),log_cot=np.concatenate(cot),future_features=np.concatenate(features))
    (a.out/'COMPLETE.json').write_text(json.dumps(dict(state='FROZEN_G4_INFERENCE_COMPLETE',mode=a.mode,shard=a.shard,
        seconds=time.monotonic()-started,test_payloads_read=0,future_observation_used_as_input=False,
        r_checkpoint_sha256='ed7be2ea2c987557d3753b3eb7d8d0f9ffb4513b63e4afedfcbb54be132600e2'),indent=2))
if __name__=='__main__':main()
