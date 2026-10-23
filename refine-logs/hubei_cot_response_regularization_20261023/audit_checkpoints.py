"""Read-only CPU cold checkpoint replay and selected-model sensitivity."""
import csv,json,sys
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent;H=P.parent/'hubei_forecast_cot_increment_20261022'
sys.path.insert(0,str(H/'code_attempt1'))
from train import ConditionalHead
from history_base import sha256
from regularization import response_loss
torch.set_num_threads(1);records=[]
for st in ['Niushou','Wangzhai','Mizhuang']:
    p=H/'prepared_attempt1'/st;meta=json.loads((p/'meta.json').read_text());take=np.flatnonzero(np.load(p/'roles.npy')=='C_select')
    ff=np.array(np.load(p/'fusion_features.npy',mmap_mode='r')[take]);future=np.load(p/'future.npy',mmap_mode='r')[take]
    valid=(future[:,13].reshape(-1,9)>np.cos(np.deg2rad(80)))&ff[:,297:306].astype(bool)
    for lo in [279,288]:ff[:,lo:lo+9]=np.where(valid,ff[:,lo:lo+9],0)
    ff[:,297:306]=valid;available=torch.from_numpy(valid.all(1).astype('float32'));hv=torch.from_numpy(ff[:,270:279]);sd=meta['future_cot_norm']['log1p_std']
    for seed in [42,43,44]:
        folder=P/'replay'/st
        with np.load(folder/f'seed{seed}_Cselect.npz') as z:d={k:z[k] for k in z.files}
        base=torch.from_numpy((d['prediction'][0]/d['clear']).astype('float32'));features=torch.cat([torch.from_numpy(ff),base[:,None]],1)
        for ai,arm in enumerate(d['arms'].tolist()[1:],1):
            bestpath=folder/f'seed{seed}_{arm}_best.pt';best=torch.load(bestpath,map_location='cpu',weights_only=False);last=torch.load(folder/f'seed{seed}_{arm}_last.pt',map_location='cpu',weights_only=False)
            assert last['updates']==last['step']==60 and last['scheduler']['last_epoch']==60
            assert all(int(v['step'])==60 for v in last['optimizer']['state'].values())
            assert abs(last['optimizer']['param_groups'][0]['lr']-1e-5)<1e-15
            model=ConditionalHead(308).eval();model.load_state_dict(best['model']);x=features.clone()
            if 'no_future' in arm:x[:,279:297]=0
            with torch.no_grad():pred=model(x,base,available).double().numpy()*d['clear']
            difference=float(abs(pred-d['prediction'][ai]).max());assert difference<.001,(st,seed,arm,difference)
            model=model.double().requires_grad_(False);x=x.double().detach().requires_grad_(True)
            y,omega,direction=response_loss(model,x,base.double(),available.double(),hv.double(),sd,use_cot='no_future' not in arm)
            assert torch.isfinite(direction).all() and torch.count_nonzero(direction[available==0])==0
            mask=available.numpy()>0;direction=direction.detach().numpy()[mask]
            records.append(dict(station=st,seed=seed,arm=arm,step=best['step'],updates=60,cold_replay_max=difference,
                positive_fraction=float((direction>1e-9).mean()),mean_positive_kt_response=float(np.maximum(direction,0).mean()),
                penalty=float(omega.detach()),best_sha256=sha256(bestpath)))
(P/'CHECKPOINT_AUDIT.json').write_text(json.dumps(dict(state='PASS',records=records,test_payloads_read=0),indent=2))
print(json.dumps([r for r in records if r['station']=='Mizhuang'],indent=2))
