import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent;H=P.parent/'hubei_forecast_cot_increment_20261022';G=P.parent/'hubei_cot_calibration_coverage_20261023'
sys.path.insert(0,str(H/'code_attempt1'))
from train import ConditionalHead
from history_base import HistoryCOTGHIHead,sha256
ap=argparse.ArgumentParser();ap.add_argument('--station',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
p=H/'prepared_attempt1'/a.station;meta=json.loads((p/'meta.json').read_text());rows=list(csv.DictReader((p/'rows.csv').open()));roles=np.load(p/'roles.npy')
audit=json.loads((G/'METADATA_AUDIT.json').read_text())[a.station];assert sha256(p/'rows.csv')==audit['row_metadata_sha256']
wide=set(audit['wide_dates']);take=np.array([i for i,(r,v) in enumerate(zip(rows,roles)) if (r['split']=='train' and r['target_time_bjt'][:8] in wide) or v=='C_select'])
rr=[rows[i] for i in take];fit=np.array([r['target_time_bjt'][:8] in wide for r in rr]);select=roles[take]=='C_select';assert np.all(fit^select)
data={k:np.array(np.load(p/(k+'.npy'),mmap_mode='r')[take]) for k in ['future','history','cot','fusion_features','ghi','clear']}
ff=data['fusion_features'];solar=(data['future'][:,13].reshape(-1,9)>np.cos(np.deg2rad(80.)))&ff[:,297:306].astype(bool)
for lo in [279,288]:ff[:,lo:lo+9]=np.where(solar,ff[:,lo:lo+9],0)
ff[:,297:306]=solar;available=solar.all(1).astype('float32');sd=meta['future_cot_norm']['log1p_std'];histvalid=ff[:,270:279].astype(bool)
t={k:torch.from_numpy(data[k]).cuda() for k in ['future','history','cot']};clear=data['clear'].astype('float64');a.out.mkdir(exist_ok=False);checks=[]
for seed in [42,43,44]:
    baseck=H/'training_attempt1/pairs'/f'{a.station}_seed{seed}'/'G1_best.pt'
    bm=HistoryCOTGHIHead().cuda().eval().requires_grad_(False);bm.load_state_dict(torch.load(baseck,map_location='cuda',weights_only=False)['model'])
    with torch.no_grad():base=torch.cat([bm(t['future'][j:j+1536],t['history'][j:j+1536],t['cot'][j:j+1536]) for j in range(0,len(take),1536)]).detach()
    ck=G/'replay'/a.station/f'seed{seed}_large_future_COT_best.pt';saved=torch.load(ck,map_location='cuda',weights_only=False);assert saved['base_sha256']==sha256(baseck)
    model=ConditionalHead(308).cuda().eval().requires_grad_(False);model.load_state_dict(saved['model']);f=torch.cat([torch.from_numpy(ff).cuda(),base[:,None]],1);av=torch.from_numpy(available).cuda()
    with torch.no_grad():fpred=torch.cat([model(f[j:j+1536],base[j:j+1536],av[j:j+1536]) for j in range(0,len(take),1536)]).double().cpu().numpy()*clear
    with np.load(P/'replay'/a.station/f'seed{seed}_response.npz') as z:
        assert np.array_equal(z['truth'],data['ghi'][select])
        replay=float(abs(fpred[select]-z['prediction'][2]).max());assert replay<.001,replay
    model=model.double();derivatives=[];predictions=[];fdmax=0.;doublemax=0.
    for j in range(0,len(take),1536):
        x=f[j:j+1536].double().detach().requires_grad_(True);b=base[j:j+1536].double();v=av[j:j+1536].double();c=torch.as_tensor(clear[j:j+1536],device='cuda');hv=torch.as_tensor(histvalid[j:j+1536],device='cuda')
        y=model(x,b,v);grad=torch.autograd.grad(y.sum(),x)[0]
        per=(grad[:,279:288]+grad[:,288:297]*hv)/sd*c[:,None]
        deriv=torch.stack([per[:,4],per.sum(1)],1);assert torch.equal(deriv[v==0],torch.zeros_like(deriv[v==0]))
        for axis in [0,1]:
            direction=torch.zeros_like(x);pixels=[4] if axis==0 else list(range(9))
            for k in pixels:direction[:,279+k]=1/sd;direction[:,288+k]=hv[:,k]/sd
            with torch.no_grad():fd=(model(x+1e-5*direction,b,v)-model(x-1e-5*direction,b,v))/2e-5*c
            error=abs(fd-deriv[:,axis]);assert torch.all(error<=1e-5+1e-4*abs(deriv[:,axis])),float(error.max());fdmax=max(fdmax,float(error.max()))
        doublemax=max(doublemax,float(abs(y.detach().cpu().numpy()*clear[j:j+1536]-fpred[j:j+1536]).max()))
        predictions.append((y.detach()*c).cpu().numpy());derivatives.append(deriv.detach().cpu().numpy())
    assert doublemax<.001
    np.savez_compressed(a.out/f'seed{seed}_sensitivity.npz',prediction=np.concatenate(predictions),base_GHI=base.cpu().numpy().astype('float64')*clear,truth=data['ghi'],derivative=np.concatenate(derivatives),fit=fit,select=select,available=available,date=np.array([r['target_time_bjt'][:8] for r in rr]),init_time=np.array([r['init_time_bjt'] for r in rr]),target_time=np.array([r['target_time_bjt'] for r in rr]),lead=np.array([int(r['lead_minutes']) for r in rr]),normalized_COT=ff[:,279:288])
    checks.append(dict(seed=seed,step=saved['step'],checkpoint_sha256=sha256(ck),replay_max=replay,finite_difference_max_error=fdmax,double_replay_max=doublemax));print(a.station,checks[-1],flush=True)
(a.out/'COMPLETE.json').write_text(json.dumps(dict(state='PASS_FROZEN_INPUT_SENSITIVITY',station=a.station,rows=len(take),fit_rows=int(fit.sum()),select_rows=int(select.sum()),checks=checks,new_fits=0,upstream_gradients=False,test_payloads_read=0),indent=2))
