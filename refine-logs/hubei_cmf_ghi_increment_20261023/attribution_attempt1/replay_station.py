"""Only C-date values; same G0, ridge and 27 slots, new fixed radiative encoding."""
import argparse,csv,json,sys,time
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"code_attempt2"))
from cmf import attenuation
ROOT=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments')
H=ROOT/'hubei_forecast_cot_increment_20261022';OLD=ROOT/'hubei_cot_linear_increment_20261023/screen_attempt1'
sys.path.insert(0,str(H/'code_attempt1'))
from history_base import HistoryCOTGHIHead,sha256
ap=argparse.ArgumentParser();ap.add_argument('--station',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
assert a.station in ['Niushou','Wangzhai','Mizhuang'] and not a.out.exists()
torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
p=H/'prepared_attempt1'/a.station;meta=json.loads((p/'meta.json').read_text());rows=list(csv.DictReader((p/'rows.csv').open()));roles=np.load(p/'roles.npy')
take=np.flatnonzero(np.isin(roles,['C_fit','C_select']));rr=[rows[i] for i in take];role=roles[take];fit=role=='C_fit';select=role=='C_select'
assert fit.any() and select.any() and not set(r['target_time_bjt'][:8] for r,v in zip(rr,fit) if v)&set(r['target_time_bjt'][:8] for r,v in zip(rr,select) if v)
data={k:np.asarray(np.load(p/(k+'.npy'),mmap_mode='r')[take]) for k in ['future','history','fusion_features','ghi','clear']}
assert all(np.isfinite(v).all() for v in data.values())
leads=np.array([int(r['lead_minutes']) for r in rr],np.float64);ff=data['fusion_features'];assert ff.shape[1]==307
hm=ff[:,207:279].reshape(-1,8,9)[:,-1].astype(bool);bn=meta['base_norm'];hc=ff[:,135:207].reshape(-1,8,9)[:,-1]*bn['history_cot_log1p_std']+bn['history_cot_log1p_mean']
fn=meta['future_cot_norm'];fc=ff[:,279:288]*fn['log1p_std']+fn['log1p_mean'];fv=ff[:,297:306].astype(bool)
assert np.all(fc[fv]>=-1e-6)
cosz=data['future'][:,13].reshape(-1,9).astype(np.float64);assert np.all(abs(cosz)<=1+1e-6)
sza=np.rad2deg(np.arccos(np.clip(cosz,-1,1)))
solar_valid=fv & (cosz>np.cos(np.deg2rad(80.)))
assert np.all(np.expm1(np.maximum(fc[~solar_valid],0))<1e-6)
assert np.all(sza[solar_valid]<80.)
tau=np.expm1(np.maximum(fc.astype(np.float64),0.));phi=np.zeros_like(tau);phi[solar_valid]=attenuation(tau[solar_valid],sza[solar_valid])
vis=data['future'][:,1].reshape(-1,9).astype(np.float64)*bn['future_agri_std'][1]+bn['future_agri_mean'][1]
def features(cot,valid):
    values=cot[fit][valid[fit]];scale=max(float(np.sqrt(np.mean(values.astype(np.float64)**2))),1e-6)
    c=np.where(valid,cot/scale,0.);l=leads/240.;basis=np.stack((np.ones_like(l),l,l*l),1)
    return (c[:,:,None]*basis[:,None,:]).reshape(len(c),27).astype(np.float64),scale
designs={name:features(c,valid) for name,c,valid in [('H',hc,hm),('F',fc,fv),('FS',fc,solar_valid),('M',phi,solar_valid),('V',vis,solar_valid)]}
clear=data['clear'].astype(np.float64);truth=data['ghi'].astype(np.float64);a.out.mkdir(parents=True);receipts=[];started=time.monotonic()
for seed in [42,43,44]:
    ck=H/'training_attempt1/pairs'/f'{a.station}_seed{seed}'/'G0_best.pt';model=HistoryCOTGHIHead().cuda().eval();model.load_state_dict(torch.load(ck,map_location='cuda',weights_only=False)['model']);pred=[]
    with torch.inference_mode():
        for j in range(0,len(take),1536):
            f=torch.from_numpy(np.ascontiguousarray(data['future'][j:j+1536])).cuda();h=torch.from_numpy(np.ascontiguousarray(data['history'][j:j+1536])).cuda()
            c=torch.zeros((len(f),8,1,3,3),device='cuda');pred.append(model(f,h,c).cpu().numpy())
    base=np.concatenate(pred).astype(np.float64)*clear;pred=[base];coefs={};witness={}
    saved_path=Path(__file__).resolve().parent.parent/'screen_attempt2'/a.station/f'seed{seed}_coefficients.json'
    saved=json.loads(saved_path.read_text());assert saved['G0_sha256']==sha256(ck)
    for name in ['H','F','FS','M','V']:
        design,scale=designs[name];valid=hm if name=='H' else (fv if name=='F' else solar_valid)
        assert scale==saved['coefficients'][name]['scale']
        beta=np.array(saved['coefficients'][name]['beta'],np.float64)
        x=design*clear[:,None]/1000.;q=np.maximum(0,base+x@beta*1000)
        assert np.array_equal(q[~valid.any(1)],base[~valid.any(1)])
        pred.append(q)
    pred=np.array(pred)
    prior=Path(__file__).resolve().parent.parent/'screen_attempt2'/a.station/f'seed{seed}_Cselect.npz'
    with np.load(prior) as old:
        maxdiff=float(np.max(abs(pred[:,select]-old['prediction'])))
        assert maxdiff<1e-8,(a.station,seed,maxdiff)
        assert np.array_equal(truth[select],old['truth'])
    np.savez_compressed(a.out/f'seed{seed}_replay.npz',prediction=pred,truth=truth,clear=clear,roles=role,
        date=np.array([r['target_time_bjt'][:8] for r in rr]),init_time=np.array([r['init_time_bjt'] for r in rr]),
        target_time=np.array([r['target_time_bjt'] for r in rr]),lead=leads,arms=['A','H','F','FS','M','V'],
        future_cot_center=tau[:,4],future_cot_valid=solar_valid,history_last_logcot=hc[:,4],sza=sza[:,4])
    receipts.append(dict(seed=seed,checkpoint_sha256=sha256(ck),old_prediction_maxdiff=maxdiff));print(a.station,seed,'DONE',flush=True)
report=dict(state='FROZEN_COEFFICIENT_C_FIT_SELECT_REPLAY_COMPLETE',station=a.station,receipts=receipts,fit_rows=int(fit.sum()),select_rows=int(select.sum()),seconds=time.monotonic()-started,
    official_validation_labels_read=0,new_period_labels_read=0,test_buffer_payloads_read=0,upstream_weights_updated=False,
    source_meta_sha256=sha256(p/'meta.json'),code_sha256=sha256(__file__))
(a.out/'COMPLETE.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
