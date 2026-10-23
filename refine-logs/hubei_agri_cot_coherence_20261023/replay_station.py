import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent;H=P.parent/'hubei_forecast_cot_increment_20261022';G=P.parent/'hubei_cot_calibration_coverage_20261023';OLD=P.parent/'hubei_matched_R_fixed_head_response_20261023';R=P.parent/'hubei_matched_R_dynamics_20261023'
sys.path.insert(0,str(H/'code_attempt1'))
from train import ConditionalHead
from history_base import HistoryCOTGHIHead,sha256
ap=argparse.ArgumentParser();ap.add_argument('--station',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
p=H/'prepared_attempt1'/a.station;meta=json.loads((p/'meta.json').read_text());rows=list(csv.DictReader((p/'rows.csv').open()));roles=np.load(p/'roles.npy');take=np.flatnonzero(roles=='C_select');rr=[rows[i] for i in take]
assert sha256(p/'rows.csv')==meta['row_sha256'] and all(r['split']=='train' for r in rr)
data={k:np.array(np.load(p/(k+'.npy'),mmap_mode='r')[take]) for k in ['future','history','cot','fusion_features','ghi','clear']}
rawroot=R/'raw_attempt1';receipt=json.loads((rawroot/'COMPLETE.json').read_text());assert sha256(rawroot/'raw16.npy')==receipt['raw_sha256']
frames=list(csv.DictReader((rawroot/'frames.csv').open()));lookup={r['bjt']:r for r in frames};raw=np.load(rawroot/'raw16.npy',mmap_mode='r')
cy,cx=meta['base_norm']['station_center_rc_in_patch'];observed=[]
for r in rr:
    f=lookup[r['target_time_bjt']];assert f['split']=='train';scene=np.array(raw[int(f['frame_index'])]);assert hashlib.sha256(scene.tobytes()).hexdigest()==f['patch_sha256'];observed.append(scene[:,cy-1:cy+2,cx-1:cx+2])
observed=np.stack(observed);geometry_difference=float(abs(observed[:,13:15]-data['future'][:,13:15]).max());assert geometry_difference<=1e-6
mu=np.array(meta['base_norm']['future_agri_mean'],np.float32)[None,:,None,None];sd=np.array(meta['base_norm']['future_agri_std'],np.float32)[None,:,None,None]
actual=data['future'].copy();actual[:,:13]=(observed[:,:13]-mu)/sd;assert np.array_equal(actual[:,13:],data['future'][:,13:])
with np.load(OLD/'replay'/a.station/'physical_COT_sources.npz') as z:source={k:z[k] for k in z.files}
assert np.array_equal(source['prepared_index'],take);joint=source['joint_mask'];available=source['common_rows'];assert np.array_equal(joint.all(1),available.astype(bool))
ff=data['fusion_features'].copy();logmu=meta['future_cot_norm']['log1p_mean'];logsd=meta['future_cot_norm']['log1p_std'];norm=meta['base_norm'];last=data['cot'][:,-1].reshape(-1,9).astype('float64')*norm['history_cot_log1p_std']+norm['history_cot_log1p_mean'];hv=ff[:,270:279].astype(bool)
def build_features(future,value,base):
    z=ff.copy();z[:,:135]=future.reshape(len(rr),-1);log=np.log1p(value)
    z[:,279:288]=np.where(joint,(log-logmu)/logsd,0);z[:,288:297]=np.where(joint&hv,(log-last)/logsd,0);z[:,297:306]=joint
    assert np.array_equal(z[:,135:279],ff[:,135:279]) and np.array_equal(z[:,306:],ff[:,306:])
    return torch.cat([torch.from_numpy(z).cuda(),base[:,None]],1)
arms=['forecast_G1','forecast_no_COT','forecast_forecast_COT','forecast_observed_COT_OFFLINE','observed_G1_OFFLINE','observed_no_COT_OFFLINE','observed_forecast_COT_OFFLINE','observed_observed_COT_OFFLINE']
t={k:torch.from_numpy(data[k]).cuda() for k in ['future','history','cot']};ta=torch.from_numpy(actual).cuda();av=torch.from_numpy(available).cuda();clear=data['clear'].astype('float64');a.out.mkdir(exist_ok=False);checks=[]
for seed in [42,43,44]:
    ck=H/'training_attempt1/pairs'/f'{a.station}_seed{seed}'/'G1_best.pt';base_model=HistoryCOTGHIHead().cuda().eval().requires_grad_(False);base_model.load_state_dict(torch.load(ck,map_location='cuda',weights_only=False)['model'])
    def base_forward(future):
        with torch.no_grad():return torch.cat([base_model(future[j:j+1536],t['history'][j:j+1536],t['cot'][j:j+1536]) for j in range(0,len(rr),1536)])
    bases=[base_forward(t['future']),base_forward(ta)];models={};hashes={}
    for name in ['large_no_future_COT','large_future_COT']:
        file=G/'replay'/a.station/f'seed{seed}_{name}_best.pt';saved=torch.load(file,map_location='cuda',weights_only=False);assert saved['base_sha256']==sha256(ck)
        model=ConditionalHead(308).cuda().eval().requires_grad_(False);model.load_state_dict(saved['model']);models[name]=model;hashes[name]=sha256(file)
    predictions=[]
    for future,base in zip([data['future'],actual],bases):
        predictions.append(base.double().cpu().numpy()*clear)
        zpred=build_features(future,source['forecast'],base);zreal=build_features(future,source['observed_R'],base);zero=zpred.clone();zero[:,279:297]=0
        for model,x in [(models['large_no_future_COT'],zero),(models['large_future_COT'],zpred),(models['large_future_COT'],zreal)]:
            with torch.no_grad():q=torch.cat([model(x[j:j+1536],base[j:j+1536],av[j:j+1536]) for j in range(0,len(rr),1536)])
            assert torch.equal(q[av==0],base[av==0]);predictions.append(q.double().cpu().numpy()*clear)
    with np.load(OLD/'replay'/a.station/f'seed{seed}_response.npz') as z:
        assert np.array_equal(z['truth'],data['ghi']) and z['target_time'].tolist()==[r['target_time_bjt'] for r in rr]
        differences=[float(abs(predictions[i]-z['prediction'][k]).max()) for i,k in [(0,0),(1,3),(2,4),(3,6)]];assert max(differences)<=.001,differences
    np.savez_compressed(a.out/f'seed{seed}_factorial.npz',prediction=np.array(predictions),truth=data['ghi'],clear=clear,arms=arms,init_time=np.array([r['init_time_bjt'] for r in rr]),target_time=np.array([r['target_time_bjt'] for r in rr]),date=np.array([r['target_time_bjt'][:8] for r in rr]),lead=np.array([int(r['lead_minutes']) for r in rr]),common=available)
    checks.append(dict(seed=seed,base_sha256=sha256(ck),head_sha256=hashes,replay_max_differences=differences));print(a.station,seed,differences,flush=True)
(a.out/'COMPLETE.json').write_text(json.dumps(dict(state='FROZEN_AGRI_COT_FACTORIAL_COMPLETE',station=a.station,rows=len(rr),common=int(available.sum()),geometry_max=geometry_difference,checks=checks,raw_sha256=receipt['raw_sha256'],new_fits=0,test_payloads_read=0,usage='All observed-future branches are offline only, not forecast scores'),indent=2))
