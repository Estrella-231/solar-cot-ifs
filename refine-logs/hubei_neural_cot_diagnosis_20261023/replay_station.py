import argparse,csv,json,sys,time
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent
H=P.parent/'hubei_forecast_cot_increment_20261022'
sys.path.insert(0,str(H/'code_attempt1'))
from history_base import HistoryCOTGHIHead,sha256
from train import ConditionalHead
ap=argparse.ArgumentParser();ap.add_argument('--station',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
assert a.station in ['Niushou','Wangzhai','Mizhuang'] and not a.out.exists()
torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
p=H/'prepared_attempt1'/a.station;meta=json.loads((p/'meta.json').read_text());rr=list(csv.DictReader((p/'rows.csv').open()));roles=np.load(p/'roles.npy')
take=np.flatnonzero(np.isin(roles,['C_fit','C_select']));rows=[rr[i] for i in take];role=roles[take]
data={k:np.asarray(np.load(p/(k+'.npy'),mmap_mode='r')[take]) for k in ['future','history','cot','ghi','clear','fusion_features','forecast_available']}
assert all(np.isfinite(v).all() for v in data.values())
solar=data['future'][:,13].reshape(-1,9)>np.cos(np.deg2rad(80.))
orig=data['fusion_features'][:,297:306].astype(bool)
fixed_available=(solar&orig).all(1).astype('float32')
assert np.all(fixed_available<=data['forecast_available'])
tensors={k:torch.from_numpy(v).cuda() for k,v in data.items()}
fixed=torch.from_numpy(fixed_available).cuda();clear=data['clear'].astype('float64')
arms=['G1','G2_best','G3_best','G2_last','G3_last','G2_best_solar_gate','G3_best_solar_gate']
a.out.mkdir();receipts=[]
for seed in [42,43,44]:
    folder=H/'training_attempt1/pairs'/f'{a.station}_seed{seed}';old=json.loads((folder/'COMPLETE.json').read_text());checks=[]
    ck=folder/'G1_best.pt';base_model=HistoryCOTGHIHead().cuda().eval();base_model.load_state_dict(torch.load(ck,map_location='cuda',weights_only=False)['model'])
    with torch.inference_mode():
        base=torch.cat([base_model(tensors['future'][j:j+1536],tensors['history'][j:j+1536],tensors['cot'][j:j+1536]) for j in range(0,len(take),1536)])
        features=torch.cat([tensors['fusion_features'],base[:,None]],1);z0=features.clone();lo,hi=meta['cot_feature_slot'];z0[:,lo:hi]=0
        predictions=[base.cpu().numpy().astype('float64')*clear];models={}
        for arm in ['G2_best','G3_best','G2_last','G3_last']:
            file=folder/(arm+'.pt');saved=torch.load(file,map_location='cuda',weights_only=False)
            model=ConditionalHead(features.shape[1]).cuda().eval();model.load_state_dict(saved['model']);models[arm]=model
            features_arm=z0 if arm.startswith('G2') else features
            q=torch.cat([model(features_arm[j:j+1536],base[j:j+1536],tensors['forecast_available'][j:j+1536]) for j in range(0,len(take),1536)])
            predictions.append(q.cpu().numpy().astype('float64')*clear)
            check=dict(arm=arm,epoch=saved['epoch'],updates=saved.get('updates'),sha256=sha256(file))
            if arm.endswith('best'):
                m=role=='C_select';score=float(np.sqrt(np.mean((predictions[-1][m]-data['ghi'][m])**2)))
                expected=old['arms'][arm[:2]]['inner_rmse_wm2'];check.update(replay_rmse=score,saved_inner_rmse=expected,score_abs_difference=abs(score-expected))
                assert abs(score-expected)<1e-4,check
            checks.append(check)
        for arm in ['G2_best','G3_best']:
            model=models[arm];features_arm=z0 if arm.startswith('G2') else features
            q=torch.cat([model(features_arm[j:j+1536],base[j:j+1536],fixed[j:j+1536]) for j in range(0,len(take),1536)])
            assert torch.equal(q[fixed==0],base[fixed==0])
            predictions.append(q.cpu().numpy().astype('float64')*clear)
    pred=np.array(predictions);assert pred.shape==(7,len(take)) and np.isfinite(pred).all()
    np.savez_compressed(a.out/f'seed{seed}_replay.npz',prediction=pred,truth=data['ghi'],clear=clear,roles=role,arms=arms,
        date=np.array([r['target_time_bjt'][:8] for r in rows]),init_time=np.array([r['init_time_bjt'] for r in rows]),target_time=np.array([r['target_time_bjt'] for r in rows]),
        lead=np.array([int(r['lead_minutes']) for r in rows]),source_role=np.array([r['source_role'] for r in rows]),solar_available=fixed_available,original_available=data['forecast_available'])
    receipts.append(dict(seed=seed,base_sha256=sha256(ck),checks=checks));print(a.station,seed,'DONE',flush=True)
(a.out/'COMPLETE.json').write_text(json.dumps(dict(state='FROZEN_NEURAL_C_DATE_REPLAY_COMPLETE',receipts=receipts,new_fits=0,official_val_new_period_test_buffer_payloads_read=0,source_meta_sha256=sha256(p/'meta.json')),indent=2))
