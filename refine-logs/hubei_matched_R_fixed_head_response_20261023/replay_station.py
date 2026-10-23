"""Frozen current large GHI heads; same-fold COT source substitution only."""
import argparse,csv,hashlib,json,sys
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent;ROOT=P.parent;H=ROOT/'hubei_forecast_cot_increment_20261022';G=ROOT/'hubei_cot_calibration_coverage_20261023';R=ROOT/'hubei_matched_R_dynamics_20261023'
sys.path.insert(0,str(H/'code_attempt1'))
from train import ConditionalHead
from history_base import HistoryCOTGHIHead,sha256
ap=argparse.ArgumentParser();ap.add_argument('--station',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assert a.station in ['Niushou','Wangzhai','Mizhuang']
torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
p=H/'prepared_attempt1'/a.station;meta=json.loads((p/'meta.json').read_text());rows=list(csv.DictReader((p/'rows.csv').open()));roles=np.load(p/'roles.npy');take=np.flatnonzero(roles=='C_select');rr=[rows[i] for i in take]
assert all(r['split']=='train' for r in rr) and sha256(p/'rows.csv')==meta['row_sha256']
keys=[(r['init_time_bjt'],r['target_time_bjt'],int(r['lead_minutes']),r['source_role']) for r in rr];assert len(keys)==len(set(keys))
for r in rr:assert datetime.strptime(r['target_time_bjt'],'%Y%m%d%H%M')==datetime.strptime(r['init_time_bjt'],'%Y%m%d%H%M')+timedelta(minutes=int(r['lead_minutes']))
data={k:np.array(np.load(p/(k+'.npy'),mmap_mode='r')[take]) for k in ['future','history','cot','fusion_features','ghi','clear']};assert all(np.isfinite(v).all() for v in data.values())
with np.load(R/'PAIRED_TRAIN_REFERENCES.npz') as z:ref={k:z[k] for k in z.files if k!='ghi' and k!='clear'}
lookup={(str(st),str(i),str(t),int(l),str(role)):j for j,(st,i,t,l,role) in enumerate(zip(ref['station'],ref['init_time'],ref['target_time'],ref['lead'],ref['source_role']))};assert len(lookup)==len(ref['lead'])
idx=np.array([lookup.get((a.station,)+key,-1) for key in keys]);found=idx>=0;assert found.all(),'Existing same-fold reference should cover complete C_select metadata'
physical={name:ref[key][idx].astype('float64') for name,key in [('forecast','forecast'),('persistence','persistence'),('observed_R','observed_R')]}
mu=meta['future_cot_norm']['log1p_mean'];sd=meta['future_cot_norm']['log1p_std'];norm=meta['base_norm']
ff=data['fusion_features'].copy();orig_valid=ff[:,297:306].astype(bool);solar=data['future'][:,13].reshape(-1,9)>np.cos(np.deg2rad(80.));original_mask=orig_valid&solar;joint=original_mask&ref['valid'][idx]
# Verify physical original-COT values independently of how replacement features are made.
reconstructed=np.expm1(data['fusion_features'][:,279:288].astype('float64')*sd+mu);raw_diff=float(abs(reconstructed[original_mask]-physical['forecast'][original_mask]).max());assert raw_diff<1e-4,raw_diff
for lo in [279,288]:ff[:,lo:lo+9]=np.where(original_mask,ff[:,lo:lo+9],0)
ff[:,297:306]=original_mask
latest=data['cot'][:,-1].reshape(-1,9).astype('float64')*norm['history_cot_log1p_std']+norm['history_cot_log1p_mean'];last_hist_valid=data['fusion_features'][:,270:279].astype(bool)
def replacement(value):
    q=ff.copy();log=np.log1p(value)
    q[:,279:288]=np.where(joint,(log-mu)/sd,0)
    q[:,288:297]=np.where(joint&last_hist_valid,(log-latest)/sd,0)
    q[:,297:306]=joint
    assert np.array_equal(q[:,:279],ff[:,:279]) and np.array_equal(q[:,306:],ff[:,306:]);return q
branches={name:replacement(v) for name,v in physical.items()}
common=joint.all(1).astype('float32');original_available=original_mask.all(1).astype('float32');assert np.all(common<=original_available)
t={k:torch.from_numpy(data[k]).cuda() for k in ['future','history','cot']};clear=data['clear'].astype('float64');tclear=torch.from_numpy(clear).cuda();ca=torch.from_numpy(common).cuda();oa=torch.from_numpy(original_available).cuda();a.out.mkdir(exist_ok=False)
arms=['G1','no_future_original','future_original','no_future_joint','forecast_joint','persistence_joint','observed_R_joint_OFFLINE'];receipts=[]
for seed in [42,43,44]:
    ck=H/'training_attempt1/pairs'/f'{a.station}_seed{seed}'/'G1_best.pt';base_model=HistoryCOTGHIHead().cuda().eval().requires_grad_(False);base_model.load_state_dict(torch.load(ck,map_location='cuda',weights_only=False)['model'])
    with torch.inference_mode():base=torch.cat([base_model(t['future'][j:j+1536],t['history'][j:j+1536],t['cot'][j:j+1536]) for j in range(0,len(take),1536)])
    def features(v):return torch.cat([torch.from_numpy(v).cuda(),base[:,None]],1)
    orig=features(ff);no_orig=orig.clone();no_orig[:,279:297]=0
    fs={k:features(v) for k,v in branches.items()};no_joint=fs['forecast'].clone();no_joint[:,279:297]=0
    models={};sources=[]
    for arm in ['large_no_future_COT','large_future_COT']:
        file=G/'replay'/a.station/f'seed{seed}_{arm}_best.pt';saved=torch.load(file,map_location='cuda',weights_only=False);assert saved['base_sha256']==sha256(ck)
        model=ConditionalHead(308).cuda().eval().requires_grad_(False);model.load_state_dict(saved['model']);models[arm]=model;sources.append(dict(arm=arm,step=saved['step'],sha256=sha256(file)))
    def forward(model,f,available):
        with torch.inference_mode():q=torch.cat([model(f[j:j+1536],base[j:j+1536],available[j:j+1536]) for j in range(0,len(take),1536)])
        assert torch.equal(q[available==0],base[available==0]);return (q.double()*tclear).cpu().numpy()
    no=models['large_no_future_COT'];yes=models['large_future_COT'];pred=[base.cpu().numpy().astype('float64')*clear,forward(no,no_orig,oa),forward(yes,orig,oa),forward(no,no_joint,ca),forward(yes,fs['forecast'],ca),forward(yes,fs['persistence'],ca),forward(yes,fs['observed_R'],ca)]
    joint_reconstruction_difference=float(abs(pred[4][common>0]-pred[2][common>0]).max());assert joint_reconstruction_difference<=.001,joint_reconstruction_difference
    with np.load(G/'replay'/a.station/f'seed{seed}_Cselect.npz') as saved:
        assert list(zip(saved['init_time'],saved['target_time'],saved['lead']))==[(i,t,l) for i,t,l,_ in keys]
        assert np.array_equal(saved['truth'],data['ghi']) and np.array_equal(saved['clear'],data['clear'])
        differences=[float(abs(pred[j]-saved['prediction'][old]).max()) for j,old in [(0,0),(1,3),(2,4)]];assert max(differences)<=.001,differences
    np.savez_compressed(a.out/f'seed{seed}_response.npz',prediction=np.array(pred),truth=data['ghi'],clear=clear,arms=arms,init_time=np.array([k[0] for k in keys]),target_time=np.array([k[1] for k in keys]),lead=np.array([k[2] for k in keys]),source_role=np.array([k[3] for k in keys]),date=np.array([k[1][:8] for k in keys]),joint_available=common,original_available=original_available)
    receipts.append(dict(seed=seed,base_sha256=sha256(ck),heads=sources,saved_prediction_max_differences=differences,joint_forecast_feature_reconstruction_GHI_difference=joint_reconstruction_difference));print(a.station,seed,differences,flush=True)
np.savez_compressed(a.out/'physical_COT_sources.npz',**physical,joint_mask=joint,common_rows=common,found=found,prepared_index=take)
report=dict(state='CURRENT_FIXED_HEAD_SAME_FOLD_COT_RESPONSE_COMPLETE',station=a.station,rows=len(take),reference_found=int(found.sum()),joint_available_rows=int(common.sum()),original_available_rows=int(original_available.sum()),raw_physical_reconstruction_max_difference=raw_diff,receipts=receipts,source_meta_sha256=sha256(p/'meta.json'),matched_reference_sha256=sha256(R/'PAIRED_TRAIN_REFERENCES.npz'),same_fold_reference_receipts_sha256=sha256(R/'reference_attempt1/COMPLETE.json'),trainer_sources={n:sha256(H/'code_attempt1'/n) for n in ['train.py','history_base.py']},upstream_training=False,new_fits=0,official_val_new_period_outerbuffer_test_payloads_read=0,claim='Observed target R replacement is offline, not deployable forecast or independent truth')
(a.out/'COMPLETE.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ['state','station','rows','joint_available_rows']}),flush=True)
