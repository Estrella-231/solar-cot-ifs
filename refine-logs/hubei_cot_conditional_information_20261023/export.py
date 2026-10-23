import csv,hashlib,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;H=P.parent/'hubei_forecast_cot_increment_20261022';G=P.parent/'hubei_cot_calibration_coverage_20261023';R=P.parent/'hubei_matched_R_dynamics_20261023';S=P.parent/'hubei_matched_R_fixed_head_response_20261023'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
out=P/'inputs_attempt2';out.mkdir(exist_ok=False);coverage=json.loads((G/'METADATA_AUDIT.json').read_text());records=[]
with np.load(R/'PAIRED_TRAIN_REFERENCES.npz') as z:ref={k:z[k] for k in z.files if k not in ['ghi','clear']}
rlookup={(str(st),str(i),str(t),int(l),str(role)):j for j,(st,i,t,l,role) in enumerate(zip(ref['station'],ref['init_time'],ref['target_time'],ref['lead'],ref['source_role']))}
for st in ['Niushou','Wangzhai','Mizhuang']:
    p=H/'prepared_attempt1'/st;rows=list(csv.DictReader((p/'rows.csv').open()));meta=json.loads((p/'meta.json').read_text());assert sha(p/'rows.csv')==meta['row_sha256'];roles=np.load(p/'roles.npy');wide=set(coverage[st]['wide_dates'])
    take=np.array([i for i,r in enumerate(rows) if r['split']=='train' and r['target_time_bjt'][:8] in wide]);rr=[rows[i] for i in take];assert len(take)==coverage[st]['wide_rows'] and np.all(roles[take]!='C_select')
    features=np.array(np.load(p/'fusion_features.npy',mmap_mode='r')[take],dtype='float64');future=np.load(p/'future.npy',mmap_mode='r')[take]
    mask=(future[:,13].reshape(-1,9)>np.cos(np.deg2rad(80)))&features[:,297:306].astype(bool);available=mask.all(1)
    cmu=meta['future_cot_norm']['log1p_mean'];csd=meta['future_cot_norm']['log1p_std']
    features[:,279:288]=np.where(mask,features[:,279:288]*csd+cmu,0);features[:,288:297]=np.where(mask,features[:,288:297]*csd,0);features[:,297:306]=mask
    idx=np.array([rlookup[st,r['init_time_bjt'],r['target_time_bjt'],int(r['lead_minutes']),r['source_role']] for r in rr]);assert np.all(ref['scope'][idx]=='C_fit')
    value=ref['observed_R'][idx].astype('float64');assert np.isfinite(value).all() and (value>=0).all()
    history=np.load(p/'cot.npy',mmap_mode='r')[take];norm=meta['base_norm'];last=history[:,-1].reshape(-1,9).astype('float64')*norm['history_cot_log1p_std']+norm['history_cot_log1p_mean'];hv=features[:,270:279].astype(bool)
    observed=np.concatenate([np.where(mask,np.log1p(value),0),np.where(mask&hv,np.log1p(value)-last,0)],1)
    ghi=np.array(np.load(p/'ghi.npy',mmap_mode='r')[take],dtype='float64');clear=np.array(np.load(p/'clear.npy',mmap_mode='r')[take],dtype='float64');bases=[];sourcehash=[]
    keys=[(r['init_time_bjt'],r['target_time_bjt'],int(r['lead_minutes'])) for r in rr]
    assert len(keys)==len(set(keys)) and all(i[:8]==t[:8] for i,t,l in keys)
    for seed in [42,43,44]:
        file=S/'sensitivity_attempt1'/st/f'seed{seed}_sensitivity.npz'
        with np.load(file) as z:d={k:z[k] for k in ['init_time','target_time','lead','truth','base_GHI','fit']}
        lookup={k:i for i,k in enumerate(zip(d['init_time'],d['target_time'],d['lead']))};idx=np.array([lookup[k] for k in keys]);assert np.all(d['fit'][idx]) and np.array_equal(d['truth'][idx],ghi);bases.append(d['base_GHI'][idx]);sourcehash.append(sha(file))
    np.savez_compressed(out/f'{st}.npz',features=features,observed_COT_features=observed,bases=np.array(bases),truth=ghi,clear=clear,available=available,init_time=np.array([k[0] for k in keys]),target_time=np.array([k[1] for k in keys]),lead=np.array([k[2] for k in keys]),source_role=np.array([r['source_role'] for r in rr]))
    records.append(dict(station=st,rows=len(take),dates=len(wide),available=int(available.sum()),row_sha256=meta['row_sha256'],actual_R_pixels_above100=int(((value>100)&mask).sum()),actual_R_max=float(value.max()),same_fold_reference_sha256=sha(R/'PAIRED_TRAIN_REFERENCES.npz'),base_source_sha256=sourcehash,output_sha256=sha(out/f'{st}.npz')))
(out/'COMPLETE.json').write_text(json.dumps(dict(state='CFIT_ONLY_PROBE_INPUTS_READY',stations=records,C_select_rows=0,test_payloads_read=0,new_GPU_runs=0),indent=2));print(json.dumps(records,indent=2))
