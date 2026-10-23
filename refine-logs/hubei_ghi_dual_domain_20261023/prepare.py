"""Actual target training views only; no actual C_select tensors are constructed."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;H=P.parent/'hubei_forecast_cot_increment_20261022';G=P.parent/'hubei_cot_calibration_coverage_20261023';R=P.parent/'hubei_matched_R_dynamics_20261023'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rawroot=R/'raw_attempt1';rawmeta=json.loads((rawroot/'COMPLETE.json').read_text());assert sha(rawroot/'raw16.npy')==rawmeta['raw_sha256']
frames=list(csv.DictReader((rawroot/'frames.csv').open()));lookup={r['bjt']:r for r in frames};raw=np.load(rawroot/'raw16.npy',mmap_mode='r')
with np.load(R/'PAIRED_TRAIN_REFERENCES.npz') as z:ref={k:z[k] for k in z.files if k not in ['ghi','clear']}
rlookup={(str(s),str(i),str(t),int(l),str(role)):j for j,(s,i,t,l,role) in enumerate(zip(ref['station'],ref['init_time'],ref['target_time'],ref['lead'],ref['source_role']))}
coverage=json.loads((G/'METADATA_AUDIT.json').read_text());out=P/'prepared_attempt1';out.mkdir(exist_ok=False);receipts=[]
for st in ['Niushou','Wangzhai','Mizhuang']:
    p=H/'prepared_attempt1'/st;meta=json.loads((p/'meta.json').read_text());assert sha(p/'rows.csv')==meta['row_sha256'];rows=list(csv.DictReader((p/'rows.csv').open()));roles=np.load(p/'roles.npy');wide=set(coverage[st]['wide_dates'])
    take=np.array([i for i,r in enumerate(rows) if r['split']=='train' and r['target_time_bjt'][:8] in wide]);assert len(take)==coverage[st]['wide_rows'] and np.all(roles[take]!='C_select')
    future=np.array(np.load(p/'future.npy',mmap_mode='r')[take]);cot=np.load(p/'cot.npy',mmap_mode='r')[take];ff=np.load(p/'fusion_features.npy',mmap_mode='r')[take]
    rr=[rows[i] for i in take];idx=np.array([rlookup[st,r['init_time_bjt'],r['target_time_bjt'],int(r['lead_minutes']),r['source_role']] for r in rr]);assert np.all(ref['scope'][idx]=='C_fit')
    cy,cx=meta['base_norm']['station_center_rc_in_patch'];actual=[]
    for r in rr:
        frame=lookup[r['target_time_bjt']];assert frame['split']=='train';x=np.array(raw[int(frame['frame_index'])]);assert hashlib.sha256(x.tobytes()).hexdigest()==frame['patch_sha256'];actual.append(x[:,cy-1:cy+2,cx-1:cx+2])
    actual=np.stack(actual);geom=float(abs(actual[:,13:15]-future[:,13:15]).max());assert geom<=1e-6
    norm=meta['base_norm'];mu=np.array(norm['future_agri_mean'],np.float32)[None,:,None,None];sd=np.array(norm['future_agri_std'],np.float32)[None,:,None,None]
    future[:,:13]=(actual[:,:13]-mu)/sd;value=ref['observed_R'][idx].astype('float64');valid=(actual[:,15].reshape(-1,9)>.5)&(actual[:,13].reshape(-1,9)>np.cos(np.deg2rad(80)))&np.isfinite(value)&(value>=0)&(value<=100)
    log=np.log1p(np.where(valid,value,0));cmu=meta['future_cot_norm']['log1p_mean'];csd=meta['future_cot_norm']['log1p_std'];last=cot[:,-1].reshape(-1,9).astype('float64')*norm['history_cot_log1p_std']+norm['history_cot_log1p_mean'];hv=ff[:,270:279].astype(bool)
    feature=np.concatenate([np.where(valid,(log-cmu)/csd,0),np.where(valid&hv,(log-last)/csd,0),valid],1).astype('float32')
    assert np.isfinite(future).all() and np.isfinite(feature).all()
    dest=out/st;dest.mkdir();np.savez_compressed(dest/'actual_training_views.npz',future=future,cot_features=feature,prepared_index=take,available=valid.all(1).astype('float32'),target_time=np.array([r['target_time_bjt'] for r in rr]),init_time=np.array([r['init_time_bjt'] for r in rr]),lead=np.array([int(r['lead_minutes']) for r in rr]))
    receipts.append(dict(station=st,fit_rows=len(take),fit_dates=len(wide),C_select_views=0,geometry_max=geom,rows_sha256=meta['row_sha256'],view_sha256=sha(dest/'actual_training_views.npz')))
(out/'COMPLETE.json').write_text(json.dumps(dict(state='ACTUAL_TRAINING_VIEWS_READY',stations=receipts,raw_sha256=rawmeta['raw_sha256'],same_fold_reference_sha256=sha(R/'PAIRED_TRAIN_REFERENCES.npz'),new_fits=0,test_payloads_read=0),indent=2));print(json.dumps(receipts,indent=2))
