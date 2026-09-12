"""Read-only station/weather decomposition of saved A/AC validation predictions."""
import hashlib, json
from pathlib import Path
import numpy as np

ROOT=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907')
OUT=ROOT/'audits/station_weather_cot_diagnosis_20260912.json'
FEATURES=('mean_log1p_cot','std_log1p_cot','p90_log1p_cot','center_log1p_cot')
WEATHER=('sunny','partly_cloudy','overcast')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def stats(a):
    a=np.asarray(a,np.float64);assert len(a) and np.isfinite(a).all()
    return dict(n=len(a),mean=float(a.mean()),std=float(a.std()),
                quantiles=np.quantile(a,[0,.01,.1,.5,.9,.99,1]).tolist())

def rmse(e,m):
    return float(np.sqrt(np.mean(e[m]**2))) if m.any() else None

def summarize_pair(errors,mask):
    values=[];mse=[]
    for ea,ec in errors:
        values.append(rmse(ec,mask)-rmse(ea,mask))
        mse.append(float(np.mean(ec[mask]**2-ea[mask]**2)))
    return dict(n=int(mask.sum()),delta_rmse_values=values,delta_rmse_mean=float(np.mean(values)),
                delta_rmse_std_ddof1=float(np.std(values,ddof=1)),delta_mse_values=mse)

def main():
    assert not OUT.exists()
    pack=ROOT/'data/head_pack_trainval_20260908_v1';audit=json.loads((pack/'audit.json').read_text())
    assert audit['state']=='COMPLETE' and audit['test_used'] is False
    data={}
    for k in ('c','ghi','clear','station','lead','split'):
        spec=audit['arrays'][k];assert sha(pack/spec['name'])==spec['sha256']
        data[k]=np.load(pack/spec['name'],allow_pickle=False)
    val=np.flatnonzero(data['split']==1);assert len(val)==99849
    norm=json.loads((pack/'norm.json').read_text());assert sha(pack/'norm.json')==audit['norm_sha256']
    assert tuple(norm['c_features'])==FEATURES
    raw=data['c'][val].astype(np.float64)*np.asarray(norm['c']['std'],np.float32)+np.asarray(norm['c']['mean'],np.float32)
    assert np.isfinite(raw).all() and (raw>-1e-5).all()

    wrun=ROOT/'runs/weather_coverage_head_seed42_20260912_v2';wdone=json.loads((wrun/'complete.json').read_text())
    wp=wrun/'validation_predictions.npz';assert sha(wp)==wdone['predictions_sha256']
    with np.load(wp,allow_pickle=False) as z:
        assert np.array_equal(z['pack_row'],val)
        predicted=z['predicted_class'].copy();reference=z['reference_class'].copy()
        assert np.array_equal(z['station'],data['station'][val]) and np.array_equal(z['lead'],data['lead'][val])
    assert np.isin(predicted,[0,1,2]).all() and np.isin(reference,[-1,0,1,2]).all()

    pair_errors={}
    for cohort,lr in (('original',.0003),('qc1',.001)):
        pair_errors[cohort]=[]
        for seed in (42,43,44):
            dirname=f'head_ghi_{cohort}_seed42_20260910_v1' if seed==42 else f'head_ghi_multiseed_{cohort}_seed{seed}_20260910_v1'
            errs=[]
            for arm in ('A','AC'):
                run=ROOT/'runs'/dirname/f'{arm}_lr{lr:g}_seed{seed}';done=json.loads((run/'complete.json').read_text())
                predfile=run/'validation_predictions.npz';assert done['test_used'] is False and sha(predfile)==done['predictions_sha256']
                with np.load(predfile,allow_pickle=False) as z:
                    assert np.array_equal(z['pack_row'],val)
                    assert np.array_equal(z['observed_ghi'],data['ghi'][val]) and np.array_equal(z['clear_sky_ghi'],data['clear'][val])
                    errs.append(z['pred_kt'].astype(np.float64)*data['clear'][val]-data['ghi'][val])
            pair_errors[cohort].append(tuple(errs))

    groups={};station_names=('sili','zhujia')
    for s,sname in enumerate(station_names):
        for w,wname in enumerate(WEATHER):
            mask=(data['station'][val]==s)&(predicted==w)
            ref_counts={('unavailable' if k==-1 else WEATHER[k]):int((mask&(reference==k)).sum()) for k in (-1,0,1,2)}
            feature={FEATURES[j]:stats(raw[mask,j]) for j in range(4)}
            lead={}
            for l in range(16):
                lm=mask&(data['lead'][val]==l)
                lead[str((l+1)*15)]={'n':int(lm.sum()),**{c:summarize_pair(pair_errors[c],lm) for c in pair_errors}}
            groups[f'{sname}:{wname}']=dict(n=int(mask.sum()),reference_counts=ref_counts,
                reference_fractions={k:v/int(mask.sum()) for k,v in ref_counts.items()},features_raw_log1p=feature,
                cohorts={c:summarize_pair(pair_errors[c],mask) for c in pair_errors},lead=lead)

    focus=(data['station'][val]==0)&(predicted==0)
    conditional={}
    for k,name in ((0,'reference_sunny'),(1,'reference_partly_cloudy'),(2,'reference_overcast')):
        m=focus&(reference==k)
        conditional[name]={c:summarize_pair(pair_errors[c],m) for c in pair_errors}
    clear=data['clear'][val]
    for lo,hi,name in ((0,20,'clear_ghi_lt20'),(20,200,'clear_ghi_20_200'),(200,600,'clear_ghi_200_600'),(600,np.inf,'clear_ghi_ge600')):
        m=focus&(clear>=lo)&(clear<hi)
        if m.any():conditional[name]={c:summarize_pair(pair_errors[c],m) for c in pair_errors}
    cot_bins={}
    for j,name in enumerate(FEATURES):
        q=np.quantile(raw[focus,j],[0,.25,.5,.75,1]);items=[]
        for b in range(4):
            m=focus&(raw[:,j]>=q[b])&(raw[:,j]<(q[b+1] if b<3 else np.inf))
            items.append(dict(bounds=[float(q[b]),float(q[b+1])],**{c:summarize_pair(pair_errors[c],m) for c in pair_errors}))
        cot_bins[name]=items
    concentration={}
    for cohort in pair_errors:
        concentration[cohort]={}
        ids=np.flatnonzero(focus)
        for seed,(ea,ec) in zip((42,43,44),pair_errors[cohort]):
            diff=ec[ids]**2-ea[ids]**2;order=np.argsort(diff)[::-1];total=diff.sum()
            concentration[cohort][str(seed)]=dict(delta_mse=float(diff.mean()),positive_fraction=float((diff>0).mean()),
                worst_1pct_share_of_signed_sum=float(diff[order[:max(1,len(diff)//100)]].sum()/total) if total else None,
                worst_5pct_share_of_signed_sum=float(diff[order[:max(1,len(diff)//20)]].sum()/total) if total else None)

    report=dict(state='COMPLETE_STATION_WEATHER_COT_DIAGNOSIS',test_used=False,new_training=False,
        source_pack_sha256=sha(pack/'audit.json'),weather_predictions_sha256=sha(wp),feature_names=FEATURES,
        definitions=dict(weather='predicted cloud-coverage class from causal forecast AGRI',feature='raw statistics of predicted log1p(COT), not physical-COT moments',delta='AC minus A; negative RMSE delta improves'),
        groups=groups,sili_predicted_sunny_conditionals=conditional,sili_predicted_sunny_cot_quartiles=cot_bins,
        sili_predicted_sunny_error_concentration=concentration,
        limitations=['saved validation predictions only','weather head has one seed','subgroups are diagnostic and not selected models','CPP source-production binding remains unresolved'])
    OUT.write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(dict(state=report['state'],focus_n=int(focus.sum()))))
if __name__=='__main__':main()
