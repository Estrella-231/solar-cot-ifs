"""CLP-stratified paired saved predictions on a fixed validation target subset."""
import csv,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907')
OUT=ROOT/'audits/kt_weather_20260912_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def score(e,station,mask):
    by={}
    for s in (0,1):
        a=e[mask&(station==s)]
        if len(a):by[str(s)]=dict(n=len(a),rmse=float(np.sqrt(np.mean(a*a))),mae=float(np.mean(np.abs(a))),bias=float(a.mean()))
    return dict(n=int(mask.sum()),stations=by,station_equal_rmse=float(np.mean([a['rmse'] for a in by.values()])) if len(by)==2 else None)
def main():
    dest=OUT/'clp_effect.json';assert not dest.exists()
    selected=json.loads((OUT/'clp_fixed_targets.json').read_text())
    source=json.loads((OUT/'clp_join.json').read_text())
    assert source['selection_sha256']==sha(OUT/'clp_fixed_targets.json')
    assert len(source['samples'])==len(selected)==256
    for sample,expected in zip(source['samples'],selected):
        assert all(sample[k]==v for k,v in expected.items())
    lookup={(s['station'],s['target_time_utc']):s for s in source['samples']}
    assert len(lookup)==256
    pack=ROOT/'data/head_pack_trainval_20260908_v1'
    audit=json.loads((pack/'audit.json').read_text());assert audit['test_used'] is False
    assert sha(pack/'rows.csv')==audit['rows_csv_sha256']
    arrays={}
    for key in ('split','station','lead','ghi','clear'):
        spec=audit['arrays'][key];assert sha(pack/spec['name'])==spec['sha256']
        arrays[key]=np.load(pack/spec['name'])
    val=np.flatnonzero(arrays['split']==1);assert len(val)==99849
    rows=list(csv.DictReader((pack/'rows.csv').open()))
    keys=[(rows[i]['station'],rows[i]['target_time_utc']) for i in val]
    classes={name:np.zeros(len(val),dtype=bool) for name in (
        'center_clear','center_cloud','center_water','center_ice','patch_clear_majority',
        'patch_cloud_majority','patch_all_valid_clear','selected_all','clp_unavailable')}
    for pos,key in enumerate(keys):
        s=lookup.get(key)
        if s is None:continue
        classes['selected_all'][pos]=True
        if s.get('state')!='READ':classes['clp_unavailable'][pos]=True;continue
        clp=s.get('center_clp')
        if clp==0:classes['center_clear'][pos]=True
        elif clp in (1,2):
            classes['center_cloud'][pos]=True
            classes['center_water' if clp==1 else 'center_ice'][pos]=True
        else:classes['clp_unavailable'][pos]=True
        counts=s['class_counts'];n=s['valid_pixels']
        if n>=128:
            classes['patch_clear_majority'][pos]=counts[0]>n/2
            classes['patch_cloud_majority'][pos]=counts[1]+counts[2]>n/2
            classes['patch_all_valid_clear'][pos]=n==256 and counts[0]==256
    counts={name:dict(forecast_rows=int(m.sum()),unique_station_targets=len({keys[i] for i in np.flatnonzero(m)}),
        station_rows={str(s):int((m&(arrays['station'][val]==s)).sum()) for s in (0,1)},
        lead_counts={str((l+1)*15):int((m&(arrays['lead'][val]==l)).sum()) for l in range(16)}) for name,m in classes.items()}
    results={}
    for cohort,lr in [('original',.0003),('qc1',.001)]:
        results[cohort]={}
        for seed in (42,43,44):
            dirname=f'head_ghi_{cohort}_seed42_20260910_v1' if seed==42 else f'head_ghi_multiseed_{cohort}_seed{seed}_20260910_v1'
            arms={}
            for group in ('A','AC'):
                run=ROOT/'runs'/dirname/f'{group}_lr{lr:g}_seed{seed}'
                done=json.loads((run/'complete.json').read_text());assert done['test_used'] is False
                f=run/'validation_predictions.npz';assert sha(f)==done['predictions_sha256']
                with np.load(f) as z:
                    assert np.array_equal(z['pack_row'],val)
                    for name,key in [('observed_ghi','ghi'),('clear_sky_ghi','clear'),('station','station'),('lead','lead')]:assert np.array_equal(z[name],arrays[key][val])
                    e=z['pred_kt'].astype('float64')*arrays['clear'][val]-arrays['ghi'][val]
                arms[group]={name:score(e,arrays['station'][val],m) for name,m in classes.items()}
            results[cohort][str(seed)]=arms
    summary={}
    for cohort in results:
        summary[cohort]={}
        for name in classes:
            a=[results[cohort][str(s)]['A'][name]['station_equal_rmse'] for s in (42,43,44)]
            c=[results[cohort][str(s)]['AC'][name]['station_equal_rmse'] for s in (42,43,44)]
            if None in a+c:summary[cohort][name]=dict(state='INSUFFICIENT_STATION_COVERAGE');continue
            d=np.array(c)-np.array(a)
            summary[cohort][name]=dict(A=float(np.mean(a)),AC=float(np.mean(c)),delta_values=d.tolist(),
                delta_mean=float(d.mean()),delta_std=float(d.std(ddof=1)),improved_seeds=int((d<0).sum()),
                relative_reduction_percent=float(100*(np.mean(a)-np.mean(c))/np.mean(a)))
    cross={}
    for s in source['samples']:
        center=s.get('center_clp');label='clear' if center==0 else 'cloud' if center in (1,2) else 'unknown'
        key=f"kt{s['cluster']}:{label}";cross[key]=cross.get(key,0)+1
    report=dict(state='COMPLETE_FIXED_TARGET_CLP_EFFECT',test_used=False,new_training=False,source_sha256=sha(OUT/'clp_join.json'),
        source_status_counts=dict(Counter(s['state'] for s in source['samples'])),counts=counts,summary=summary,results=results,
        cluster_CLP_cross_tab=cross,scope='fixed 256 unique validation targets; all matching forecast rows; same samples in A and AC',
        caveats=['missing CPP not clear sky','center pixel proxy, not whole-day weather','instantaneous CLP versus interval GHI',
        'subset reference-product diagnosis, no causal claim or deployment gating','no iid forecast-row inference'])
    dest.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({k:report[k] for k in ('state','source_status_counts','counts','summary','cluster_CLP_cross_tab')}))
if __name__=='__main__':main()
