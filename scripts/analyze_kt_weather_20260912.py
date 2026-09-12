"""Train-only kt regime clustering and saved-model diagnostics; no test/training."""
import csv,hashlib,json,os
from pathlib import Path
os.environ['OMP_NUM_THREADS']='2'
import numpy as np

ROOT=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907')
OUT=ROOT/'audits/kt_weather_20260912_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metrics(pred,truth,clear,station,mask):
    e=pred.astype('float64')*clear-truth
    stations={}
    for s in (0,1):
        m=mask&(station==s)
        if m.any():stations[str(s)]={'n':int(m.sum()),'rmse':float(np.sqrt(np.mean(e[m]**2))),'mae':float(np.mean(np.abs(e[m]))),'bias':float(e[m].mean())}
    return {'n':int(mask.sum()),'stations':stations,'station_equal_rmse':float(np.mean([x['rmse'] for x in stations.values()])) if len(stations)==2 else None}

def main():
    OUT.mkdir(exist_ok=False)
    pack=ROOT/'data/head_pack_trainval_20260908_v1'
    audit=json.loads((pack/'audit.json').read_text())
    assert audit['state']=='COMPLETE' and audit['test_used'] is False
    data={}
    for k in ('kt','ghi','clear','station','lead','split'):
        spec=audit['arrays'][k];assert sha(pack/spec['name'])==spec['sha256']
        data[k]=np.load(pack/spec['name'],allow_pickle=False)
    assert sha(pack/'rows.csv')==audit['rows_csv_sha256']
    assert np.isin(data['split'],[0,1]).all()
    train=np.flatnonzero(data['split']==0);val=np.flatnonzero(data['split']==1)
    assert len(train)==538053 and len(val)==99849
    assert np.isfinite(data['kt']).all() and (data['kt']>=0).all()
    rows=list(csv.DictReader((pack/'rows.csv').open()))
    assert len(rows)==len(data['kt'])
    unique={};val_unique={}
    for i,r in enumerate(rows):
        key=(r['station'],r['target_time_utc'])
        table=unique if data['split'][i]==0 else val_unique
        if key in table:
            j=table[key]
            assert data['kt'][j]==data['kt'][i] and data['ghi'][j]==data['ghi'][i]
        else:table[key]=i
    assert not (set(unique)&set(val_unique)), 'target leakage across train/validation'
    fit_ids=np.array(list(unique.values()))
    x=np.log1p(data['kt'][fit_ids].astype('float64'))
    centers=np.quantile(x,[1/6,1/2,5/6])
    for iteration in range(200):
        labels=np.argmin(np.abs(x[:,None]-centers),axis=1)
        assert all(np.any(labels==k) for k in range(3))
        new=np.array([x[labels==k].mean() for k in range(3)])
        if np.max(np.abs(new-centers))<1e-10:
            centers=new;break
        centers=new
    else:raise RuntimeError('kmeans did not converge')
    centers=np.sort(centers)
    groups=np.argmin(np.abs(np.log1p(data['kt'])[:,None]-centers),axis=1)
    partitions={};seeds={}
    for k in range(3):
        a=fit_ids[groups[fit_ids]==k];v=val[groups[val]==k]
        partitions[str(k)]={'train_unique_targets':len(a),'val_forecast_rows':len(v),
            'train_kt_quantiles':np.quantile(data['kt'][a],[0,.1,.5,.9,1]).tolist(),
            'val_low_Gcs20_rows':int((data['clear'][v]<20).sum())}
    for cohort,lr in [('original',.0003),('qc1',.001)]:
        seeds[cohort]={}
        for seed in (42,43,44):
            dirname=f'head_ghi_{cohort}_seed42_20260910_v1' if seed==42 else f'head_ghi_multiseed_{cohort}_seed{seed}_20260910_v1'
            arms={}
            for group in ('A','AC'):
                run=ROOT/'runs'/dirname/f'{group}_lr{lr:g}_seed{seed}'
                done=json.loads((run/'complete.json').read_text())
                assert done['test_used'] is False
                predfile=run/'validation_predictions.npz'
                assert sha(predfile)==done['predictions_sha256']
                with np.load(predfile) as z:
                    assert np.array_equal(z['pack_row'],val)
                    for name,key in [('observed_ghi','ghi'),('clear_sky_ghi','clear'),('station','station'),('lead','lead')]:assert np.array_equal(z[name],data[key][val])
                    pred=z['pred_kt'].copy()
                arms[group]={}
                for k in range(3):
                    for band in ('all','Gcs_ge20','Gcs_lt20'):
                        m=groups[val]==k
                        if band=='Gcs_ge20':m &= data['clear'][val]>=20
                        elif band=='Gcs_lt20':m &= data['clear'][val]<20
                        arms[group][f'{k}_{band}']=metrics(pred,data['ghi'][val],data['clear'][val],data['station'][val],m)
            paired={}
            for name in arms['A']:
                a=arms['A'][name]['station_equal_rmse'];c=arms['AC'][name]['station_equal_rmse']
                paired[name]=None if a is None or c is None else c-a
            seeds[cohort][str(seed)]={'metrics':arms,'AC_minus_A':paired}
    aggregate={}
    for cohort in seeds:
        aggregate[cohort]={}
        for name in seeds[cohort]['42']['AC_minus_A']:
            values=[seeds[cohort][str(s)]['AC_minus_A'][name] for s in (42,43,44)]
            aggregate[cohort][name]={'values':values,'mean':float(np.mean(values)) if None not in values else None,
                'std_ddof1':float(np.std(values,ddof=1)) if None not in values else None}
    selection=[]
    for station in ('sili','zhujia'):
        candidates=sorted([i for (st,_),i in val_unique.items() if st==station],key=lambda i:rows[i]['target_time_utc'])
        for pos in np.linspace(0,len(candidates)-1,min(128,len(candidates)),dtype=int):
            i=candidates[pos];r=rows[i]
            selection.append({'pack_row':i,'station':station,'target_time_utc':r['target_time_utc'],'kt':float(data['kt'][i]),'cluster':int(groups[i]),'clear_ghi':float(data['clear'][i])})
    (OUT/'clp_fixed_targets.json').write_text(json.dumps(selection,indent=2))
    np.savez_compressed(OUT/'assignments.npz',pack_row=np.arange(len(groups)),cluster=groups,fit_unique_rows=fit_ids)
    report={'state':'COMPLETE_KT_DIAGNOSTIC_CLP_JOIN_PENDING','test_used':False,'model_training':False,
        'cluster_input':'log1p(observed interval kt); target oracle for diagnostic only',
        'fit':'original train unique station,target_time; k=3 deterministic quantile initialization; no outcome selection',
        'centers_log1p':centers.tolist(),'centers_backtransformed':np.expm1(centers).tolist(),
        'kt_assignment_boundaries':np.expm1((centers[:-1]+centers[1:])/2).tolist(),
        'fit_unique_targets':len(fit_ids),'val_unique_targets':len(val_unique),'iterations':iteration+1,
        'partitions':partitions,'seeds':seeds,'aggregate':aggregate,'source_pack_sha256':sha(pack/'audit.json'),
        'script_sha256':sha(__file__),'CLP':'0 clear/no cloud; 1 water cloud; 2 ice cloud; missing separate',
        'limitations':['not observed weather labels','no observed kt used as deployment input','no test','cluster fit original reused unchanged for QC1','no iid forecast-row significance'],
        'clp_target_selection':'128 per station equally spaced in sorted unique validation targets; not selected by kt or error'}
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({k:report[k] for k in ('state','centers_backtransformed','kt_assignment_boundaries','partitions','aggregate')}))
if __name__=='__main__':main()
