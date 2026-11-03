from pathlib import Path
import numpy as np,pandas as pd,json,hashlib
BASE=Path(__file__).resolve().parents[2]
SRC=BASE/'refine-logs/hunan_dydiff_20261028/hunan_full_validation_20261030/recovered/evaluation_attempt1'
NATIVE=BASE/'refine-logs/cloud_probability_input_20261020/results/four_arm_paired_v1_attempt1'
OUT=Path(__file__).resolve().parent
rows=pd.read_csv(SRC/'ghi_rows.csv');z=np.load(SRC/'ghi_predictions.npz',allow_pickle=False)
y=z['truth'].astype(np.float64);assert np.array_equal(y,rows.ghi.to_numpy())
assert not rows.duplicated(['station','init_time_utc','target_time_utc']).any()
assert not ((rows.station=='sili')&(rows.target_date_bjt=='2024-09-05')).any()
# No import of the original scorer. Fixed BJT date-block sampling, all six station/seed groups retained.
def score(groups):
    dates=sorted(set().union(*(set(q['date']) for q in groups)))
    w=np.random.default_rng(20261028).multinomial(len(dates),np.full(len(dates),1/len(dates)),size=5000).astype(float)
    points=[];bs=[];aa=[]
    for q in groups:
        di=pd.Categorical(q['date'],categories=dates).codes
        b=q['before']-q['truth'];a=q['after']-q['truth']
        assert np.isfinite(b).all() and np.isfinite(a).all()
        n=np.bincount(di,minlength=len(dates));den=w@n;assert (den>0).all()
        bs.append(np.sqrt((w@np.bincount(di,weights=b*b,minlength=len(dates)))/den))
        aa.append(np.sqrt((w@np.bincount(di,weights=a*a,minlength=len(dates)))/den))
        br=float(np.sqrt(np.mean(b*b)));ar=float(np.sqrt(np.mean(a*a)))
        points.append({'station':q['station'],'seed':q['seed'],'n':len(b),'before':br,'after':ar,'delta':ar-br})
    br=np.mean([q['before'] for q in points]);ar=np.mean([q['after'] for q in points])
    by={}
    for st in ('sili','zhujia'):
        cell=[i for i,q in enumerate(points) if q['station']==st]
        b=np.mean([points[i]['before'] for i in cell]);a=np.mean([points[i]['after'] for i in cell])
        by[st]={'before':b,'after':a,'gain_percent':100*(b-a)/b,'delta_ci95':np.quantile(np.mean([aa[i]-bs[i] for i in cell],axis=0),[.025,.975]).tolist()}
    return {'before':br,'after':ar,'gain_percent':100*(br-ar)/br,'delta':ar-br,'delta_ci95':np.quantile(np.mean(aa,axis=0)-np.mean(bs,axis=0),[.025,.975]).tolist(),'unique_dates':len(dates),'by_station':by,'per_run':points}
g=[];u=[];spreads=[];matches=[]
for st in ('sili','zhujia'):
    ix=np.flatnonzero(rows.station.eq(st));r=rows.iloc[ix]
    for seed in (42,43,44):
        b=z[f'{st}_seed{seed}_M00_observed'][ix];a=z[f'{st}_seed{seed}_M10_observed'][ix]
        g.append({'station':st,'seed':seed,'before':b,'after':a,'truth':y[ix],'date':r.target_date_bjt.to_numpy()})
        f=r[['pack_index','target_date_bjt']].copy();f['b']=b;f['a']=a;f['y']=y[ix]
        agg=f.groupby('pack_index',sort=True).agg({'b':'mean','a':'mean','y':'mean','target_date_bjt':'first'})
        ranges=f.groupby('pack_index')[['b','a','y']].agg(lambda x:x.max()-x.min()).max()
        spreads.append({'station':st,'seed':seed,'max_repeated_target_spread':ranges.to_dict()})
        u.append({'station':st,'seed':seed,'before':agg.b.to_numpy(),'after':agg.a.to_numpy(),'truth':agg.y.to_numpy(),'date':agg.target_date_bjt.to_numpy()})
        for arm,col in [('M00','b'),('M10','a')]:
            n=np.load(NATIVE/f'seed{seed}/{st}/{arm}/validation_predictions.npz')
            lookup=dict(zip(n['pack_index'],n['prediction']));e=np.array([lookup[k] for k in agg.index])
            matches.append({'station':st,'seed':seed,'arm':arm,'max_abs_replay_difference':float(abs(agg[col].to_numpy()-e).max())})
result={'forecast_weighted_observed':score(g),'unique_target_observed_diagnostic':score(u),'replay_spreads':spreads,'native_saved_comparison':matches}
vr=pd.read_csv(NATIVE/'validation_rows.csv');ng=[]
for st in ('sili','zhujia'):
    for seed in (42,43,44):
        b=np.load(NATIVE/f'seed{seed}/{st}/M00/validation_predictions.npz');a=np.load(NATIVE/f'seed{seed}/{st}/M10/validation_predictions.npz')
        assert np.array_equal(b['pack_index'],a['pack_index']) and np.array_equal(b['truth'],a['truth'])
        ix=vr.set_index('pack_index') if 'pack_index' in vr.columns else None
        # pack_index is the original combined pack row; map via aligned validation_inputs instead.
        with np.load(NATIVE/'validation_inputs.npz') as inp: vi=inp['pack_index'].astype(int)
        assert len(vi)==len(vr)
        lookup=dict(zip(vi,vr.target_time_utc));dates=np.array([(pd.Timestamp(lookup[int(k)])+pd.Timedelta(hours=8)).strftime('%Y-%m-%d') for k in b['pack_index']])
        assert not ((st=='sili')&(dates=='2024-09-05')).any()
        ng.append({'station':st,'seed':seed,'before':b['prediction'].astype(float),'after':a['prediction'].astype(float),'truth':b['truth'].astype(float),'date':dates})
result['native_full_validation_observed']=score(ng)
ref=json.loads((SRC/'ghi_scores_fp64.json').read_text())['comparisons']['observed_add_COT']
calc=result['forecast_weighted_observed'];assert max(abs(calc['before']-ref['before_macro_rmse_Wm2']),abs(calc['after']-ref['after_macro_rmse_Wm2']),np.max(abs(np.array(calc['delta_ci95'])-ref['date_block_95CI'])))<1e-10
completion=json.loads((SRC/'completion.json').read_text());hash_checks=[]
for key,expected in completion['head_sha256'].items():
    st,seed,arm=key.split('/');f=NATIVE/f'seed{seed}/{st}/{arm}/best.pt'
    if f.exists():hash_checks.append({'head':key,'matches':hashlib.sha256(f.read_bytes()).hexdigest()==expected})
assert all(q['matches'] for q in hash_checks)
result.update({'status':'PASS','rows_repeated':len(rows),'unique_station_targets':rows[['station','target_time_utc']].drop_duplicates().shape[0],'native_rows':len(vr),'heads_hash_checked':hash_checks,'source_hashes':{str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SRC/'ghi_predictions.npz',SRC/'ghi_rows.csv',SRC/'completion.json']},'bootstrap':'5000 shared BJT target-date draws, RNG 20261028, fixed station/seed groups, all denominators positive','test_payloads_read':0,'no_new_model_inference':True})
(OUT/'RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k in ['rows_repeated','unique_station_targets','native_rows','forecast_weighted_observed','unique_target_observed_diagnostic','native_full_validation_observed','native_saved_comparison']},ensure_ascii=False,indent=2))
