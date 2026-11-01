"""Recompute GHI comparisons from saved arrays without importing the scorer."""
from pathlib import Path
import csv, hashlib, json
import numpy as np

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'recovered/evaluation_attempt1'
with (DATA/'ghi_rows.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
with np.load(DATA/'ghi_predictions.npz',allow_pickle=False) as z:g={k:np.asarray(z[k],np.float64) for k in z.files}
assert len(rows)==63480 and len({(r['station'],r['init_time_utc'],r['target_time_utc']) for r in rows})==63480
y=g['truth'];assert np.allclose(y,[float(r['ghi']) for r in rows],rtol=0,atol=0)
assert np.allclose(g['clear'],[float(r['clear']) for r in rows],rtol=1e-7,atol=1e-5)
station=np.array([r['station'] for r in rows]);date=np.array([r['target_date_bjt'] for r in rows]);lead=np.array([int(r['lead_minutes']) for r in rows])
stored=json.loads((DATA/'ghi_scores_fp64.json').read_text())

def calc(before,after,allowed):
    dates=np.unique(date[allowed]);position={v:i for i,v in enumerate(dates)}
    before_rmse=[];after_rmse=[];counts=[];b_sse=[];a_sse=[];by={}
    for s in ('sili','zhujia'):
        per=[]
        ids=np.flatnonzero(allowed&(station==s));di=np.array([position[v] for v in date[ids]])
        for seed in (42,43,44):
            b=g[f'{s}_seed{seed}_{before}'][ids];a=g[f'{s}_seed{seed}_{after}'][ids]
            assert np.isfinite(b).all() and np.isfinite(a).all()
            be=b-y[ids];ae=a-y[ids];br=float(np.sqrt(np.mean(be**2)));ar=float(np.sqrt(np.mean(ae**2)))
            before_rmse.append(br);after_rmse.append(ar)
            counts.append(np.bincount(di,minlength=len(dates)))
            b_sse.append(np.bincount(di,weights=be**2,minlength=len(dates)))
            a_sse.append(np.bincount(di,weights=ae**2,minlength=len(dates)))
            per.append({'seed':seed,'before_rmse':br,'after_rmse':ar,'delta_rmse':ar-br,
                        'before_mae':float(np.abs(be).mean()),'after_mae':float(np.abs(ae).mean()),
                        'before_bias':float(be.mean()),'after_bias':float(ae.mean())})
        bb=np.mean([v['before_rmse'] for v in per]);aa=np.mean([v['after_rmse'] for v in per])
        by[s]={'before_rmse':float(bb),'after_rmse':float(aa),'relative_reduction_pct':float(100*(1-aa/bb)),
               'seed_deltas':[v['delta_rmse'] for v in per],'runs':per}
    w=np.random.default_rng(20261028).multinomial(len(dates),np.full(len(dates),1/len(dates)),size=5000).astype(np.float64)
    denom=w@np.array(counts,dtype=np.float64).T
    bs=np.sqrt((w@np.array(b_sse).T)/np.maximum(denom,1))
    ass=np.sqrt((w@np.array(a_sse).T)/np.maximum(denom,1))
    ci=np.quantile(ass.mean(1)-bs.mean(1),[.025,.975])
    b=float(np.mean(before_rmse));a=float(np.mean(after_rmse))
    return {'before_macro_rmse_Wm2':b,'after_macro_rmse_Wm2':a,
            'delta_rmse_after_minus_before':a-b,'relative_rmse_reduction_pct':100*(1-a/b),
            'date_block_95CI':ci.tolist(),'by_station':by,'rows':int(allowed.sum()),'target_dates':len(dates)}

pairs={
 'dydiff_add_COT':('M00_dydiff','M10_dydiff'),
 'simvp_add_COT':('M00_simvp','M10_simvp'),
 'observed_add_COT':('M00_observed','M10_observed'),
 'dydiff_oracle_COT_replace_forecast':('M10_dydiff','M10_dydiff_oracle_cot'),
 'dydiff_history_COT_replace_forecast':('M10_dydiff','M10_dydiff_history_cot'),
 'dydiff_vs_simvp_no_COT':('M00_simvp','M00_dydiff'),
 'dydiff_vs_simvp_with_COT':('M10_simvp','M10_dydiff')}
result={};maxdiff=0.0
for label,(b,a) in pairs.items():
    v=calc(b,a,lead>=15);result[label]=v;ref=stored['comparisons'][label]
    for key in ('before_macro_rmse_Wm2','after_macro_rmse_Wm2','delta_rmse_after_minus_before','relative_rmse_reduction_pct','date_block_95CI'):
        diff=float(np.max(np.abs(np.asarray(v[key])-np.asarray(ref[key]))));maxdiff=max(maxdiff,diff)
        assert diff<1e-9,(label,key,diff)
sub=calc('M00_dydiff','M10_dydiff',lead>=45)
ref=json.loads((DATA/'lead_subgroups_fp64.json').read_text())['all_forecast_slots_45_240']
assert np.max(abs(np.array(sub['date_block_95CI'])-ref['date_block_95CI']))<1e-9
out={'status':'PASS','method':'Independent NumPy from saved predictions; did not import score_saved.py',
     'rows':len(rows),'windows':len({r['window'] for r in rows}),'target_dates':len(set(date)),
     'max_difference_from_original_scores':maxdiff,'comparisons':result,'all_future_slots_45_240':sub,
     'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (DATA/'ghi_predictions.npz',DATA/'ghi_rows.csv')},'test_payloads_read':0}
(DATA/'LOCAL_INDEPENDENT_GHI_RECHECK.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':out['status'],'max_diff':maxdiff,'dydiff_add_COT':result['dydiff_add_COT'], 'all_future':sub},ensure_ascii=False,indent=2))
