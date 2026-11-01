from pathlib import Path
import csv, json, hashlib
import numpy as np

root=Path(__file__).resolve().parents[2]
source=root/'refine-logs/hunan_dydiff_20261028/full_validation_20261029/evaluation_attempt1'
with (source/'ghi_rows.csv').open(encoding='utf-8-sig',newline='') as stream: rows=list(csv.DictReader(stream))
stations=np.array([r['station'] for r in rows]); dates=np.array([r['target_date_bjt'] for r in rows])
unique,di=np.unique(dates,return_inverse=True)
assert len(rows)==63480 and len(unique)==51
assert len({(r['station'],r['init_time_utc'],r['target_time_utc']) for r in rows})==len(rows)
assert all(int(r['lead_minutes']) in range(15,241,15) for r in rows)
weights=np.random.default_rng(20261028).multinomial(len(unique),np.full(len(unique),1/len(unique)),size=5000)
out={'scope':'Complete frozen Hunan validation; source NPZ independently recomputed, no inference/training','rows':len(rows),'target_dates':len(unique),'test_or_buffer_payload_read':False,'methods':{}}
with np.load(source/'ghi_predictions.npz',allow_pickle=False) as z:
 y=np.asarray(z['truth'],np.float64)
 assert np.isfinite(y).all()
 for method in ['simvp','dydiff']:
  point=[[],[]];boot=[[],[]];by_station={}
  for st in ['sili','zhujia']:
   ids=np.flatnonzero(stations==st); count=np.bincount(di[ids],minlength=len(unique)); denom=weights@count
   assert (denom>0).all()
   cell=[[],[]]
   for seed in [42,43,44]:
    for arm,index in [('M00',0),('M10',1)]:
     pred=np.asarray(z[f'{st}_seed{seed}_{arm}_{method}'][ids],np.float64)
     assert np.isfinite(pred).all()
     err=pred-y[ids]; squared=err*err
     value=float(np.sqrt(squared.mean()));point[index].append(value);cell[index].append(value)
     sums=np.bincount(di[ids],weights=squared,minlength=len(unique))
     boot[index].append(np.sqrt((weights@sums)/denom))
   b,a=map(np.mean,cell)
   by_station[st]={'before':float(b),'after':float(a),'gain_percent':float(100*(b-a)/b),'seed_deltas':(np.array(cell[1])-np.array(cell[0])).tolist()}
  b,a=map(np.mean,point); delta_boot=np.mean(boot[1],axis=0)-np.mean(boot[0],axis=0)
  out['methods'][method]={'before':float(b),'after':float(a),'delta':float(a-b),'gain_percent':float(100*(b-a)/b),'ci95':np.quantile(delta_boot,[.025,.975]).tolist(),'by_station':by_station}
existing=json.loads((source/'ghi_scores_fp64.json').read_text(encoding='utf-8'))['comparisons']
for method,rec in out['methods'].items():
 expected=existing[method+'_add_COT']
 assert abs(rec['before']-expected['before_macro_rmse_Wm2'])<1e-10
 assert abs(rec['after']-expected['after_macro_rmse_Wm2'])<1e-10
 assert np.max(abs(np.array(rec['ci95'])-expected['date_block_95CI']))<1e-10
out['source_sha256']={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ['ghi_rows.csv','ghi_predictions.npz','ghi_scores_fp64.json']}
out['matches_saved_scores_and_CI']=True
dest=Path(__file__).with_name('LATEST_HUNAN_RECHECK.json');dest.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out['methods'],ensure_ascii=False))
