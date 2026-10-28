"""Independent array-only audit of coverage, percent denominators and interaction."""
from pathlib import Path
import csv,hashlib,json
from datetime import datetime,timedelta
from collections import Counter
import numpy as np
B=Path(__file__).parent;R=B/'attempt1'
# Resolve explicitly relative to the repository refine-logs folder.
H=B.parents[1]/'cloud_probability_input_20261020/results/four_arm_paired_v1_attempt1'
def csvrows(p):
 with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def rmse(p,y):return float(np.sqrt(np.mean((np.array(p,np.float64)-np.array(y,np.float64))**2)))
rows=csvrows(R/'ghi_rows.csv');st=np.array([r['station'] for r in rows]);dates=np.array([r['target_date_bjt'] for r in rows]);lead=np.array([int(r['lead_minutes']) for r in rows])
with np.load(R/'ghi_predictions.npz') as z:p={k:z[k].copy() for k in z.files}
y=p['truth'];units=[];coverage={};sums={m:[] for m in ('M00_observed','M10_observed','M00_dydiff','M10_dydiff')}
groups=[]
for s in ('sili','zhujia'):
 ids=np.flatnonzero(st==s)
 for seed in (42,43,44):
  groups.append((s,seed,ids))
  for m in sums:sums[m].append(rmse(p[f'{s}_seed{seed}_{m}'][ids],y[ids]))
full={arm:[] for arm in ('M00','M10')};hashes={}
for s in ('sili','zhujia'):
 for seed in (42,43,44):
  for arm in ('M00','M10'):
   f=H/f'seed{seed}'/s/arm/'validation_predictions.npz'
   with np.load(f) as z:full[arm].append(rmse(z['prediction'],z['truth']))
   hashes[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
def record(before,after):
 b=float(np.mean(before));a=float(np.mean(after));return {'before_RMSE':b,'after_RMSE':a,'delta_RMSE':a-b,'relative_reduction_pct':100*(b-a)/b}
coverage['full_validation_observed_same_heads']=record(full['M00'],full['M10'])
coverage['same_432_observed']=record(sums['M00_observed'],sums['M10_observed'])
coverage['same_432_forecast']=record(sums['M00_dydiff'],sums['M10_dydiff'])
ud=np.unique(dates);w=np.random.default_rng(20261028).multinomial(len(ud),np.ones(len(ud))/len(ud),size=5000).astype(float)
br={m:[] for m in sums};point=[]
for s,seed,ids in groups:
 count=np.array([sum(dates[ids]==d) for d in ud]);den=w@count;assert (den>0).all()
 for m in br:
  e=(p[f'{s}_seed{seed}_{m}'][ids]-y[ids])**2
  ss=np.array([e[dates[ids]==d].sum() for d in ud]);br[m].append(np.sqrt((w@ss)/den))
diff=(np.mean(br['M10_dydiff'],axis=0)-np.mean(br['M00_dydiff'],axis=0))-(np.mean(br['M10_observed'],axis=0)-np.mean(br['M00_observed'],axis=0))
interaction={'delta_forecast_COT_effect_minus_observed_COT_effect':coverage['same_432_forecast']['delta_RMSE']-coverage['same_432_observed']['delta_RMSE'],'paired_date_95CI':np.quantile(diff,[.025,.975]).tolist()}
lead_groups={}
for label,mask in [('15_30_min_some_history_inputs',lead<=30),('45_240_min_all_three_images_forecast',lead>=45)]:
 before=[];after=[]
 for s,seed,_ in groups:
  ids=np.flatnonzero((st==s)&mask);before.append(rmse(p[f'{s}_seed{seed}_M00_dydiff'][ids],y[ids]));after.append(rmse(p[f'{s}_seed{seed}_M10_dydiff'][ids],y[ids]))
 lead_groups[label]={'n':int(mask.sum()),**record(before,after)}
selection=json.loads((B.parent/'cross_hubei_on_hunan/selection.json').read_text())
times=[(datetime.fromisoformat(r['BJT_start'])+timedelta(minutes=105)).strftime('%H:%M') for r in selection['rows']]
result={'status':'PASS_NUMERIC_RECOMPUTATION_WITH_SCOPE_WARNING','coverage_comparison':coverage,'interaction':interaction,'lead_groups':lead_groups,
 'suspect_long_zero_records':sum(r.get('suspect_daylight_long_zero')=='1' for r in rows),'exact_zero_GHI_records':int(sum(y==0)),
 'selection_actual_init_times_BJT':dict(Counter(times)),'GHI_record_init_times_BJT':dict(Counter((datetime.fromisoformat(r['init_time_utc'])+timedelta(hours=8)).strftime('%H:%M') for r in rows)),
 'full_source_prediction_hashes':hashes,'test_payloads_read':0,'warning':'midday metadata wording inaccurate; mostly08:00 initialization; selected reused-validation rather than full forecast evaluation'}
(B/'NUMERIC_AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
