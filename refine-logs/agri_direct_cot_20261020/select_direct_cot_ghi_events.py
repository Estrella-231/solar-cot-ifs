from __future__ import annotations
import csv,json,sys,hashlib
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np

score=Path(sys.argv[1]); workers=Path(sys.argv[2]); bank=Path(sys.argv[3]); pack=Path(sys.argv[4]); cpp=Path(sys.argv[5]); out=score/'event_diagnostics';out.mkdir(parents=True,exist_ok=True)
# CPP station-center reference lookup: retrieval reference only, never an input.
cot=np.load(cpp/'cot.npy',mmap_mode='r',allow_pickle=False);mask=np.load(cpp/'mask.npy',mmap_mode='r',allow_pickle=False)
ref={}
with (cpp/'rows.csv').open(newline='',encoding='utf-8') as f:
 for r in csv.DictReader(f):
  if r['split']=='val' and r['state']=='SUPERVISED':
   i=int(r['array_index']);ref[(r['station'].lower(),r['target_time_utc'])]=(float(cot[i,0,8,8]),bool(mask[i,0,8,8]))
# Common physical rows and seed-42 forecast trajectories; case plots are explicitly illustrative diagnostics.
tracks={}; candidate_counts={}
for s in ('sili','zhujia'):
 d=pack/s/'val'; rows=[json.loads(x) for x in (d/'keys.jsonl').read_text(encoding='utf-8').splitlines() if x]
 y=np.load(d/'ghi.npy',mmap_mode='r',allow_pickle=False);geo=np.load(bank/s/'val'/'geometry_raw.npy',mmap_mode='r',allow_pickle=False)
 cots=np.load(bank/s/'val'/'cot_raw'/'p_xc.npy',mmap_mode='r',allow_pickle=False)[:,0,1,1]
 pred={}
 for a in ('none','simvp_r','p_c','p_xc'):
  z=np.load(workers/f'{s}_worker'/s/a/'seed42'/'validation_predictions.npz',allow_pickle=False)
  pred[a]=np.asarray(z['prediction_ghi_wm2'],np.float64)
  kraw=np.asarray(z['keys_jsonl_bytes'],dtype=np.uint8).tobytes()
  assert kraw==(d/'keys.jsonl').read_bytes()
 records=[]
 for i,r in enumerate(rows):
  ri=ref.get((s,r['target_time_utc'])); rv=ri[0] if ri and ri[1] else float('nan')
  zi=float(geo[i,0,1,1]); t=datetime.strptime(r['target_time_utc'],'%Y-%m-%d %H:%M:%S')+timedelta(hours=8)
  records.append({'i':i,'station':s,'init':r['init_time_utc'],'target':r['target_time_utc'],'lead':int(r['lead_minutes']),
    'bjt_month':t.strftime('%Y-%m'),'ghi':float(y[i]),'cossoz':zi,'cpp_cot':rv,'p_xc_cot':float(cots[i]),
    **{f'pred_{a}':float(pred[a][i]) for a in pred}})
 by_init={}
 for r in records:by_init.setdefault(r['init'],[]).append(r)
 tracks[s]={'records':records,'by_init':by_init}
# Extract contiguous event runs by predeclared diagnostic bins: CPP center COT<1 for Zhujia and >=30 for Sili.
def runs_for(s, predicate, tag):
 out_runs=[]
 for init,rr in tracks[s]['by_init'].items():
  rr=sorted(rr,key=lambda r:r['lead']); cur=[]
  def emit(x):
   if len(x)<3:return
   y=np.array([v['ghi'] for v in x]); p0=np.array([v['pred_none'] for v in x]); px=np.array([v['pred_p_xc'] for v in x])
   pr=np.array([v['pred_simvp_r'] for v in x]);
   d_rmse=float(np.sqrt(np.mean((px-y)**2))-np.sqrt(np.mean((p0-y)**2)))
   base_rmse=float(np.sqrt(np.mean((p0-y)**2)))
   out_runs.append({'station':s,'init':init,'month':x[0]['bjt_month'],'bin':tag,'rows':x,'n':len(x),
      'lead_start':x[0]['lead'],'lead_end':x[-1]['lead'],'delta_rmse_pxc_minus_none':d_rmse,
      'delta_rmse_pxc_minus_simvp_r':float(np.sqrt(np.mean((px-y)**2))-np.sqrt(np.mean((pr-y)**2))),
      'base_rmse':base_rmse,'mean_cossoz':float(np.nanmean([v['cossoz'] for v in x]))})
  for r in rr:
   good=predicate(r)
   if good and (not cur or r['lead']==cur[-1]['lead']+15):cur.append(r)
   else:
    emit(cur);cur=[]
    if good:cur=[r]
  emit(cur)
 return out_runs
sili_runs=runs_for('sili',lambda r:np.isfinite(r['cpp_cot']) and r['cpp_cot']>=30,'CPP COT >=30')
zhu_runs=runs_for('zhujia',lambda r:np.isfinite(r['cpp_cot']) and r['cpp_cot']<1,'CPP COT <1')
# Select result-defined examples, explicitly not independent evidence.
selected={}
for s,cands,mode in (('sili',sili_runs,'best'),('zhujia',zhu_runs,'worst')):
 if not cands:continue
 primary=min(cands,key=lambda r:r['delta_rmse_pxc_minus_none']) if mode=='best' else max(cands,key=lambda r:r['delta_rmse_pxc_minus_none'])
 # Matched case: same station/reference COT bin, same BJT month where available,
 # similar solar geometry, baseline RMSE at or below the median, P-XC does not worsen.
 controls=[r for r in cands if r is not primary and r['delta_rmse_pxc_minus_none']<=0 and r['base_rmse']<=np.median([x['base_rmse'] for x in cands])]
 if controls:
  same=[r for r in controls if r['month']==primary['month']]
  pool=same or controls
  control=min(pool,key=lambda r:abs(r['mean_cossoz']-primary['mean_cossoz']))
 else:control=None
 selected[s]={'signal':primary,'control':control,'eligible_runs':len(cands),'control_candidates':len(controls)}
# If predeclared cloud-bin selections lack an eligible control, leave it missing rather than invent one.
case_rows=[]
for s,ss in selected.items():
 for label in ('signal','control'):
  e=ss[label]
  if e:
   for r in e['rows']:
    case_rows.append({'station':s,'case_type':label,'init_time_utc':e['init'],'bjt_month':e['month'],'lead_minutes':r['lead'],'target_time_utc':r['target'],
      'ghi_obs_wm2':r['ghi'],'ghi_none_seed42_wm2':r['pred_none'],'ghi_simvp_r_seed42_wm2':r['pred_simvp_r'],'ghi_p_c_seed42_wm2':r['pred_p_c'],'ghi_p_xc_seed42_wm2':r['pred_p_xc'],
      'cpp_center_cot':r['cpp_cot'],'p_xc_predicted_cot_center':r['p_xc_cot'],'cossoz_center':r['cossoz'],
      'event_bin':e['bin'],'event_delta_rmse_pxc_minus_none_wm2':e['delta_rmse_pxc_minus_none'],'event_baseline_rmse_wm2':e['base_rmse']})
with (out/'selected_event_rows_seed42.csv').open('w',newline='',encoding='utf-8-sig') as f:
 if case_rows:
  w=csv.DictWriter(f,fieldnames=list(case_rows[0]));w.writeheader();w.writerows(case_rows)
summary={'state':'COMPLETE_DIRECT_COT_B2_EVENT_DIAGNOSTIC_SELECTION','claim_scope':'posthoc validation examples only; not causal proof or independent test',
 'test_payload_opened':False,'buffer_payload_opened':False,'future_observed_agri_used_as_input':False,'cpp_used_as_inference_input':False,
 'selection_rule':'Sili: longest contiguous (>=3 leads) CPP-center COT>=30 run with largest P-XC improvement vs none; Zhujia: CPP-center COT<1 run with largest P-XC degradation vs none. Matched controls use same bin, same month if available, similar cosSOZ, non-degrading P-XC and baseline RMSE <= candidate-run median.',
 'eligible_runs':{'sili_cpp_ge30':len(sili_runs),'zhujia_cpp_lt1':len(zhu_runs)},'selected':{}}
for s,ss in selected.items():
 summary['selected'][s]={}
 for label in ('signal','control'):
  e=ss[label]
  summary['selected'][s][label]=None if e is None else {k:v for k,v in e.items() if k!='rows'}
summary['event_csv_sha256']=hashlib.sha256((out/'selected_event_rows_seed42.csv').read_bytes()).hexdigest()
(out/'event_selection_manifest.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2,allow_nan=False))
