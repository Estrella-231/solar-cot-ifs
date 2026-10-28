"""Independent FP64 scoring of saved COT and GHI paired predictions."""
import argparse,csv,json
from pathlib import Path
import numpy as np

def csvrows(p):
 with Path(p).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def scores(p,y):
 e=np.asarray(p,np.float64)-np.asarray(y,np.float64)
 assert len(e)>0 and np.isfinite(e).all()
 return {'n':len(e),'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(np.mean(abs(e))),'bias':float(e.mean())}
def macro_boot(dates,groups,pairs,seed=20261028,nboot=5000):
 dates=np.array(dates);unique=np.unique(dates[np.concatenate(groups)]);rng=np.random.default_rng(seed)
 w=rng.multinomial(len(unique),np.ones(len(unique))/len(unique),size=nboot).astype(np.float64)
 bm=[];am=[];point=[]
 for ids,(before,after,y) in zip(groups,pairs):
  before=np.array(before,np.float64);after=np.array(after,np.float64);y=np.array(y,np.float64)
  assert len(ids)==len(before)==len(after)==len(y)
  count=np.array([sum(dates[ids]==d) for d in unique],np.float64)
  bs=np.array([np.sum((before[dates[ids]==d]-y[dates[ids]==d])**2) for d in unique]);ass=np.array([np.sum((after[dates[ids]==d]-y[dates[ids]==d])**2) for d in unique])
  denom=w@count;bm.append(np.sqrt((w@bs)/np.maximum(denom,1)));am.append(np.sqrt((w@ass)/np.maximum(denom,1)))
  point.append(float(np.sqrt(np.mean((after-y)**2))-np.sqrt(np.mean((before-y)**2))))
 b=np.mean(bm,axis=0);a=np.mean(am,axis=0);dif=a-b
 return {'delta_rmse_after_minus_before':float(np.mean(point)),'date_block_95CI':np.quantile(dif,[.025,.975]).tolist(),'target_dates':unique.tolist(),'bootstrap_replicates':nboot,'scheme':'joint target-BJT-date multinomial; equal groups, all fixed seeds retained; descriptive selected-validation interval'}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);a=ap.parse_args();root=a.directory
 complete=json.loads((root/'completion.json').read_text());assert complete['status']=='PASS' and complete['test_payloads_read']==0
 cr=csvrows(root/'cot_rows.csv');gr=csvrows(root/'ghi_rows.csv')
 with np.load(root/'cot_predictions.npz') as z:c={k:z[k].copy() for k in z.files}
 methods=('observed','dydiff','simvp','persistence');station=np.array([r['station'] for r in cr]);dates=np.array([r['target_date_bjt'] for r in cr]);lead=np.array([int(r['lead_minutes']) for r in cr])
 pred={m:np.expm1(np.array(c['cot_'+m][:,8,8],np.float64)) for m in methods}
 proxy=pred['observed'];cpp=np.array(c['cpp'][:,8,8],np.float64)
 proxyvalid=c['observed_valid_center'].astype(bool);cppvalid=c['cpp_mask'][:,8,8].astype(bool)&np.isfinite(cpp)
 cot={'scope':'station pixel [8,8]; forecast COT versus same-R response and sparse independently keyed CPP retrieval reference','proxy_response':{},'CPP_reference':{}}
 for refname,ref,valid in [('proxy_response',proxy,proxyvalid),('CPP_reference',cpp,cppvalid)]:
  cot[refname]['total_rows']=int(valid.sum());cot[refname]['dates']=sorted(set(dates[valid]));cot[refname]['stations']={}
  for s in ('sili','zhujia'):
   ids=np.flatnonzero((station==s)&valid)
   if not len(ids):cot[refname]['stations'][s]={'n':0};continue
   rec={m:scores(pred[m][ids],ref[ids]) for m in methods};rec['n_dates']=len(set(dates[ids]));cot[refname]['stations'][s]=rec
  for before,after in [('simvp','dydiff'),('persistence','dydiff')]:
   groups=[np.flatnonzero((station==s)&valid) for s in ('sili','zhujia')];groups=[ids for ids in groups if len(ids)]
   if groups:
    pairs=[(pred[before][ids],pred[after][ids],ref[ids]) for ids in groups]
    info=macro_boot(dates,groups,pairs);b=float(np.mean([scores(pred[before][ids],ref[ids])['rmse'] for ids in groups]));aa=float(np.mean([scores(pred[after][ids],ref[ids])['rmse'] for ids in groups]));info.update(before_macro_rmse=b,after_macro_rmse=aa,relative_rmse_reduction_pct=100*(b-aa)/b if b else None)
    cot[refname][after+'_vs_'+before]=info
 with (root/'cot_lead_metrics.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['reference','station','lead_minutes','method','n','rmse_COT','mae_COT','bias_COT'])
  for refname,ref,valid in [('R_observed_proxy',proxy,proxyvalid),('CPP_retrieval',cpp,cppvalid)]:
   for s in ('sili','zhujia'):
    for t in range(15,241,15):
     ids=np.flatnonzero((station==s)&(lead==t)&valid)
     if len(ids):
      for m in methods:
       v=scores(pred[m][ids],ref[ids]);w.writerow([refname,s,t,m,v['n'],v['rmse'],v['mae'],v['bias']])
 dump(root/'cot_scores_fp64.json',cot)
 with np.load(root/'ghi_predictions.npz') as z:g={k:z[k].copy() for k in z.files}
 y=np.array(g.pop('truth'),np.float64);clear=g.pop('clear');st=np.array([r['station'] for r in gr]);dates=np.array([r['target_date_bjt'] for r in gr])
 assert len(y)==len(gr) and len({(r['station'],r['window'],r['lead_minutes']) for r in gr})==len(gr)
 comparisons={};raw=[];groups=[];labels=[]
 for s in ('sili','zhujia'):
  ids=np.flatnonzero(st==s)
  for seed in (42,43,44):
   groups.append(ids);labels.append((s,seed))
   for name,p in g.items():
    if not name.startswith(f'{s}_seed{seed}_'):continue
    v=scores(p[ids],y[ids]);raw.append({'station':s,'seed':seed,'condition':name.removeprefix(f'{s}_seed{seed}_'),**v})
 compare=[('dydiff_add_COT','M00_dydiff','M10_dydiff'),('simvp_add_COT','M00_simvp','M10_simvp'),('observed_add_COT','M00_observed','M10_observed'),
  ('dydiff_oracle_COT_replace_forecast','M10_dydiff','M10_dydiff_oracle_cot'),('dydiff_history_COT_replace_forecast','M10_dydiff','M10_dydiff_history_cot'),
  ('dydiff_vs_simvp_no_COT','M00_simvp','M00_dydiff'),('dydiff_vs_simvp_with_COT','M10_simvp','M10_dydiff')]
 for label,before,after in compare:
  pairs=[(g[f'{s}_seed{seed}_{before}'][ids],g[f'{s}_seed{seed}_{after}'][ids],y[ids]) for ids,(s,seed) in zip(groups,labels)]
  rec=macro_boot(dates,groups,pairs);b=float(np.mean([scores(p[0],p[2])['rmse'] for p in pairs]));aa=float(np.mean([scores(p[1],p[2])['rmse'] for p in pairs]));rec.update(before_macro_rmse_Wm2=b,after_macro_rmse_Wm2=aa,relative_rmse_reduction_pct=100*(b-aa)/b)
  rec['by_station']={}
  for s in ('sili','zhujia'):
   sel=[p for p,l in zip(pairs,labels) if l[0]==s];bb=np.mean([scores(p[0],p[2])['rmse'] for p in sel]);aaa=np.mean([scores(p[1],p[2])['rmse'] for p in sel]);rec['by_station'][s]={'before_rmse':float(bb),'after_rmse':float(aaa),'delta_rmse':float(aaa-bb),'relative_reduction_pct':float(100*(bb-aaa)/bb),'seed_deltas':[scores(p[1],p[2])['rmse']-scores(p[0],p[2])['rmse'] for p in sel]}
  comparisons[label]=rec
 with (root/'ghi_run_metrics_fp64.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(raw[0]));w.writeheader();w.writerows(raw)
 result={'scope':'16 selected Hunan validation windows; frozen corrected R and same-time trained paired GHI heads, no forecast-adapted retraining','rows':len(gr),'stations':{s:int(sum(st==s)) for s in set(st)},'unique_dates':sorted(set(dates)),
  'seeds':[42,43,44],'aggregation':'RMSE per station/seed then mean; not pooled RMSE or seed ensemble','comparisons':comparisons,'test_payloads_read':0}
 dump(root/'ghi_scores_fp64.json',result);dump(root/'FP64_SCORE_COMPLETE.json',{'status':'PASS','independent_saved_prediction_scoring':True,'test_payloads_read':0})
 # Influence check and coverage-matched COT response: diagnostics, not selection.
 influence=[]
 for d in sorted(set(dates)):
  per=[];loo=[]
  for s,seed in labels:
   for mask,values in [((st==s)&(dates==d),per),((st==s)&(dates!=d),loo)]:
    if not mask.any():continue
    p=g[f'{s}_seed{seed}_M00_dydiff'][mask];q=g[f'{s}_seed{seed}_M10_dydiff'][mask]
    values.append(scores(q,y[mask])['rmse']-scores(p,y[mask])['rmse'])
  influence.append({'date':str(d),'n':int(sum(dates==d)),'per_date_delta_rmse':float(np.mean(per)),'leave_one_date_out_delta_rmse':float(np.mean(loo))})
 dump(root/'date_sensitivity_fp64.json',{'scope':'post-hoc influence diagnostic; no samples selected from these results','dates':influence})
 keys={(r['station'],r['window'],r['lead_minutes']) for r in gr}
 keep=np.array([(r['station'],r['window'],r['lead_minutes']) in keys for r in cr]);csites=np.array([r['station'] for r in cr]);common={}
 for label,ref,valid in [('R_observed_proxy',proxy,proxyvalid),('CPP_retrieval',cpp,cppvalid)]:
  common[label]={}
  for method in ('dydiff','simvp','persistence'):
   by={}
   for s in ('sili','zhujia'):
    ids=np.flatnonzero(keep&valid&(csites==s))
    by[s]=scores(pred[method][ids],ref[ids])
   common[label][method]={'stations':by,'equal_station_RMSE':float(np.mean([v['rmse'] for v in by.values()]))}
 dump(root/'COT_on_GHI_common_keys_fp64.json',common)
 print(json.dumps({'COT':cot,'GHI':result},indent=2))
if __name__=='__main__':main()
