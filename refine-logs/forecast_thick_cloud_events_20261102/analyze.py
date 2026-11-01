from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent;ROOT=P.parents[1]
SEEDS=[42,43,44];BINS=['Near-zero','Thin','Moderate','Thick'];NBOOT=5000

def load():
 allrows=[]
 sources={
  'Hubei':ROOT/'refine-logs/hubei_threeframe_dydiff_forecast_20261102/recovered/evaluation',
  'Hunan':ROOT/'refine-logs/fulltrain_forecast_cot_20261101/observed_trained_heads_evaluation_v1'}
 for reg,base in sources.items():
  f=pd.read_csv(base/'keys.csv');z=np.load(base/'predictions.npz');f['region']=reg
  if reg=='Hubei':
   f['target']=pd.to_datetime(f['target_time_bjt'].astype(str),format='%Y%m%d%H%M');f['init']=pd.to_datetime(f['init_time_bjt'].astype(str),format='%Y%m%d%H%M');pred=z['prediction']
  else:
   f['target']=pd.to_datetime(f['target_time_utc'])+pd.Timedelta(hours=8);f['init']=pd.to_datetime(f['init_time_utc'])+pd.Timedelta(hours=8);pred=np.stack([z['prediction_A'].T,z['prediction_AC'].T],axis=-1)
  f['truth']=z['truth'].astype(float);f['clear']=z['clear'].astype(float)
  for j,s in enumerate(SEEDS):f[f'A{s}']=pred[:,j,0];f[f'AC{s}']=pred[:,j,1]
  allrows.append(f)
 d=pd.concat(allrows,ignore_index=True);d['date']=d.target.dt.strftime('%Y-%m-%d')
 assert (d.target-d.init).dt.total_seconds().eq(d.lead_minutes*60).all()
 assert not d.duplicated(['region','station','init','target']).any()
 ref=pd.read_csv(P/'reference_centers.csv');ref['target']=pd.to_datetime(ref.target.astype(str),format='%Y%m%d%H%M')
 ref.loc[~ref.valid.eq(1),'cpp']=np.nan
 d=d.merge(ref,on=['region','station','target'],how='left',validate='many_to_one')
 cp=pd.read_csv(P/'forecast_cot_centers.csv');cp['target']=pd.to_datetime(cp.target.astype(str),format='%Y%m%d%H%M');cp['init']=pd.to_datetime(cp.init.astype(str),format='%Y%m%d%H%M')
 d=d.merge(cp,on=['region','station','init','target'],how='left',validate='one_to_one');assert d['cot'].notna().all()
 d['class']=pd.cut(d.cpp,[-1e-10,.1,5,30,100.00001],labels=BINS,right=False).astype(str).replace('nan','Unknown')
 grouped=d.groupby(['region','station','target'])
 max_truth_difference=float((grouped.truth.max()-grouped.truth.min()).max());max_clear_difference=float((grouped.clear.max()-grouped.clear.min()).max())
 assert max_truth_difference<1e-3 and max_clear_difference<1e-3
 (P/'TRUTH_KEY_CHECK.json').write_text(json.dumps({'max_GHI_difference_across_origins':max_truth_difference,'max_clear_sky_difference_across_origins':max_clear_difference,'unique_forecast_rows':len(d),'nonnegative_reference_classification':True},indent=2))
 # Validate reference units/indexing against the existing independent Hubei reference dump.
 rv=ROOT/'refine-logs/hubei_same_time_geometry_20261102/r_validation';kk=pd.read_csv(rv/'val_keys.csv');ar=np.load(rv/'val_reference_cpp_cot.npy');mask=np.load(rv/'val_valid_mask.npy');positions={'Niushou':(8,6),'Wangzhai':(9,8),'Mizhuang':(7,10)}
 lookup={str(t):i for i,t in enumerate(kk.time_bjt)};maxdiff=0.;n=0;class_mismatches=0
 for r in ref[ref.region.eq('Hubei')&ref.valid.eq(1)].itertuples():
  j=lookup.get(r.target.strftime('%Y%m%d%H%M'));y,x=positions[r.station]
  if j is not None and mask[j,0,y,x]:
   old=float(ar[j,0,y,x]);maxdiff=max(maxdiff,abs(r.cpp-old));n+=1;class_mismatches+=int(np.searchsorted([.1,5,30],old,side='right')!=np.searchsorted([.1,5,30],r.cpp,side='right'))
 assert n>100 and maxdiff<1e-4 and class_mismatches==0,(n,maxdiff,class_mismatches)
 (P/'REFERENCE_JOIN_CHECK.json').write_text(json.dumps({'hubei_reference_common_points':n,'maximum_absolute_difference':maxdiff,'stratum_class_mismatches':class_mismatches,'keys_unique':True,'all_rows':len(d),'reference_role':'offline strata only; no new model inference'},indent=2))
 return d

def score(q,boot=True):
 if q.empty:return None
 dates=sorted(q.date.unique());point=[];mae=[];bias=[];ss=[];counts=[];perseed={s:[] for s in SEEDS}
 for st in sorted(q.station.unique()):
  r=q[q.station.eq(st)];idx=pd.Categorical(r.date,categories=dates).codes;count=np.bincount(idx,minlength=len(dates))
  for seed in SEEDS:
   e=r[[f'A{seed}',f'AC{seed}']].to_numpy()-r.truth.to_numpy()[:,None];v=np.sqrt(np.mean(e**2,axis=0));point.append(v);perseed[seed].append(v);mae.append(np.mean(abs(e),axis=0));bias.append(np.mean(e,axis=0));ss.append(np.stack([np.bincount(idx,weights=e[:,a]**2,minlength=len(dates)) for a in [0,1]],axis=-1));counts.append(count)
 v=np.mean(point,axis=0);unique=q.drop_duplicates(['station','target']);tb=[]
 for seed in SEEDS:
  t=q.assign(eA=(q[f'A{seed}']-q.truth)**2,eAC=(q[f'AC{seed}']-q.truth)**2).groupby(['station','target'])[['eA','eAC']].mean().reset_index()
  tb.append(np.sqrt(t.groupby('station')[['eA','eAC']].mean()).mean().to_numpy())
 out={'n_rows':len(q),'n_station_targets':len(unique),'n_dates':len(dates),'n_stations':q.station.nunique(),'A_rmse':v[0],'AC_rmse':v[1],'delta_rmse':v[1]-v[0],'gain_percent':100*(1-v[1]/v[0]),'A_mae':np.mean(mae,axis=0)[0],'AC_mae':np.mean(mae,axis=0)[1],'A_bias':np.mean(bias,axis=0)[0],'AC_bias':np.mean(bias,axis=0)[1],'A_seed_sd':np.std([np.mean(perseed[s],axis=0)[0] for s in SEEDS],ddof=1),'AC_seed_sd':np.std([np.mean(perseed[s],axis=0)[1] for s in SEEDS],ddof=1),'target_balanced_A':np.mean(tb,axis=0)[0],'target_balanced_AC':np.mean(tb,axis=0)[1]}
 if boot and len(dates)>1:
  w=np.random.default_rng(20261102).multinomial(len(dates),np.full(len(dates),1/len(dates)),size=NBOOT);den=w@np.asarray(counts).T;ok=(den>0).all(1);b=np.sqrt(np.einsum('bd,cda->bca',w,np.asarray(ss))[ok]/den[ok,:,None]).mean(axis=1)
  out.update(delta_ci95=np.quantile(b[:,1]-b[:,0],[.025,.975]).tolist(),bootstrap_draws_defined=int(ok.sum()),bootstrap_draws_total=NBOOT,bootstrap_missing_cell_fraction=float(1-ok.mean()))
  # Leave-one-date-out, only when the fixed station/seed estimand remains defined.
  l=[];ssa=np.asarray(ss);cc=np.asarray(counts)
  for i in range(len(dates)):
   nn=cc.sum(1)-cc[:,i]
   if (nn<=0).any():continue
   p=np.sqrt((ssa.sum(1)-ssa[:,i])/nn[:,None]).mean(0);l.append(p[1]-p[0])
  out['leave_one_date_out_delta_range']=[float(min(l)),float(max(l))] if l else None
 return out

def episodes(d):
 u=d.drop_duplicates(['region','station','target']).sort_values(['region','station','target']);events=[]
 for (reg,st),g in u.groupby(['region','station']):
  run=[]
  def flush():
   if not run:return
   r=g.loc[run];sub=d[d.region.eq(reg)&d.station.eq(st)&d.target.isin(r.target)]
   # Equal target weight when deciding case direction, not repeated-origin frequency.
   a=[];b=[]
   for seed in SEEDS:
    e=sub.assign(a=(sub[f'A{seed}']-sub.truth)**2,b=(sub[f'AC{seed}']-sub.truth)**2).groupby('target')[['a','b']].mean();a.append(np.sqrt(e.a.mean()));b.append(np.sqrt(e.b.mean()))
   av=float(np.mean(a));bv=float(np.mean(b));events.append({'event_id':f'{reg}_{st}_{r.target.iloc[0]:%Y%m%d%H%M}','region':reg,'station':st,'start':r.target.iloc[0].isoformat(),'end':r.target.iloc[-1].isoformat(),'n_targets':len(r),'n_forecast_rows':len(sub),'peak_cpp':r.cpp.max(),'peak_time':r.loc[r.cpp.idxmax(),'target'].isoformat(),'A_rmse_target_balanced':av,'AC_rmse_target_balanced':bv,'delta_rmse_target_balanced':bv-av,'direction':'improved' if bv<av else 'degraded'})
  prev=None
  for idx,r in g.iterrows():
   if r['class']!='Thick':flush();run=[];prev=None;continue
   if prev is not None and r.target-prev!=pd.Timedelta(minutes=15):flush();run=[]
   run.append(idx);prev=r.target
  flush()
 return pd.DataFrame(events)

def main():
 d=load();events=episodes(d);events.to_csv(P/'events.csv',index=False)
 records=[];cot=[];station_metrics=[]
 for reg,g in d.groupby('region'):
  base=score(g);expected={'Hubei':(101.01782738063906,96.07131999287587),'Hunan':(122.58830279749982,120.50183451559512)}[reg]
  assert np.allclose([base['A_rmse'],base['AC_rmse']],expected,atol=1e-7)
  for scope,keep in [('all',g.index==g.index),('lead15_30',g.lead_minutes<=30),('lead45_240',g.lead_minutes>=45)]+[(f'lead{l}',g.lead_minutes.eq(l)) for l in range(15,241,15)]:
   gg=g[keep]
   for name in ['All','CPP-valid']+BINS+['Unknown']:
    q=gg if name=='All' else gg[gg.cpp.notna()] if name=='CPP-valid' else gg[gg['class'].eq(name)]
    v=score(q)
    if v:records.append({'region':reg,'scope':scope,'stratum':name,**v})
  for st,q in [('All',g)]+list(g.groupby('station')):
   q=q[q.cpp.notna()];actual=q.cpp>=30;forecast=q.cot>=30;tp=int((actual&forecast).sum());fn=int((actual&~forecast).sum());fp=int((~actual&forecast).sum())
   cot.append({'region':reg,'station':st,'valid_rows':len(q),'valid_station_targets':q.drop_duplicates(['station','target']).shape[0],'TP':tp,'FN':fn,'FP':fp,'POD':tp/(tp+fn) if tp+fn else None,'FAR':fp/(tp+fp) if tp+fp else None,'CSI':tp/(tp+fn+fp) if tp+fn+fp else None,'COT_RMSE':float(np.sqrt(np.mean((q.cot-q.cpp)**2))),'thick_COT_bias':float((q.cot-q.cpp)[actual].mean()) if actual.any() else None})
  for st,ss in g.groupby('station'):
   for scope,keep in [('all',ss.index==ss.index),('lead15_30',ss.lead_minutes<=30),('lead45_240',ss.lead_minutes>=45)]:
    for name in BINS:
     v=score(ss[keep&ss['class'].eq(name)])
     if v:station_metrics.append({'region':reg,'station':st,'scope':scope,'stratum':name,**v})
 pd.DataFrame(records).to_csv(P/'strata_metrics.csv',index=False)
 pd.DataFrame(station_metrics).to_csv(P/'station_strata_metrics.csv',index=False)
 def default(x):
  if isinstance(x,np.generic):return x.item()
  raise TypeError(type(x).__name__)
 coverage=[]
 for reg,q in d.groupby('region'):
  u=q.drop_duplicates(['station','target']);ee=events[events.region.eq(reg)];coverage.append({'region':reg,'GHI_rows':len(q),'station_targets':len(u),'CPP_rows':int(q.cpp.notna().sum()),'CPP_station_targets':int(u.cpp.notna().sum()),'thick_rows':int(q['class'].eq('Thick').sum()),'thick_station_targets':int(u['class'].eq('Thick').sum()),'thick_dates':q[q['class'].eq('Thick')].date.nunique(),'thick_segments':len(ee),'multi_time_segments':int((ee.n_targets>=2).sum()),'event_directions':ee.direction.value_counts().to_dict()})
 result={'state':'COMPLETE_STATISTICS','reference':'QA-valid station-centre CPP; bins [0,0.1,5,30,100]; Unknown retained in total','estimand':'equal station / seed RMSE; repeated forecast origins retained; separate target-balanced sensitivity','bootstrap':'5000 BJT target-date draws; undefined fixed-cell draws omitted and count reported, not replaced with zero; intervals descriptive after validation selection','coverage':coverage,'metrics':records,'cot_detection':cot,'test_payloads_read':0,'new_training':False}
 (P/'RESULTS.json').write_text(json.dumps(result,indent=2,default=default,allow_nan=False))
 # Deterministic conditional case selection, date first (not largest gain).
 chosen=[]
 for reg in ['Hubei','Hunan']:
  for direction in ['improved','degraded']:
   cand=events[events.region.eq(reg)&events.direction.eq(direction)&events.n_targets.ge(2)].sort_values(['start','station'])
   fallback=False
   if cand.empty:cand=events[events.region.eq(reg)&events.direction.eq(direction)].sort_values(['start','station']);fallback=True
   if cand.empty:continue
   e=cand.iloc[0].to_dict();sub=d[d.region.eq(reg)&d.station.eq(e['station'])];start=pd.Timestamp(e['start']);end=pd.Timestamp(e['end'])
   eligible=[]
   for init,ss in sub.groupby('init'):
    hits=ss.target.between(start,end).sum()
    if hits:
     eligible.append((int(hits),int((ss.target>=end).sum()>0),-abs((start-init).total_seconds()/60-60),init))
   eligible.sort(reverse=True);init=eligible[0][-1];q=sub[sub.init.eq(init)].sort_values('target');peak=pd.Timestamp(e['peak_time']);frames=[]
   for target in [peak-pd.Timedelta(minutes=30),peak,q.target.max()]:frames.append(q.iloc[np.argmin(abs(q.target-target))].target.isoformat())
   e.update(init=init.isoformat(),display_times=list(dict.fromkeys(frames)),single_reference_fallback=fallback);chosen.append(e)
 (P/'CASE_SELECTION.json').write_text(json.dumps({'selection':'earliest qualifying improved/degraded segment per region; origin maximizes event coverage, then proximity to 60-min advance; no maximization of error gain','display_frame_rule':'reference peak minus30min / reference peak / last GHI-valid target of selected initialization; same rule for all four; no error-based frame selection','cases':chosen},indent=2,default=default))
 print(json.dumps(coverage,default=default))
 for r in records:
  if r['scope']=='all' and r['stratum']=='Thick':print('THICK',r['region'],r['A_rmse'],r['AC_rmse'],r['gain_percent'],r.get('delta_ci95'))
 print('CASES',[(c['event_id'],c['direction'],c['init']) for c in chosen])

if __name__=='__main__':main()
