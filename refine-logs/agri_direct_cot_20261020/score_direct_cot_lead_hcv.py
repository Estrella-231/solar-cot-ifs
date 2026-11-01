"""Paired date-block scoring for the H-only COT/lead factorial experiment."""
import argparse, csv, hashlib, json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

STATIONS=('sili','zhujia'); SEEDS=(42,43,44)
COMPARISONS={
 'cot_effect_with_lead':('pxc_lead','none_lead'),
 'cot_effect_without_lead':('pxc_no_lead','none_no_lead'),
 'lead_cot_interaction':None,
 'lead_effect_no_cot':('none_lead','none_no_lead'),
}

def bjt_date(k):
 return (datetime.fromisoformat(k['target_time_utc'])+timedelta(hours=8)).date()

def block_id(day):
 return f'{day.year:04d}-{day.month:02d}-B{(day.day-1)//2:02d}'

def load_all(root):
 data={}; seen={s:set() for s in STATIONS}; arm_set=set()
 for st in STATIONS:
  for f in range(5):
   fd=root/st/f'fold{f}'; keys=[json.loads(x) for x in (fd/'oof_keys.jsonl').read_text(encoding='utf-8').splitlines()]
   for arm in ('none_no_lead','pxc_no_lead','none_lead','pxc_lead'):
    for seed in SEEDS:
     p=fd/arm/f'seed{seed}'/'oof_predictions.npz'
     with np.load(p,allow_pickle=False) as z:
      pred=z['prediction_ghi'].astype(np.float64); truth=z['truth_ghi'].astype(np.float64)
     if len(pred)!=len(keys) or len(truth)!=len(keys): raise ValueError(f'key/pred length mismatch {st}/{f}/{arm}/{seed}')
     idkey=(st,f,arm,seed)
     data[idkey]={'keys':keys,'pred':pred,'truth':truth}
     arm_set.add(arm)
     if arm=='none_no_lead' and seed==SEEDS[0]:
      ids={(k['init_time_utc'],k['target_time_utc'],int(k['lead_minutes'])) for k in keys}
      if seen[st]&ids: raise ValueError('OOF key appears in multiple folds')
      seen[st]|=ids
 if arm_set!=set(('none_no_lead','pxc_no_lead','none_lead','pxc_lead')): raise ValueError('arm coverage mismatch')
 return data,seen

def aligned(data,st,fold,seed,arm1,arm0,short=False):
 a=data[(st,fold,arm1,seed)]; b=data[(st,fold,arm0,seed)]
 if a['keys']!=b['keys'] or not np.array_equal(a['truth'],b['truth']): raise ValueError('paired key/truth mismatch')
 ix=np.asarray([i for i,k in enumerate(a['keys']) if not short or int(k['lead_minutes'])<=60],dtype=int)
 if len(ix)==0: raise ValueError('empty lead slice')
 y=a['truth'][ix]
 return a['keys'],np.square(a['pred'][ix]-y),np.square(b['pred'][ix]-y),ix

def _components(comp):
 if comp=='lead_cot_interaction':
  return [('pxc_lead','none_lead',1.0),('pxc_no_lead','none_no_lead',-1.0)]
 a,b=COMPARISONS[comp]
 return [(a,b,1.0)]

def aggregate_blocks(data,comp,short):
 comps=_components(comp); arms=sorted({x for a,b,_ in comps for x in (a,b)})
 # block sums/counts by station, seed and arm; all arms share each fold's exact keys.
 agg={}
 for st in STATIONS:
  for sd in SEEDS:
   for f in range(5):
    ref=data[(st,f,arms[0],sd)]
    for arm in arms[1:]:
     other=data[(st,f,arm,sd)]
     if ref['keys']!=other['keys'] or not np.array_equal(ref['truth'],other['truth']): raise ValueError('factorial arm key/truth mismatch')
    keys=ref['keys']; truth=ref['truth']
    keep=np.asarray([i for i,k in enumerate(keys) if not short or int(k['lead_minutes'])<=60],dtype=int)
    if not len(keep): raise ValueError('empty lead slice')
    bls=[block_id(bjt_date(keys[i])) for i in keep]
    for arm in arms:
     pred=data[(st,f,arm,sd)]['pred']
     local=defaultdict(lambda:[0.0,0])
     for i,bl in zip(keep,bls):
      local[bl][0]+=float((pred[i]-truth[i])**2);local[bl][1]+=1
     for bl,v in local.items():
      z=agg.setdefault((st,sd,arm,bl),[0.0,0]);z[0]+=v[0];z[1]+=v[1]
 blocks=sorted({k[3] for k in agg})
 return agg,blocks,arms,comps

def _delta_from_weights(agg,blocks,comps,weights):
 values=[]
 for st in STATIONS:
  for sd in SEEDS:
   ds=0.0
   for a,b,sign in comps:
    pair=[]
    for arm in (a,b):
     sse=count=0.0
     for bl,w in zip(blocks,weights):
      if w:
       z=agg.get((st,sd,arm,bl))
       if z: sse+=z[0]*w;count+=z[1]*w
     if count<=0: raise ValueError('bootstrap sample has no rows for a station/seed')
     pair.append(np.sqrt(sse/count))
    ds+=sign*(pair[0]-pair[1])
   values.append(float(ds))
 return float(np.mean(values))

def point_delta(data,comp,short=False,drop_block=None):
 agg,blocks,arms,comps=aggregate_blocks(data,comp,short)
 weights=np.ones(len(blocks),dtype=np.float64)
 if drop_block is not None:
  weights=np.asarray([0.0 if b==drop_block else 1.0 for b in blocks])
 return _delta_from_weights(agg,blocks,comps,weights)

def bootstrap(data,comp,short,nboot=5000,seed=20261021):
 agg,blocks,arms,comps=aggregate_blocks(data,comp,short)
 rng=np.random.default_rng(seed); vals=np.empty(nboot,dtype=np.float64)
 for q in range(nboot):
  draw=rng.integers(0,len(blocks),size=len(blocks))
  weights=np.bincount(draw,minlength=len(blocks)).astype(np.float64)
  vals[q]=_delta_from_weights(agg,blocks,comps,weights)
 return _delta_from_weights(agg,blocks,comps,np.ones(len(blocks))),float(np.quantile(vals,.025)),float(np.quantile(vals,.975))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--bootstrap',type=int,default=5000);a=ap.parse_args()
 m=json.loads((a.run/'run_manifest.json').read_text(encoding='utf-8'))
 if m.get('state')!='COMPLETE_DIRECT_COT_LEAD_H_CV':raise RuntimeError('training receipt is not complete')
 for flag in ('test_payload_opened','buffer_payload_opened','future_observed_agri_used_as_input','cpp_used_as_input'):
  if m.get(flag) is not False:raise RuntimeError(f'forbidden source flag {flag}')
 data,seen=load_all(a.run)
 results={}
 for c in ('cot_effect_with_lead','cot_effect_without_lead','lead_cot_interaction','lead_effect_no_cot'):
  results[c]={}
  for label,short in (('all_leads',False),('plus15_60',True)):
   results[c][label]=dict(zip(('delta_rmse_wm2','ci95_low','ci95_high'),bootstrap(data,c,short,a.bootstrap)))
 # station/seed diagnostics and lead curve for the primary matched comparison
 results['station_delta_all_leads']={};results['seed_delta_all_leads']={}
 for st in STATIONS:
  by=[]
  for sd in SEEDS:
   aa=[];bb=[]
   for f in range(5):
    _,x,y,_=aligned(data,st,f,sd,'pxc_lead','none_lead',False);aa.extend(x);bb.extend(y)
   by.append(float(np.sqrt(np.mean(aa))-np.sqrt(np.mean(bb))))
  results['station_delta_all_leads'][st]={'seed_deltas':by,'mean':float(np.mean(by))}
 for sd in SEEDS:
  vals=[]
  for st in STATIONS:
   aa=[];bb=[]
   for f in range(5):
    _,x,y,_=aligned(data,st,f,sd,'pxc_lead','none_lead',False);aa.extend(x);bb.extend(y)
   vals.append(float(np.sqrt(np.mean(aa))-np.sqrt(np.mean(bb))))
  results['seed_delta_all_leads'][str(sd)]=float(np.mean(vals))
 results['leadwise_pxc_lead_minus_none_lead']=[]
 for lead in range(15,241,15):
  vals=[]
  for st in STATIONS:
   for sd in SEEDS:
    aa=[];bb=[]
    for f in range(5):
     keys,x,y,ix=aligned(data,st,f,sd,'pxc_lead','none_lead',False)
     sel=[j for j,i in enumerate(ix) if int(keys[i]['lead_minutes'])==lead]
     aa.extend(x[sel]);bb.extend(y[sel])
    vals.append(float(np.sqrt(np.mean(aa))-np.sqrt(np.mean(bb))))
  results['leadwise_pxc_lead_minus_none_lead'].append({'lead_minutes':lead,'delta_rmse_wm2':float(np.mean(vals))})
 blocks=sorted({block_id(bjt_date(k)) for st in STATIONS for f in range(5) for k in data[(st,f,'none_no_lead',42)]['keys']})
 loo=[]
 for bl in blocks:
  loo.append({'dropped_block':bl,'delta_rmse_wm2':point_delta(data,'cot_effect_with_lead',False,bl)})
 results['leave_one_2day_block_out_min']=min(x['delta_rmse_wm2'] for x in loo)
 results['leave_one_2day_block_out_max']=max(x['delta_rmse_wm2'] for x in loo)
 results['leave_one_2day_block_out_all_negative']=all(x['delta_rmse_wm2']<0 for x in loo)
 results['leave_one_2day_block_out']=loo
 primary=results['cot_effect_with_lead']['all_leads'];short=results['cot_effect_with_lead']['plus15_60']
 seeds=[v for v in results['seed_delta_all_leads'].values()]
 stations=[v['mean'] for v in results['station_delta_all_leads'].values()]
 results['predeclared_gate']={'pass':bool(primary['delta_rmse_wm2']<0 and primary['ci95_high']<0 and short['delta_rmse_wm2']<0 and short['ci95_high']<0 and all(x<=0 for x in stations) and sum(x<0 for x in seeds)>=2 and results['leave_one_2day_block_out_all_negative']),
   'conditions':{'full_region_delta_negative':primary['delta_rmse_wm2']<0,'full_ci_upper_below_zero':primary['ci95_high']<0,
   'short_region_delta_negative':short['delta_rmse_wm2']<0,'short_ci_upper_below_zero':short['ci95_high']<0,
   'both_station_point_estimates_nonpositive':all(x<=0 for x in stations),'at_least_two_of_three_seed_directions_negative':sum(x<0 for x in seeds)>=2,
   'leave_one_2day_block_out_never_flips':results['leave_one_2day_block_out_all_negative']}}
 a.output.parent.mkdir(parents=True,exist_ok=True)
 report={'state':'COMPLETE_DIRECT_COT_LEAD_H_CV_SCORE','evidence_tier':'outer-training H-only internal date-block cross-validation; not outer validation/test',
  'run_manifest_sha256':hashlib.sha256((a.run/'run_manifest.json').read_bytes()).hexdigest(),'results':results,
  'test_payload_opened':False,'buffer_payload_opened':False,'future_observed_agri_used_as_input':False,'cpp_used_as_input':False}
 a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'state':report['state'],'primary':results['cot_effect_with_lead'],'short':short,'gate':results['predeclared_gate']},indent=2),flush=True)

if __name__=='__main__':main()
