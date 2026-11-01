"""Paired date-block bootstrap for monotone-calibration H-only CV."""
from __future__ import annotations
import argparse, hashlib, json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

STATIONS=('sili','zhujia'); SEEDS=(42,43,44); ARMS=('none','pxc_raw','pxc_cal')
def bjt_date(k): return (datetime.fromisoformat(k['target_time_utc'])+timedelta(hours=8)).date()
def block_id(d): return f'{d.year:04d}-{d.month:02d}-B{(d.day-1)//2:02d}'
def load_all(root):
    data={}; seen={s:set() for s in STATIONS}
    for st in STATIONS:
        for f in range(5):
            refkeys=reftruth=None
            for arm in ARMS:
                for seed in SEEDS:
                    run=root/st/f'fold{f}'/arm/f'seed{seed}'
                    keys=[json.loads(x) for x in (run/'oof_keys.jsonl').read_text(encoding='utf-8').splitlines()]
                    with np.load(run/'oof_predictions.npz',allow_pickle=False) as z: pred=z['prediction_ghi'].astype(np.float64); truth=z['truth_ghi'].astype(np.float64)
                    if len(keys)!=len(pred) or len(keys)!=len(truth): raise ValueError('key/pred count mismatch')
                    if refkeys is not None and keys!=refkeys: raise ValueError(f'arm key mismatch {st}/{f}/{arm}/{seed}')
                    if reftruth is not None and not np.array_equal(truth,reftruth): raise ValueError(f'arm truth mismatch {st}/{f}/{arm}/{seed}')
                    refkeys,reftruth=keys,truth; data[(st,f,arm,seed)]={'keys':keys,'pred':pred,'truth':truth}
            ids={(k['init_time_utc'],k['target_time_utc'],int(k['lead_minutes'])) for k in refkeys}
            if seen[st]&ids: raise ValueError(f'duplicate OOF keys: {st}')
            seen[st]|=ids
    return data,seen
def aggregate(data,a,b,short):
    agg={}
    for st in STATIONS:
        for seed in SEEDS:
            for fold in range(5):
                x=data[(st,fold,a,seed)]; y=data[(st,fold,b,seed)]
                if x['keys']!=y['keys'] or not np.array_equal(x['truth'],y['truth']): raise ValueError('paired key/truth mismatch')
                local=defaultdict(lambda:[[0.0,0],[0.0,0]])
                for i,k in enumerate(x['keys']):
                    if short and int(k['lead_minutes'])>60: continue
                    bl=block_id(bjt_date(k)); truth=x['truth'][i]
                    local[bl][0][0]+=(x['pred'][i]-truth)**2; local[bl][0][1]+=1
                    local[bl][1][0]+=(y['pred'][i]-truth)**2; local[bl][1][1]+=1
                for bl,pair in local.items():
                    z=agg.setdefault((st,seed,bl),[[0.0,0],[0.0,0]])
                    for j in range(2): z[j][0]+=pair[j][0]; z[j][1]+=pair[j][1]
    return agg,sorted({k[2] for k in agg})
def delta(agg,blocks,weights):
    vals=[]
    for st in STATIONS:
        for seed in SEEDS:
            rm=[]
            for ai in range(2):
                sse=count=0.0
                for bl,w in zip(blocks,weights):
                    q=agg.get((st,seed,bl))
                    if w and q: sse+=q[ai][0]*w; count+=q[ai][1]*w
                if count<=0: raise ValueError('empty bootstrap support')
                rm.append(np.sqrt(sse/count))
            vals.append(rm[0]-rm[1])
    return float(np.mean(vals))
def estimate(data,a,b,short,reps,rng):
    agg,blocks=aggregate(data,a,b,short); point=delta(agg,blocks,np.ones(len(blocks))); draws=np.empty(reps)
    for i in range(reps): draws[i]=delta(agg,blocks,np.bincount(rng.integers(0,len(blocks),len(blocks)),minlength=len(blocks)))
    loo=[delta(agg,blocks,np.asarray([0. if z==drop else 1. for z in blocks])) for drop in blocks]
    return {'delta_rmse_wm2':point,'ci95_low':float(np.quantile(draws,.025)),'ci95_high':float(np.quantile(draws,.975)),'blocks':len(blocks),'bootstrap_reps':reps,'leave_one_block_min':float(min(loo)),'leave_one_block_max':float(max(loo)),'leave_one_block_all_negative':bool(max(loo)<0)}
def point_metrics(data,arm,short=False,station=None):
    vals=[]
    for st in STATIONS:
        if station is not None and st!=station: continue
        for seed in SEEDS:
            y=[]; p=[]
            for f in range(5):
                q=data[(st,f,arm,seed)]; ix=[i for i,k in enumerate(q['keys']) if not short or int(k['lead_minutes'])<=60]
                y.extend(q['truth'][ix]); p.extend(q['pred'][ix])
            y=np.asarray(y); p=np.asarray(p); vals.append((np.sqrt(np.mean((p-y)**2)),np.mean(np.abs(p-y)),np.mean(p-y)))
    return {'rmse_wm2':float(np.mean([v[0] for v in vals])),'mae_wm2':float(np.mean([v[1] for v in vals])),'bias_wm2':float(np.mean([v[2] for v in vals])),'station_seed_equal':True}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--bootstrap',type=int,default=5000); a=p.parse_args()
    m=json.loads((a.run/'run_manifest.json').read_text(encoding='utf-8'))
    if m.get('state')!='COMPLETE_COT_MONOCAL_H_CV' or m.get('arms')!=list(ARMS): raise RuntimeError('complete monocal H-CV receipt required')
    for k in ('test_payload_opened','buffer_payload_opened','future_observed_agri_used_as_input','cpp_used_as_input'):
        if m.get(k) is not False: raise RuntimeError(f'forbidden input flag: {k}')
    data,seen=load_all(a.run); rng=np.random.default_rng(20261021); comps={'cal_minus_raw':('pxc_cal','pxc_raw'),'cal_minus_none':('pxc_cal','none'),'raw_minus_none':('pxc_raw','none')}; results={}
    for name,(x,y) in comps.items():
        results[name]={}
        for scope,short in (('all_leads',False),('plus15_60',True)): results[name][scope]=estimate(data,x,y,short,a.bootstrap,rng)
    results['station_seed_equal_all_leads']={}; results['seed_equal_region_all_leads']={}
    for st in STATIONS:
        results['station_seed_equal_all_leads'][st]={}
        for name,(x,y) in comps.items():
            sd=[]
            for seed in SEEDS:
                se=[]; te=[]
                for f in range(5):
                    q=data[(st,f,x,seed)]; r=data[(st,f,y,seed)]; se.extend((q['pred']-q['truth'])**2); te.extend((r['pred']-r['truth'])**2)
                sd.append(float(np.sqrt(np.mean(se))-np.sqrt(np.mean(te))))
            results['station_seed_equal_all_leads'][st][name]={'seed_deltas':sd,'mean':float(np.mean(sd))}
    for name,(x,y) in comps.items():
        vals=[]
        for seed in SEEDS:
            sts=[]
            for st in STATIONS:
                se=[]; te=[]
                for f in range(5):
                    q=data[(st,f,x,seed)]; r=data[(st,f,y,seed)]; se.extend((q['pred']-q['truth'])**2); te.extend((r['pred']-r['truth'])**2)
                sts.append(float(np.sqrt(np.mean(se))-np.sqrt(np.mean(te))))
            vals.append(float(np.mean(sts)))
        results['seed_equal_region_all_leads'][name]=vals
    results['absolute_metrics']={}
    for arm in ARMS:
        results['absolute_metrics'][arm]={'all_leads':point_metrics(data,arm,False),'plus15_60':point_metrics(data,arm,True),
          'stations':{st:{'all_leads':point_metrics(data,arm,False,st),'plus15_60':point_metrics(data,arm,True,st)} for st in STATIONS}}
    results['leadwise_cal_minus_raw']=[]
    for lead in range(15,241,15):
        d=[]
        for st in STATIONS:
            for seed in SEEDS:
                p=[]; r=[]; y=[]
                for f in range(5):
                    x=data[(st,f,'pxc_cal',seed)]; z=data[(st,f,'pxc_raw',seed)]
                    ix=[i for i,k in enumerate(x['keys']) if int(k['lead_minutes'])==lead]
                    p.extend(x['pred'][ix]); r.extend(z['pred'][ix]); y.extend(x['truth'][ix])
                d.append(float(np.sqrt(np.mean((np.asarray(p)-y)**2))-np.sqrt(np.mean((np.asarray(r)-y)**2))))
        results['leadwise_cal_minus_raw'].append({'lead_minutes':lead,'delta_rmse_wm2':float(np.mean(d))})
    slope_runs=[]
    for st in STATIONS:
        for f in range(5):
            for seed in SEEDS:
                q=json.loads((a.run/st/f'fold{f}'/'pxc_cal'/f'seed{seed}'/'training_curve.json').read_text(encoding='utf-8'))[-1]
                slope_runs.append(q['slopes'])
    slopes=np.asarray(slope_runs,dtype=np.float64)
    results['learned_slope_summary']={'knots_log1p':[0.0,.5,1.0,2.0,3.0,4.6],'mean':slopes.mean(0).tolist(),'sd':slopes.std(0,ddof=1).tolist(),'n_models':int(len(slopes))}
    checks={}
    for name in ('cal_minus_raw','cal_minus_none'):
        full=results[name]['all_leads']; short=results[name]['plus15_60']; seeds=results['seed_equal_region_all_leads'][name]
        checks[name]={'all_lead_point_and_ci_pass':bool(full['delta_rmse_wm2']<0 and full['ci95_high']<0),'short_point_and_ci_pass':bool(short['delta_rmse_wm2']<0 and short['ci95_high']<0),'both_stations_nonworse':all(results['station_seed_equal_all_leads'][s][name]['mean']<=0 for s in STATIONS),'at_least_2_of_3_seed_deltas_negative':sum(x<0 for x in seeds)>=2,'leave_one_block_stable_all_and_short':bool(full['leave_one_block_all_negative'] and short['leave_one_block_all_negative'])}
    result={'state':'COMPLETE_COT_MONOCAL_H_CV_SCORE','evidence_tier':'outer-training H-only internal date-block CV; exploratory screening only, not independent validation/test','run_manifest_sha256':hashlib.sha256((a.run/'run_manifest.json').read_bytes()).hexdigest(),'results':results,'predeclared_gate':{'pass':all(all(v.values()) for v in checks.values()),'comparisons':checks},'station_rows':{s:len(seen[s]) for s in STATIONS},'test_payload_opened':False,'buffer_payload_opened':False,'future_observed_agri_used_as_input':False,'cpp_used_as_input':False}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'state':result['state'],'cal_minus_raw':results['cal_minus_raw'],'cal_minus_none':results['cal_minus_none'],'gate':result['predeclared_gate']},indent=2),flush=True)
if __name__=='__main__': main()
