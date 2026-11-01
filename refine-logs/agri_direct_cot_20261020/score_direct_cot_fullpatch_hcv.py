"""Paired two-day block bootstrap for the full 16x16 forecast-COT H-CV screen."""
from __future__ import annotations
import argparse,hashlib,json
from collections import defaultdict
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
STATIONS=('sili','zhujia'); SEEDS=(42,43,44); ARMS=('none','pxc3','pxc16')
def bjt_date(k): return (datetime.fromisoformat(k['target_time_utc'])+timedelta(hours=8)).date()
def block_id(d): return f'{d.year:04d}-{d.month:02d}-B{(d.day-1)//2:02d}'
def load_all(root):
    data={}; seen={s:set() for s in STATIONS}
    for st in STATIONS:
        for fold in range(5):
            refkeys=reftruth=None
            for arm in ARMS:
                for seed in SEEDS:
                    run=root/st/f'fold{fold}'/arm/f'seed{seed}'
                    keys=[json.loads(x) for x in (run/'oof_keys.jsonl').read_text(encoding='utf-8').splitlines()]
                    with np.load(run/'oof_predictions.npz',allow_pickle=False) as z:
                        pred=z['prediction_ghi'].astype(np.float64); truth=z['truth_ghi'].astype(np.float64)
                    if len(keys)!=len(pred) or len(pred)!=len(truth): raise ValueError('prediction/key count mismatch')
                    if refkeys is not None and keys!=refkeys: raise ValueError('paired arms have different OOF keys')
                    if reftruth is not None and not np.array_equal(truth,reftruth): raise ValueError('paired arms have different truth')
                    refkeys,reftruth=keys,truth; data[(st,fold,arm,seed)]={'keys':keys,'pred':pred,'truth':truth}
            ids={(k['init_time_utc'],k['target_time_utc'],int(k['lead_minutes'])) for k in refkeys}
            if seen[st]&ids: raise ValueError(f'duplicate OOF key for {st}')
            seen[st]|=ids
    return data,seen
def aggregate(data,aa,bb,short):
    agg={}
    for st in STATIONS:
        for seed in SEEDS:
            for fold in range(5):
                x=data[(st,fold,aa,seed)]; y=data[(st,fold,bb,seed)]
                if x['keys']!=y['keys'] or not np.array_equal(x['truth'],y['truth']): raise ValueError('paired key/truth mismatch')
                local=defaultdict(lambda:[[0.0,0],[0.0,0]])
                for i,k in enumerate(x['keys']):
                    if short and int(k['lead_minutes'])>60: continue
                    b=block_id(bjt_date(k)); truth=x['truth'][i]
                    local[b][0][0]+=(x['pred'][i]-truth)**2; local[b][0][1]+=1
                    local[b][1][0]+=(y['pred'][i]-truth)**2; local[b][1][1]+=1
                for b,pair in local.items():
                    dst=agg.setdefault((st,seed,b),[[0.0,0],[0.0,0]])
                    for j in range(2): dst[j][0]+=pair[j][0]; dst[j][1]+=pair[j][1]
    return agg,sorted({k[2] for k in agg})
def delta(agg,blocks,weights):
    vals=[]
    for st in STATIONS:
        for seed in SEEDS:
            rm=[]
            for arm in range(2):
                sse=count=0.0
                for b,w in zip(blocks,weights):
                    v=agg.get((st,seed,b))
                    if w and v: sse+=v[arm][0]*w; count+=v[arm][1]*w
                if count<=0: raise ValueError('empty bootstrap support')
                rm.append(np.sqrt(sse/count))
            vals.append(rm[0]-rm[1])
    return float(np.mean(vals))
def estimate(data,aa,bb,short,reps,rng):
    agg,blocks=aggregate(data,aa,bb,short); w=np.ones(len(blocks)); point=delta(agg,blocks,w)
    draws=np.empty(reps)
    for i in range(reps): draws[i]=delta(agg,blocks,np.bincount(rng.integers(0,len(blocks),len(blocks)),minlength=len(blocks)))
    loo=[delta(agg,blocks,np.asarray([0.0 if b==drop else 1.0 for b in blocks])) for drop in blocks]
    return {'delta_rmse_wm2':point,'ci95_low':float(np.quantile(draws,.025)),'ci95_high':float(np.quantile(draws,.975)),
            'blocks':len(blocks),'bootstrap_reps':reps,'leave_one_block_min':float(min(loo)),'leave_one_block_max':float(max(loo)),
            'leave_one_block_all_negative':bool(max(loo)<0)}
def station_seed_equal(data,aa,bb):
    out={}
    for st in STATIONS:
        out[st]={}
        for name,a,b in [('candidate_minus_reference',aa,bb)]:
            seeds=[]
            for seed in SEEDS:
                p=[]; q=[]
                for fold in range(5):
                    x=data[(st,fold,a,seed)]; y=data[(st,fold,b,seed)]
                    p.extend((x['pred']-x['truth'])**2); q.extend((y['pred']-y['truth'])**2)
                seeds.append(float(np.sqrt(np.mean(p))-np.sqrt(np.mean(q))))
            out[st][name]={'seed_deltas':seeds,'mean':float(np.mean(seeds))}
    return out
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--bootstrap',type=int,default=5000); a=ap.parse_args()
    m=json.loads((a.run/'run_manifest.json').read_text(encoding='utf-8'))
    if m.get('state')!='COMPLETE_DIRECT_COT_FULLPATCH_H_CV': raise RuntimeError('complete H-only run receipt required')
    for k in ('test_payload_opened','buffer_payload_opened','future_observed_agri_used_as_input','cpp_used_as_input'):
        if m.get(k) is not False: raise RuntimeError(f'forbidden input/source flag {k}')
    data,seen=load_all(a.run); rng=np.random.default_rng(20261021); results={}
    comparisons={'pxc16_minus_pxc3':('pxc16','pxc3'),'pxc16_minus_none':('pxc16','none'),'pxc3_minus_none':('pxc3','none')}
    for name,(aa,bb) in comparisons.items():
        results[name]={}
        for scope,short in (('all_leads',False),('plus15_60',True)): results[name][scope]=estimate(data,aa,bb,short,a.bootstrap,rng)
        results[name]['station_seed_equal_all_leads']=station_seed_equal(data,aa,bb)
        seed_deltas=[]
        for seed in SEEDS:
            station_vals=[]
            for st in STATIONS:
                p=[];q=[]
                for fold in range(5):
                    x=data[(st,fold,aa,seed)]; y=data[(st,fold,bb,seed)]
                    p.extend((x['pred']-x['truth'])**2); q.extend((y['pred']-y['truth'])**2)
                station_vals.append(float(np.sqrt(np.mean(p))-np.sqrt(np.mean(q))))
            seed_deltas.append(float(np.mean(station_vals)))
        results[name]['seed_equal_region_all_leads']=seed_deltas
    primary=results['pxc16_minus_pxc3']; protection=results['pxc16_minus_none']
    station=primary['station_seed_equal_all_leads']; seed=primary['seed_equal_region_all_leads']
    conditions={'primary_all_leads_ci_below_zero':bool(primary['all_leads']['delta_rmse_wm2']<0 and primary['all_leads']['ci95_high']<0),
        'primary_short_ci_below_zero':bool(primary['plus15_60']['delta_rmse_wm2']<0 and primary['plus15_60']['ci95_high']<0),
        'fullpatch_beats_none_all_leads_ci':bool(protection['all_leads']['delta_rmse_wm2']<0 and protection['all_leads']['ci95_high']<0),
        'fullpatch_beats_none_short_ci':bool(protection['plus15_60']['delta_rmse_wm2']<0 and protection['plus15_60']['ci95_high']<0),
        'both_stations_nonworse':all(station[s]['candidate_minus_reference']['mean']<=0 for s in STATIONS),
        'at_least_2_of_3_seeds_improve':sum(x<0 for x in seed)>=2,
        'leave_one_block_stable_primary_all_and_short':bool(primary['all_leads']['leave_one_block_all_negative'] and primary['plus15_60']['leave_one_block_all_negative'])}
    results['predeclared_gate']={'pass':all(conditions.values()),'conditions':conditions}
    result={'state':'COMPLETE_DIRECT_COT_FULLPATCH_H_CV_SCORE','evidence_tier':'repeatedly used outer-training H internal CV; exploratory, not confirmation',
        'run_manifest_sha256':hashlib.sha256((a.run/'run_manifest.json').read_bytes()).hexdigest(),'comparisons':comparisons,'results':results,
        'station_rows':{s:len(seen[s]) for s in STATIONS},'bootstrap_unit':'fixed calendar two-day BJT blocks','test_payload_opened':False,'buffer_payload_opened':False,
        'future_observed_agri_used_as_input':False,'cpp_used_as_input':False}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'state':result['state'],'primary':primary,'protection':protection,'gate':results['predeclared_gate']},indent=2),flush=True)
if __name__=='__main__': main()
