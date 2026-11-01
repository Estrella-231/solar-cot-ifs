"""Paired calendar-two-day bootstrap for the solar-path COT encoding H-CV."""
from __future__ import annotations
import argparse, hashlib, json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

STATIONS=('sili','zhujia'); SEEDS=(42,43,44); ARMS=('none','raw','transmission')

def bjt_date(k): return (datetime.fromisoformat(k['target_time_utc'])+timedelta(hours=8)).date()
def block_id(d): return f'{d.year:04d}-{d.month:02d}-B{(d.day-1)//2:02d}'

def load_all(root):
    data={}; seen={s:set() for s in STATIONS}
    for st in STATIONS:
        for f in range(5):
            refkeys=None; reftruth=None
            for arm in ARMS:
                for seed in SEEDS:
                    run=root/st/f'fold{f}'/arm/f'seed{seed}'
                    keys=[json.loads(x) for x in (run/'oof_keys.jsonl').read_text(encoding='utf-8').splitlines()]
                    with np.load(run/'oof_predictions.npz',allow_pickle=False) as z:
                        pred=z['prediction_ghi'].astype(np.float64); truth=z['truth_ghi'].astype(np.float64)
                    if len(keys)!=len(pred) or len(keys)!=len(truth): raise ValueError(f'key/pred count mismatch {st}/{f}/{arm}/{seed}')
                    if refkeys is not None and keys!=refkeys: raise ValueError(f'arm key mismatch {st}/{f}/{arm}/{seed}')
                    if reftruth is not None and not np.array_equal(truth,reftruth): raise ValueError(f'arm truth mismatch {st}/{f}/{arm}/{seed}')
                    refkeys,reftruth=keys,truth
                    data[(st,f,arm,seed)]={'keys':keys,'pred':pred,'truth':truth}
            ids={(k['init_time_utc'],k['target_time_utc'],int(k['lead_minutes'])) for k in refkeys}
            if seen[st]&ids: raise ValueError(f'OOF key duplicated across folds for {st}')
            seen[st]|=ids
    return data,seen

def aggregate(data,arm_a,arm_b,short):
    agg={}
    for st in STATIONS:
        for seed in SEEDS:
            for fold in range(5):
                a=data[(st,fold,arm_a,seed)]; b=data[(st,fold,arm_b,seed)]
                if a['keys']!=b['keys'] or not np.array_equal(a['truth'],b['truth']): raise ValueError('paired input mismatch')
                local=defaultdict(lambda:[[0.0,0],[0.0,0]])
                for i,k in enumerate(a['keys']):
                    if short and int(k['lead_minutes'])>60: continue
                    bl=block_id(bjt_date(k)); y=a['truth'][i]
                    local[bl][0][0]+=(a['pred'][i]-y)**2; local[bl][0][1]+=1
                    local[bl][1][0]+=(b['pred'][i]-y)**2; local[bl][1][1]+=1
                for bl,pair in local.items():
                    z=agg.setdefault((st,seed,bl),[[0.0,0],[0.0,0]])
                    for j in range(2): z[j][0]+=pair[j][0]; z[j][1]+=pair[j][1]
    blocks=sorted({k[2] for k in agg})
    return agg,blocks

def delta(agg,blocks,weights):
    values=[]
    for st in STATIONS:
        for seed in SEEDS:
            pair=[]
            for armidx in range(2):
                sse=count=0.0
                for bl,w in zip(blocks,weights):
                    z=agg.get((st,seed,bl))
                    if w and z: sse+=z[armidx][0]*w; count+=z[armidx][1]*w
                if count<=0: raise ValueError('empty bootstrap station/seed support')
                pair.append(np.sqrt(sse/count))
            values.append(pair[0]-pair[1])
    return float(np.mean(values))

def estimate(data,a,b,short,reps,rng):
    agg,blocks=aggregate(data,a,b,short); w=np.ones(len(blocks))
    point=delta(agg,blocks,w); draws=np.empty(reps,dtype=np.float64)
    for i in range(reps): draws[i]=delta(agg,blocks,np.bincount(rng.integers(0,len(blocks),len(blocks)),minlength=len(blocks)))
    loo=[delta(agg,blocks,np.asarray([0.0 if x==drop else 1.0 for x in blocks])) for drop in blocks]
    return {'delta_rmse_wm2':point,'ci95_low':float(np.quantile(draws,.025)),'ci95_high':float(np.quantile(draws,.975)),
            'blocks':len(blocks),'bootstrap_reps':reps,'leave_one_block_min':float(min(loo)),'leave_one_block_max':float(max(loo)),
            'leave_one_block_all_negative':bool(max(loo)<0)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--bootstrap',type=int,default=5000); a=ap.parse_args()
    m=json.loads((a.run/'run_manifest.json').read_text(encoding='utf-8'))
    if m.get('state')!='COMPLETE_COT_SOLAR_PATH_H_CV': raise RuntimeError('complete H-only CV receipt required')
    for k in ('test_payload_opened','buffer_payload_opened','future_observed_agri_used_as_input','cpp_used_as_input'):
        if m.get(k) is not False: raise RuntimeError(f'forbidden source flag: {k}')
    data,seen=load_all(a.run); rng=np.random.default_rng(20261021); results={}
    comps={'transmission_minus_raw':('transmission','raw'),'transmission_minus_none':('transmission','none'),'raw_minus_none':('raw','none')}
    for name,(aa,bb) in comps.items():
        results[name]={}
        for scope,short in (('all_leads',False),('plus15_60',True)):
            results[name][scope]=estimate(data,aa,bb,short,a.bootstrap,rng)
    results['station_seed_equal_all_leads']={}
    for st in STATIONS:
        results['station_seed_equal_all_leads'][st]={}
        for name,(aa,bb) in comps.items():
            pa=[]
            for seed in SEEDS:
                se=[]; te=[]
                for fold in range(5):
                    x=data[(st,fold,aa,seed)]; y=data[(st,fold,bb,seed)]
                    if x['keys']!=y['keys']: raise ValueError('paired key mismatch')
                    se.extend((x['pred']-x['truth'])**2); te.extend((y['pred']-y['truth'])**2)
                pa.append(float(np.sqrt(np.mean(se))-np.sqrt(np.mean(te))))
            results['station_seed_equal_all_leads'][st][name]={'seed_deltas':pa,'mean':float(np.mean(pa))}
    results['seed_equal_region_all_leads']={}
    for name,(aa,bb) in comps.items():
        sd=[]
        for seed in SEEDS:
            vals=[]
            for st in STATIONS:
                sq=[]; tq=[]
                for fold in range(5):
                    x=data[(st,fold,aa,seed)]; y=data[(st,fold,bb,seed)]
                    sq.extend((x['pred']-x['truth'])**2); tq.extend((y['pred']-y['truth'])**2)
                vals.append(float(np.sqrt(np.mean(sq))-np.sqrt(np.mean(tq))))
            sd.append(float(np.mean(vals)))
        results['seed_equal_region_all_leads'][name]=sd
    results['leadwise_descriptive_transmission_minus_raw']=[]
    for lead in range(15,241,15):
        v=[]
        for st in STATIONS:
            for seed in SEEDS:
                sq=[]; tq=[]
                for fold in range(5):
                    x=data[(st,fold,'transmission',seed)]; y=data[(st,fold,'raw',seed)]
                    ix=[i for i,k in enumerate(x['keys']) if int(k['lead_minutes'])==lead]
                    sq.extend((x['pred'][ix]-x['truth'][ix])**2); tq.extend((y['pred'][ix]-y['truth'][ix])**2)
                v.append(float(np.sqrt(np.mean(sq))-np.sqrt(np.mean(tq))))
        results['leadwise_descriptive_transmission_minus_raw'].append({'lead_minutes':lead,'delta_rmse_wm2':float(np.mean(v))})
    p=results['transmission_minus_raw']; s=results['transmission_minus_raw']['plus15_60']; station=results['station_seed_equal_all_leads']
    seedvals=results['seed_equal_region_all_leads']['transmission_minus_raw']
    results['predeclared_gate']={'pass':bool(p['all_leads']['delta_rmse_wm2']<0 and p['all_leads']['ci95_high']<0 and s['delta_rmse_wm2']<0 and s['ci95_high']<0 and all(station[st]['transmission_minus_raw']['mean']<=0 for st in STATIONS) and sum(x<0 for x in seedvals)>=2 and p['all_leads']['leave_one_block_all_negative'] and s['leave_one_block_all_negative']),
        'conditions':{'all_lead_point_and_ci_pass':bool(p['all_leads']['delta_rmse_wm2']<0 and p['all_leads']['ci95_high']<0),'short_point_and_ci_pass':bool(s['delta_rmse_wm2']<0 and s['ci95_high']<0),'both_stations_nonworse':all(station[st]['transmission_minus_raw']['mean']<=0 for st in STATIONS),'at_least_2_of_3_seed_deltas_negative':sum(x<0 for x in seedvals)>=2,'leave_one_block_stable_all_and_short':bool(p['all_leads']['leave_one_block_all_negative'] and s['leave_one_block_all_negative'])}}
    result={'state':'COMPLETE_COT_SOLAR_PATH_H_CV_SCORE','evidence_tier':'outer-training H-only internal date-block CV; not outer validation/test','run_manifest_sha256':hashlib.sha256((a.run/'run_manifest.json').read_bytes()).hexdigest(),'results':results,'station_rows':{st:len(seen[st]) for st in STATIONS},'test_payload_opened':False,'buffer_payload_opened':False,'future_observed_agri_used_as_input':False,'cpp_used_as_input':False}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'state':result['state'],'primary':results['transmission_minus_raw'],'gate':results['predeclared_gate']},indent=2),flush=True)

if __name__=='__main__': main()
