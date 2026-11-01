#!/usr/bin/env python3
"""Independent common-key FP64 audit of T18 AL versus A/AC/AP."""
from __future__ import annotations
import argparse,csv,hashlib,json
from datetime import datetime,timedelta,timezone
from pathlib import Path
import numpy as np

ARMS=('A','AC','AP','AL');STATIONS=('sili','zhujia');SEEDS=(42,43,44)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<22),b''):h.update(b)
    return h.hexdigest()
def read_csv(p):
    with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def bjt_date(s):
    d=datetime.fromisoformat(s.replace('Z','+00:00'))
    if d.tzinfo is None:d=d.replace(tzinfo=timezone.utc)
    return (d.astimezone(timezone.utc)+timedelta(hours=8)).date().isoformat()
def rmse(y,p):return float(np.sqrt(np.mean((p-y)**2,dtype=np.float64)))
def main():
    a=argparse.ArgumentParser();a.add_argument('--experiment',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--bootstrap',type=int,default=2000);args=a.parse_args()
    exp=args.experiment;pack=exp/'data/ghi_head_pack_new_r_v1';pm=json.loads((pack/'pack_manifest.json').read_text())
    if pm.get('state')!='COMPLETE' or pm.get('test_payload_opened') is not False:raise ValueError('pack manifest/test gate failed')
    rows=read_csv(pack/'rows.csv')
    if any(r['split'] not in ('train','val') for r in rows):raise ValueError('unexpected split in train/val pack')
    val=[r for r in rows if r['split']=='val'];keys=[(r['station'],r['target_time_utc']) for r in val]
    if len(keys)!=len(set(keys)):raise ValueError('duplicate val key')
    y=np.asarray([float(r['ghi']) for r in val],dtype=np.float64);stations=np.asarray([r['station'] for r in val]);times=np.asarray([r['target_time_utc'] for r in val]);dates=np.asarray([bjt_date(s) for s in times])
    pred={};input_hashes={};rows_by_station={}
    for st in STATIONS:
        ix=np.flatnonzero(stations==st);rows_by_station[st]=ix
        if not len(ix):raise ValueError(f'empty station {st}')
        for arm in ARMS:
            for seed in SEEDS:
                if arm=='AL':p=exp/'results/r_al_formal_t18_20261014_v1'/f'{st}_seed{seed}'/'AL_val_prediction_ghi_wm2.npy'
                else:p=exp/'results'/f'ghi_heads_new_r_seed{seed}_v1'/st/f'{arm}_val_prediction_ghi_wm2.npy'
                meta=json.loads((p.parent/'run_metadata.json').read_text()) if (p.parent/'run_metadata.json').exists() else json.loads((p.parent/'AL_validation.json').read_text())
                if meta.get('test_payload_opened') is True:raise ValueError(f'test flag true: {p}')
                if meta.get('GHI_pack_manifest_sha256') not in (None,sha(pack/'pack_manifest.json')):raise ValueError(f'pack provenance mismatch: {p}')
                z=np.asarray(np.load(p,allow_pickle=False),dtype=np.float64)
                if z.shape!=(len(ix),):raise ValueError(f'prediction length mismatch {p}: {z.shape} != {(len(ix),)}')
                full=np.full(len(val),np.nan,np.float64);full[ix]=z
                pred[(arm,seed,st)]=full;input_hashes[str(p)]=sha(p)
    # Recompute each seed/station on identical validation keys.
    metrics={}
    for arm in ARMS:
        metrics[arm]={}
        for st in STATIONS:
            ix=rows_by_station[st];metrics[arm][st]={}
            for seed in SEEDS:
                pp=pred[(arm,seed,st)][ix];yy=y[ix]
                metrics[arm][st][str(seed)]={'n':len(ix),'RMSE_Wm2':rmse(yy,pp),'MAE_Wm2':float(np.mean(np.abs(pp-yy),dtype=np.float64)),'bias_Wm2':float(np.mean(pp-yy,dtype=np.float64))}
    station_equal={arm:float(np.mean([np.mean([metrics[arm][st][str(sd)]['RMSE_Wm2'] for sd in SEEDS]) for st in STATIONS])) for arm in ARMS}
    contrasts=[('AL','A'),('AL','AC'),('AL','AP'),('AC','A')]
    point={f'{x}-{z}':station_equal[x]-station_equal[z] for x,z in contrasts}
    date_sets={st:np.array(sorted(set(dates[stations==st]))) for st in STATIONS}
    date_rows={st:{d:np.flatnonzero((dates==d)&(stations==st)) for d in date_sets[st]} for st in STATIONS}
    rng=np.random.default_rng(20261014);boot={k:[] for k in point}
    for _ in range(args.bootstrap):
        vals={arm:[] for arm in ARMS}
        for st in STATIONS:
            draw=rng.choice(date_sets[st],size=len(date_sets[st]),replace=True)
            si=np.concatenate([date_rows[st][d] for d in draw])
            for arm in ARMS:
                vals[arm].append(float(np.mean([rmse(y[si],pred[(arm,sd,st)][si]) for sd in SEEDS])))
        se={arm:float(np.mean(vals[arm])) for arm in ARMS}
        for x,z in contrasts:boot[f'{x}-{z}'].append(se[x]-se[z])
    ci={k:[float(q) for q in np.quantile(v,[.025,.975])] for k,v in boot.items()}
    args.output.mkdir(parents=True,exist_ok=False)
    arrays={'ghi_truth_wm2':y,'station':stations,'target_time_utc':times,'split':np.asarray(['val']*len(val))}
    for arm in ARMS:
        for sd in SEEDS:
            aligned=np.zeros(len(val),np.float64)
            for st in STATIONS:aligned[rows_by_station[st]]=pred[(arm,sd,st)][rows_by_station[st]]
            arrays[f'{arm}_seed{sd}_ghi_wm2']=aligned
    # Keep a corresponding row key table for the saved concatenation order.
    with (args.output/'paired_val_keys.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['station','target_time_utc','BJT_date','GHI_Wm2'])
        for i,r in enumerate(val):w.writerow([r['station'],r['target_time_utc'],dates[i],y[i]])
    np.savez_compressed(args.output/'paired_predictions_fp64.npz',**arrays)
    finite=all(np.isfinite(pred[(arm,sd,st)][rows_by_station[st]]).all() for arm in ARMS for sd in SEEDS for st in STATIONS)
    if not finite:raise ValueError('non-finite prediction on its station validation support')
    result={'state':'INDEPENDENT_FP64_T18_AL_AUDIT_PASS','scope':'new monthly split validation only; AL hidden R feature versus paired A/AC/AP heads','sample_key':'(station,target_time_utc)','n_val_rows':len(val),'by_station_n':{st:len(rows_by_station[st]) for st in STATIONS},'unique_BJT_dates_by_station':{st:date_sets[st].tolist() for st in STATIONS},'split_id':pm['split_id'],'test_payload_opened':False,'all_inputs_finite_on_station_support':finite,'metrics_by_arm_station_seed':metrics,'station_equal_mean_RMSE_Wm2':station_equal,'station_equal_delta_RMSE_Wm2':point,'date_block_paired_bootstrap_95CI_Wm2':ci,'bootstrap_replicates':args.bootstrap,'bootstrap_unit':'within-station BJT target-date blocks, paired across arms and seeds; station date support resampled independently; duplicate dates retained','bootstrap_seed':20261014,'input_prediction_sha256':input_hashes,'pack_manifest_sha256':sha(pack/'pack_manifest.json'),'training_manifest_sha256':{f'{st}_seed{sd}':sha(exp/'results/r_al_formal_t18_20261014_v1'/f'{st}_seed{sd}'/'run_metadata.json') for st in STATIONS for sd in SEEDS},'limitations':'Validation-only causal representation control under one frozen R and 3x3 head; CPP is not independent truth; no test/generalization claim.'}
    (args.output/'fp64_independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    with (args.output/'metrics_by_seed.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['arm','station','seed','n','RMSE_Wm2','MAE_Wm2','bias_Wm2'])
        for arm in ARMS:
            for st in STATIONS:
                for sd in SEEDS:
                    m=metrics[arm][st][str(sd)];w.writerow([arm,st,sd,m['n'],m['RMSE_Wm2'],m['MAE_Wm2'],m['bias_Wm2']])
    print(json.dumps({'state':result['state'],'n_val_rows':len(val),'station_equal_mean_RMSE_Wm2':station_equal,'delta_RMSE':point,'CI':ci,'dates_by_station':{st:len(date_sets[st]) for st in STATIONS}},indent=2))
if __name__=='__main__':main()
