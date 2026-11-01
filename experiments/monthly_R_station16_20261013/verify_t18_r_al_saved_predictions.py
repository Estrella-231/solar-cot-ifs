#!/usr/bin/env python3
"""Independent local recomputation from the delivered T18 FP64 prediction bank."""
import argparse,csv,json
from pathlib import Path
import numpy as np
ARMS=('A','AC','AP','AL');STATIONS=('sili','zhujia');SEEDS=(42,43,44)
def rmse(y,p):return float(np.sqrt(np.mean((np.asarray(p,dtype=np.float64)-np.asarray(y,dtype=np.float64))**2,dtype=np.float64)))
def main():
    p=argparse.ArgumentParser();p.add_argument('--audit-dir',type=Path,required=True);a=p.parse_args()
    d=a.audit_dir;src=json.loads((d/'fp64_independent_audit.json').read_text());z=np.load(d/'paired_predictions_fp64.npz',allow_pickle=False)
    with (d/'paired_val_keys.csv').open(newline='',encoding='utf-8') as f:keys=list(csv.DictReader(f))
    y=np.asarray(z['ghi_truth_wm2'],np.float64);st=np.asarray(z['station']).astype(str);dates=np.asarray([r['BJT_date'] for r in keys]);
    if len(y)!=len(keys) or not np.array_equal(st,np.asarray([r['station'] for r in keys])):raise ValueError('saved array/key alignment mismatch')
    pred={(arm,seed):np.asarray(z[f'{arm}_seed{seed}_ghi_wm2'],np.float64) for arm in ARMS for seed in SEEDS}
    if not all(np.isfinite(v).all() for v in pred.values()):raise ValueError('non-finite saved prediction')
    metrics={}
    for arm in ARMS:
        metrics[arm]={}
        for station in STATIONS:
            ix=np.flatnonzero(st==station);metrics[arm][station]={}
            for seed in SEEDS:metrics[arm][station][str(seed)]={'n':len(ix),'RMSE_Wm2':rmse(y[ix],pred[(arm,seed)][ix])}
    station_equal={arm:float(np.mean([np.mean([metrics[arm][s][str(seed)]['RMSE_Wm2'] for seed in SEEDS]) for s in STATIONS])) for arm in ARMS}
    contrasts=(('AL','A'),('AL','AC'),('AL','AP'),('AC','A'));point={f'{x}-{b}':station_equal[x]-station_equal[b] for x,b in contrasts}
    ds={s:np.array(sorted(set(dates[st==s]))) for s in STATIONS};dr={s:{day:np.flatnonzero((dates==day)&(st==s)) for day in ds[s]} for s in STATIONS}
    rng=np.random.default_rng(20261014);boot={f'{x}-{b}':[] for x,b in contrasts}
    for _ in range(2000):
        vals={arm:[] for arm in ARMS}
        for station in STATIONS:
            draw=rng.choice(ds[station],size=len(ds[station]),replace=True);ix=np.concatenate([dr[station][day] for day in draw])
            for arm in ARMS:vals[arm].append(float(np.mean([rmse(y[ix],pred[(arm,seed)][ix]) for seed in SEEDS])))
        agg={arm:float(np.mean(vals[arm])) for arm in ARMS}
        for x,b in contrasts:boot[f'{x}-{b}'].append(agg[x]-agg[b])
    ci={name:[float(v) for v in np.quantile(samples,[.025,.975])] for name,samples in boot.items()}
    errors={'station_equal_rmse':max(abs(station_equal[k]-src['station_equal_mean_RMSE_Wm2'][k]) for k in ARMS),
            'delta':max(abs(point[k]-src['station_equal_delta_RMSE_Wm2'][k]) for k in point),
            'ci':max(abs(ci[k][j]-src['date_block_paired_bootstrap_95CI_Wm2'][k][j]) for k in ci for j in (0,1))}
    if max(errors.values())>1e-10:raise ValueError(f'FP64 re-computation mismatch: {errors}')
    out={'state':'PASS_LOCAL_INDEPENDENT_FP64_RECOMPUTE','errors_max_abs':errors,'station_equal_mean_RMSE_Wm2':station_equal,'station_equal_delta_RMSE_Wm2':point,'date_block_paired_bootstrap_95CI_Wm2':ci,'n_rows':len(y),'test_payload_opened':False}
    (d/'local_independent_recompute.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
