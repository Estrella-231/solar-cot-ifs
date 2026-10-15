"""Independent FP64 scoring from saved predictions, with paired date bootstrap."""
import csv,hashlib,json
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/cot_interval_history_20261015';RUN=OUT/'results_attempt1'


def load_rows(path):return list(csv.DictReader(path.open(encoding='utf-8')))


def main():
    rows=load_rows(RUN/'validation_rows.csv');index=np.array([int(r['source_current_index']) for r in rows])
    # Saved pack indices refer to the new pack, not to the source pack indices.
    inp=np.load(RUN/'validation_inputs.npz');pack_index=inp['pack_index'];assert len(rows)==len(pack_index)
    station=np.array([r['station'] for r in rows]);dates=np.array([(datetime.fromisoformat(r['target_time_utc'])+timedelta(hours=8)).date().isoformat() for r in rows])
    udays=np.unique(dates);day=np.searchsorted(udays,dates);D=len(udays);rng=np.random.default_rng(20261015);weights=rng.multinomial(D,np.full(D,1/D),2000)
    original=load_rows(ROOT/'results/gate_clear_physics_20261015/val_labels_5min.csv');lookup={(r['station'],r['target_time_utc']):r for r in original}
    spread=np.array([np.ptp([float(v) for v in lookup[(r['station'],r['target_time_utc'])]['raw_ghi_5min_wm2'].split('|')]) for r in rows])
    y=np.full(len(rows),np.nan);pred=np.empty((3,2,len(rows)));seed_rows=[];hashes={}
    completion=json.loads((RUN/'completion.json').read_text());assert completion['state']=='COMPLETE_TWELVE_PAIRED_RUNS'
    metrics={(r['seed'],r['station'],r['arm']):r for r in completion['runs']}
    for si,seed in enumerate((42,43,44)):
        for st in ('sili','zhujia'):
            ids=np.where(station==st)[0];expected=pack_index[ids]
            for ai,arm in enumerate(('A_history','AC_history')):
                path=RUN/f'seed{seed}'/st/arm/'validation_predictions.npz';z=np.load(path)
                assert np.array_equal(z['pack_index'],expected)
                assert np.isfinite(z['prediction']).all()
                if np.isfinite(y[ids]).all():np.testing.assert_array_equal(y[ids],z['truth'])
                y[ids]=z['truth'];pred[si,ai,ids]=z['prediction']
                rmse=float(np.sqrt(np.mean((z['prediction']-z['truth'])**2)));mae=float(np.mean(np.abs(z['prediction']-z['truth'])))
                assert abs(rmse-metrics[(seed,st,arm)]['rmse'])<1e-8
                hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
                seed_rows.append(dict(seed=seed,station=st,arm=arm,n=len(ids),rmse=rmse,mae=mae))
    for r,truth in zip(rows,y):
        source=lookup[(r['station'],r['target_time_utc'])];assert abs(float(source['observed_ghi_15min_wm2'])-truth)<.0001
        t=datetime.fromisoformat(r['target_time_utc']);assert datetime.fromisoformat(r['previous_time_utc'])==t-timedelta(minutes=15)
        assert datetime.fromisoformat(r['earlier_time_utc'])==t-timedelta(minutes=30)
    zero=np.array([r['suspect_daylight_long_zero']=='1' for r in rows]);se=(pred-y[None,None])**2
    def score(mask):
        per_site=[];boot_sites=[]
        for st in ('sili','zhujia'):
            m=mask&(station==st)
            if not m.any():continue
            n=np.bincount(day[m],minlength=D);sums=np.stack([[np.bincount(day[m],weights=se[s,a,m],minlength=D) for a in range(2)] for s in range(3)])
            vals=np.sqrt(sums.sum(-1)/n.sum());per_site.append(vals)
            bn=weights@n;assert (bn>0).all();boot=np.sqrt(np.einsum('bd,sad->bsa',weights,sums)/bn[:,None,None]);boot_sites.append(boot)
        means=np.stack(per_site).mean((0,1));boot=np.stack(boot_sites).mean((0,2));bd=boot[:,1]-boot[:,0]
        delta=means[1]-means[0]
        return dict(n=int(mask.sum()),target_dates=int(len(np.unique(dates[mask]))),A_RMSE=float(means[0]),AC_RMSE=float(means[1]),delta_RMSE=float(delta),RMSE_improvement_percent=float(-100*delta/means[0]),date_cluster_CI95=np.quantile(bd,[.025,.975]).tolist())
    groups={'all_validation':np.ones(len(rows),bool),'sili':station=='sili','zhujia':station=='zhujia','exclude_suspect_zero_scoring_only':~zero,'low_interval_variation_le25':spread<=25,'high_interval_variation_gt100':spread>100}
    summary={k:score(m) for k,m in groups.items()}
    main=summary['all_validation'];deltas=[]
    for seed in (42,43,44):
        ds=[]
        for st in ('sili','zhujia'):
            rs=[r for r in seed_rows if r['seed']==seed and r['station']==st];ds.append(next(r['rmse'] for r in rs if r['arm']=='AC_history')-next(r['rmse'] for r in rs if r['arm']=='A_history'))
        deltas.append(float(np.mean(ds)))
    cases=[];mean_delta_se=(se[:,1]-se[:,0]).mean(0)
    for st in ('sili','zhujia'):
        choices=np.flatnonzero((station==st)&(y>0)&(np.array([float(r['clear']) for r in rows])>200))
        for name,idx in [('largest_nonzero_improvement',choices[np.argmin(mean_delta_se[choices])]),('largest_nonzero_deterioration',choices[np.argmax(mean_delta_se[choices])])]:
            cases.append(dict(station=st,utc=rows[idx]['target_time_utc'],category=name,validation_row_index=int(idx),truth=float(y[idx]),A_seed_predictions=pred[:,0,idx].tolist(),AC_seed_predictions=pred[:,1,idx].tolist(),mean_delta_squared_error=float(mean_delta_se[idx]),raw_GHI_5min=lookup[(st,rows[idx]['target_time_utc'])]['raw_ghi_5min_wm2']))
    (OUT/'selected_cases.json').write_text(json.dumps(cases,indent=2))
    np.savez_compressed(OUT/'paired_validation_scoring.npz',prediction=pred,truth=y,station=station,dates=dates,zero_flag=zero,interval_spread=spread,pack_index=pack_index)
    with (OUT/'seed_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(seed_rows[0]));w.writeheader();w.writerows(seed_rows)
    direction={}
    for st in ('sili','zhujia'):
        m=station==st;pa=pred[:,0,m];pc=pred[:,1,m];yy=y[m][None];change=pc-pa;under=pa<yy;over=pa>yy
        direction[st]=dict(mean_abs_GHI_adjustment=float(np.abs(change).mean()),fraction_correct_up_when_A_underestimates=float((change[under]>0).mean()),fraction_correct_down_when_A_overestimates=float((change[over]<0).mean()),fraction_samples_with_increased_squared_error=float(((pc-yy)**2>(pa-yy)**2).mean()),
            flagged_zero_share_A_squared_error=float(se[:,0,m&zero].sum()/se[:,0,m].sum()),flagged_zero_share_AC_squared_error=float(se[:,1,m&zero].sum()/se[:,1,m].sum()))
    result=dict(state='PASS_INDEPENDENT_FP64_SAVED_PREDICTION_RECOMPUTATION',groups=summary,seed_equal_station_delta_RMSE=deltas,seed_delta_std=float(np.std(deltas,ddof=1)),direction_diagnostics=direction,
        main_estimator='arithmetic mean of station-specific RMSE differences, across two stations and all three seeds',bootstrap='2000 paired target-BJT-date multinomial cluster resamples; fixed three seeds; conditional on validation checkpoint selection',
        source_zero_flagged_scored=int(zero.sum()),source_hashes=hashes,test_payloads_read=0,scientific_scope='same-time observed three-frame local AGRI reconstruction of unchanged 15min GHI, not forecasting; fixed original R bank')
    (OUT/'independent_score.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'},indent=2))


if __name__=='__main__':main()
