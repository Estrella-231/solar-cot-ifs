"""Independent FP64 paired scoring; group labels are never model inputs."""
import argparse,csv,hashlib,json
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np

ARMS=('A_uniform','A_bright2x','AC_uniform','AC_bright2x')
CONTRASTS=((1,0),(3,2),(2,0),(3,1))
SEEDS=(42,43,44)
STATIONS=('sili','zhujia')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    here=Path(__file__).resolve().parent;repo=here.parents[1]
    rows=list(csv.DictReader((a.run/'validation_rows.csv').open(encoding='utf-8',newline='')))
    done=json.loads((a.run/'completion.json').read_text());assert done['state']=='COMPLETE_24_LOSS_PAIRED_RUNS' and len(done['runs'])==24 and done['test_payloads_read']==0
    setup=json.loads((a.run/'setup.json').read_text());assert setup['parameters_each_arm']==41665 and setup['batch']==1792 and setup['epochs']==30
    assert len(rows)==4263 and {r['split'] for r in rows}=={'val'}
    station=np.array([r['station'] for r in rows]);times=np.array([r['target_time_utc'] for r in rows])
    dates=np.array([(datetime.fromisoformat(t)+timedelta(hours=8)).date().isoformat() for t in times]);days=np.unique(dates);di=np.searchsorted(days,dates);D=len(days)
    pred=np.full((3,4,len(rows)),np.nan);truth=np.full(len(rows),np.nan);pack_index=np.full(len(rows),-1,np.int64);recomputed=[]
    for si,seed in enumerate(SEEDS):
        for st in STATIONS:
            ids=np.flatnonzero(station==st)
            for ai,arm in enumerate(ARMS):
                f=a.run/f'seed{seed}'/st/arm/'validation_predictions.npz'
                with np.load(f,allow_pickle=False) as z:
                    p=z['prediction'].astype('f8');y=z['truth'].astype('f8');ix=z['pack_index']
                    assert np.isfinite(p).all() and np.isfinite(y).all()
                    assert np.max(np.abs(p-z['cpu_prediction']))<.1
                    if np.isfinite(truth[ids]).all():np.testing.assert_array_equal(truth[ids],y);np.testing.assert_array_equal(pack_index[ids],ix)
                    truth[ids]=y;pred[si,ai,ids]=p;pack_index[ids]=ix
                error=p-y;m=json.loads((f.parent/'metrics.json').read_text())
                metrics=dict(RMSE=float(np.sqrt(np.mean(error**2))),MAE=float(np.mean(np.abs(error))),bias=float(error.mean()))
                assert abs(metrics['RMSE']-m['rmse'])<1e-8 and abs(metrics['MAE']-m['mae'])<1e-8 and abs(metrics['bias']-m['mbe'])<1e-8
                recomputed.append(dict(seed=seed,station=st,arm=arm,n=len(ids),**metrics))
    assert np.isfinite(pred).all()
    np.testing.assert_array_equal(truth,np.array([float(np.float32(float(r['ghi']))) for r in rows]))
    clear=np.array([float(r['clear']) for r in rows]);kt=np.array([float(r['kt']) for r in rows]);suspect=np.array([r['suspect_daylight_long_zero']=='1' for r in rows])
    cot=np.expm1(np.asarray(np.load(here/'data/bank/cot_corrected_log1p.npy',allow_pickle=False,mmap_mode='r')[pack_index,2,0,1,1],np.float64))
    bright=(clear>=100)&(kt>=.9)
    groups={'all':np.ones(len(rows),bool),'high_transmittance':bright,'high_GHI_800':truth>=800,
            'bright_cloudy':bright&(cot>=3),'low_kt':(clear>=100)&(kt<.3),
            'middle_kt':(clear>=100)&(kt>=.3)&(kt<.9),'without_suspect_zero':~suspect}
    e=pred-truth[None,None,:]
    # All arms and seeds share each BJT-date draw, including across stations.
    rng=np.random.default_rng(20261016);w=rng.multinomial(D,np.full(D,1/D),2000)
    def summarize(mask):
        by_station={};point=[];boot=[]
        for st in STATIONS:
            ids=np.flatnonzero(mask&(station==st));assert len(ids)>0
            err=e[:,:,ids];rm=np.sqrt(np.mean(err**2,axis=-1));point.append(rm)
            by_station[st]={arm:dict(n=len(ids),RMSE=float(rm[:,ai].mean()),MAE=float(np.abs(err[:,ai]).mean()),bias=float(err[:,ai].mean())) for ai,arm in enumerate(ARMS)}
            counts=np.bincount(di[ids],minlength=D)
            sse=np.array([[np.bincount(di[ids],weights=err[si,ai]**2,minlength=D) for ai in range(4)] for si in range(3)])
            denom=w@counts;assert np.all(denom>0)
            boot.append(np.sqrt(np.einsum('bd,sad->bsa',w,sse)/denom[:,None,None]))
        p=np.mean(point,axis=(0,1));b=np.mean(boot,axis=(0,2));contrasts={}
        for ci,ri in CONTRASTS:
            diff=b[:,ci]-b[:,ri]
            contrasts[ARMS[ci]+' - '+ARMS[ri]]=dict(delta_RMSE=float(p[ci]-p[ri]),relative_percent=float(100*(p[ci]-p[ri])/p[ri]),CI95=np.quantile(diff,[.025,.975]).tolist())
        return dict(n=int(mask.sum()),dates=int(len(np.unique(dates[mask]))),RMSE={arm:float(p[ai]) for ai,arm in enumerate(ARMS)},
                    MAE={arm:float(np.mean([by_station[st][arm]['MAE'] for st in STATIONS])) for arm in ARMS},
                    bias={arm:float(np.mean([by_station[st][arm]['bias'] for st in STATIONS])) for arm in ARMS},per_station=by_station,contrasts=contrasts)
    summaries={name:summarize(mask) for name,mask in groups.items()}
    direction={}
    for scope,mask in [('all',groups['all']),('high_transmittance',bright)]:
        under=(pred[:,0]<truth[None,:])&mask[None,:]
        direction[scope]={arm:dict(A_uniform_under_seed_rows=int(under.sum()),upward_fraction=float((pred[:,ai][under]>pred[:,0][under]).mean()),smaller_SSE_fraction=float((e[:,ai][under]**2<e[:,0][under]**2).mean())) for ai,arm in enumerate(ARMS[1:],start=1)}
    within_loss={}
    for ci,ri in [(2,0),(3,1)]:
        mask=pred[:,ri]<truth[None,:]
        within_loss[ARMS[ci]+' - '+ARMS[ri]]=dict(under_seed_rows=int(mask.sum()),upward_fraction=float((pred[:,ci][mask]>pred[:,ri][mask]).mean()),smaller_SSE_fraction=float((e[:,ci][mask]**2<e[:,ri][mask]**2).mean()))
    # Fixed previously selected failures are a regression check, not a selection set.
    diagnostic=repo/'experiments/cot_underestimate_forensics_20261016/results'
    prior=list(csv.DictReader((diagnostic/'row_diagnostics.csv').open(encoding='utf-8',newline='')))
    assert len(prior)==len(rows)
    severe=np.array([r['severe_under_drop_3seed']=='1' for r in prior]);assert severe.sum()==254
    summaries['prior_254_failures']=summarize(severe)
    cases=json.loads((diagnostic/'selected_cases.json').read_text(encoding='utf-8'))['cases'];case_rows=[]
    for case in cases:
        if case['role']!='severe_under_drop':continue
        j=case['validation_index'];assert rows[j]['station']==case['station'] and rows[j]['target_time_utc']==case['utc'] and truth[j]==case['ghi']
        case_rows.append(dict(station=case['station'],bjt=case['bjt'],truth=float(truth[j]),**{arm:float(pred[:,ai,j].mean()) for ai,arm in enumerate(ARMS)}))
    reference=repo/'experiments/r_geometry_downstream_ghi_20261016/results/paired_three_arm_v1/analysis/paired_validation_scoring.npz'
    with np.load(reference,allow_pickle=False) as old:
        np.testing.assert_array_equal(old['truth'],truth);np.testing.assert_array_equal(old['pack_index'],pack_index);np.testing.assert_array_equal(old['station'],station)
        replay={arm:dict(mean_abs_prediction_difference=float(np.abs(pred[:,ai]-old['prediction'][:,oi]).mean()),max_abs_prediction_difference=float(np.abs(pred[:,ai]-old['prediction'][:,oi]).max())) for arm,ai,oi in [('A_uniform',0,0),('AC_uniform',2,2)]}
    decisions={}
    for ci,ri in [(1,0),(3,2)]:
        cand,ctrl=ARMS[ci],ARMS[ri];h=summaries['high_transmittance'];all_=summaries['all'];low=summaries['low_kt'];key=cand+' - '+ctrl
        checks=dict(high_group_RMSE_decreases=h['RMSE'][cand]<h['RMSE'][ctrl],
                    high_group_absolute_bias_decreases=abs(h['bias'][cand])<abs(h['bias'][ctrl]),
                    all_validation_RMSE_does_not_increase=all_['RMSE'][cand]<=all_['RMSE'][ctrl],
                    each_station_overall_regression_at_most_1=all(all_['per_station'][st][cand]['RMSE']-all_['per_station'][st][ctrl]['RMSE']<=1 for st in STATIONS),
                    low_kt_regression_at_most_1=low['RMSE'][cand]-low['RMSE'][ctrl]<=1)
        decisions[key]=dict(checks=checks,point_gate_pass=all(checks.values()),high_group_date_CI_upper_below_zero=h['contrasts'][key]['CI95'][1]<0)
    a.output.mkdir(exist_ok=False,parents=True)
    np.savez_compressed(a.output/'paired_validation_scoring.npz',prediction=pred,truth=truth,station=station,target_time_utc=times,dates=dates,pack_index=pack_index,clear=clear,kt=kt,cot_center=cot,prior_severe=severe)
    for name,data in [('per_seed_metrics.csv',recomputed),('six_failure_cases.csv',case_rows)]:
        with (a.output/name).open('w',encoding='utf-8',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=list(data[0]));wr.writeheader();wr.writerows(data)
    summary=dict(state='PASS_INDEPENDENT_FP64_COMMON_SUPPORT',arms=list(ARMS),validation_rows=len(rows),dates=D,groups=summaries,
                 direction_from_fixed_A_uniform=direction,COT_direction_within_each_loss=within_loss,point_gates=decisions,
                 old_hardware_control_replay=replay,train_weight_receipt=setup['train_weight_receipt'],bootstrap_replicates=2000,
                 bootstrap_unit='paired BJT target dates; seed mean within station; equal stations',
                 test_payloads_read=0,selection_scope='reused validation selected checkpoints; not independent test',
                 completion_sha256=sha(a.run/'completion.json'),trainer_sha256=sha(here/'train_loss_pair.py'))
    (a.output/'independent_score.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(groups={name:{k:v for k,v in result.items() if k in ('n','RMSE','bias','contrasts')} for name,result in summaries.items()},point_gates=decisions),indent=2))

if __name__=='__main__':main()
