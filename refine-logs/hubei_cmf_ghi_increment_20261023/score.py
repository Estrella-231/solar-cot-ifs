"""FP64 date-paired GHI screen, only the saved C_select predictions."""
import argparse,json
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
out=a.root/'scores_attempt1';assert not out.exists()
arms=['A','H','F','M','V'];sites=['Niushou','Wangzhai','Mizhuang'];seeds=[42,43,44]
records=[];day_stats={};hourly=[]
for si,st in enumerate(sites):
    folder=a.root/'screen_attempt1'/st
    assert json.loads((folder/'COMPLETE.json').read_text())['state']=='C_DATE_CMF_FIVE_ARM_STATION_COMPLETE'
    prior=None
    for sj,seed in enumerate(seeds):
        with np.load(folder/f'seed{seed}_Cselect.npz') as z:data={k:z[k] for k in z.files}
        assert data['arms'].tolist()==arms and data['prediction'].shape==(5,len(data['truth']))
        keys=list(zip(data['init_time'],data['target_time'],data['lead']));assert len(keys)==len(set(keys))
        if prior is not None:
            assert keys==prior[0] and np.array_equal(data['truth'],prior[1]) and np.array_equal(data['clear'],prior[2])
        else:prior=keys,data['truth'],data['clear']
        pred=data['prediction'];truth=data['truth'];dates=data['date'];leads=data['lead'];assert np.isfinite(pred).all()
        for j,name in enumerate(arms):
            e=pred[j]-truth;high=truth>=800
            records.append(dict(station=st,seed=seed,arm=name,n=len(e),rmse=float(np.sqrt(np.mean(e*e))),mae=float(np.mean(abs(e))),bias=float(e.mean()),high_ghi_n=int(high.sum()),high_ghi_bias=float(e[high].mean()) if high.any() else None))
            for day in sorted(set(dates)):
                day_stats.setdefault(day,np.zeros((3,3,5,2),np.float64));ee=e[dates==day];day_stats[day][si,sj,j]=[float(ee@ee),len(ee)]
            for hour in range(1,5):
                m=(leads>(hour-1)*60)&(leads<=hour*60)
                if m.any():hourly.append(dict(station=st,seed=seed,arm=name,hour=hour,n=int(m.sum()),rmse=float(np.sqrt(np.mean(e[m]**2))),bias=float(e[m].mean())))
days=sorted(day_stats);daily=np.array([day_stats[d] for d in days]);tot=daily.sum(0);points=np.sqrt(tot[...,0]/tot[...,1]).mean((0,1))
w=np.random.default_rng(20261023).multinomial(len(days),np.ones(len(days))/len(days),5000);den=w@daily[...,1].reshape(len(days),-1);assert np.all(den>0)
boot=np.sqrt((w@daily[...,0].reshape(len(days),-1))/den).reshape(5000,3,3,5).mean((1,2))
comparisons={}
for j,k in [(3,0),(3,2),(3,4),(3,1),(4,0),(2,0)]:
    delta=float(points[j]-points[k]);ci=np.quantile(boot[:,j]-boot[:,k],[.025,.975]);site=[];improving=0;high=[]
    for st in sites:
        r=[np.mean([v['rmse'] for v in records if v['station']==st and v['arm']==name]) for name in arms];site.append(100*(r[j]/r[k]-1))
    for seed in seeds:
        r=[np.mean([v['rmse'] for v in records if v['seed']==seed and v['arm']==name]) for name in arms];improving+=r[j]<r[k]
    for st in sites:
        for seed in seeds:
            r=[next(v['high_ghi_bias'] for v in records if v['station']==st and v['seed']==seed and v['arm']==name) for name in arms]
            if r[j] is not None and r[k] is not None:high.append(abs(r[j])-abs(r[k]))
    comparisons[f'{arms[j]}_minus_{arms[k]}']=dict(delta_rmse=delta,ci95=ci.tolist(),site_change_percent=site,improving_seeds=int(improving),mean_abs_high_bias_increase=float(np.mean(high)),
        passes=bool(delta<0 and ci[1]<0 and max(site)<=1 and improving>=2 and np.mean(high)<=5))
passed=all(comparisons[c]['passes'] for c in ['M_minus_A','M_minus_F','M_minus_V'])
report=dict(state='CSELECT_CMF_SCREEN_COMPLETE',arms=dict(zip(arms,points.tolist())),comparisons=comparisons,records=records,hourly=hourly,dates=days,
    passes_future_cot_screen=passed,decision='freeze_then_consider_development_validation' if passed else 'close_encoding_candidate_no_outer_validation_or_hyperparameter_search',
    parameters_per_correction_arm=27,ridge=.01,official_validation_labels_read=0,new_period_labels_read=0,test_buffer_payloads_read=0,
    note='Future COT from frozen SimVP->R. M is a literature-based encoding, not observed future COT; C_select is reused internal development data, not test.')
out.mkdir();(out/'SCREEN_RESULT.json').write_text(json.dumps(report,indent=2));np.savez_compressed(out/'DATE_ERROR_SUMS.npz',daily=daily,dates=days,arms=arms)
print(json.dumps({k:report[k] for k in ['state','arms','comparisons','decision']},indent=2))
