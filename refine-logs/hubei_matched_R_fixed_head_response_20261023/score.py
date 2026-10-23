"""Saved prediction FP64 response analysis. Offline replacements are not forecasts."""
import csv,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;assert json.loads((P/'replay/COMPLETE.json').read_text())['state']=='CURRENT_FIXED_HEAD_SAME_FOLD_REPLAY_ALL_STATIONS_COMPLETE'
sites=['Niushou','Wangzhai','Mizhuang'];seeds=[42,43,44];arms=['G1','no_future_original','future_original','no_future_joint','forecast_joint','persistence_joint','observed_R_joint_OFFLINE'];records=[];daily={};changes=[];physical={};coverage={}
for si,st in enumerate(sites):
    folder=P/'replay'/st;receipt=json.loads((folder/'COMPLETE.json').read_text());assert receipt['state']=='CURRENT_FIXED_HEAD_SAME_FOLD_COT_RESPONSE_COMPLETE';coverage[st]={k:receipt[k] for k in ['rows','reference_found','joint_available_rows','original_available_rows']}
    with np.load(folder/'physical_COT_sources.npz') as z:q={k:z[k] for k in z.files}
    vm=q['joint_mask'];assert np.array_equal(vm.all(1),q['common_rows'].astype(bool));physical[st]={}
    for name in ['forecast','persistence']:
        e=(q[name]-q['observed_R'])[vm];c=(q[name]-q['observed_R'])[vm[:,4],4]
        physical[st][name]=dict(valid_pixels=len(e),pixel_rmse=float(np.sqrt(np.mean(e*e))),center_n=len(c),center_rmse=float(np.sqrt(np.mean(c*c))))
    prior=None
    for sj,seed in enumerate(seeds):
        with np.load(folder/f'seed{seed}_response.npz') as z:d={k:z[k] for k in z.files}
        assert d['arms'].tolist()==arms and d['prediction'].shape==(7,len(d['truth']))
        keys=list(zip(d['init_time'],d['target_time'],d['lead'],d['source_role']));assert len(keys)==len(set(keys))
        if prior is not None:assert keys==prior[0] and np.array_equal(d['truth'],prior[1]) and np.array_equal(d['joint_available'],prior[2])
        prior=keys,d['truth'],d['joint_available']
        truth=d['truth'];pred=d['prediction'];valid=d['joint_available']>0
        for scope,mask in [('all_rows',np.ones(len(truth),bool)),('joint_available',valid)]:
            for horizon in ['all','1h','2h','3h','4h']:
                hm=mask.copy()
                if horizon!='all':hour=int(horizon[0]);hm&=(d['lead']>(hour-1)*60)&(d['lead']<=hour*60)
                for ai,arm in enumerate(arms):
                    err=pred[ai,hm]-truth[hm];assert len(err)>0
                    records.append(dict(scope=scope,horizon=horizon,station=st,seed=seed,arm=arm,n=len(err),rmse=float(np.sqrt(np.mean(err*err))),mae=float(abs(err).mean()),bias=float(err.mean())))
                    for day in sorted(set(d['date'])):
                        mm=hm&(d['date']==day);e=pred[ai,mm]-truth[mm]
                        daily.setdefault((scope,horizon,day),np.zeros((3,3,7,2),dtype='float64'))[si,sj,ai]=[float(e@e),len(e)]
            for j,k in [(6,4),(5,4),(4,2),(6,0),(4,0),(4,3)]:
                diff=pred[j,mask]-pred[k,mask];e=pred[k,mask]-truth[mask];net=(pred[j,mask]-truth[mask])**2-e*e
                assert float(abs(net-(2*e*diff+diff*diff)).max())<1e-7
                changes.append(dict(scope=scope,station=st,seed=seed,comparison=f'{arms[j]}_minus_{arms[k]}',n=len(e),mean_shift=float(diff.mean()),mean_absolute_shift=float(abs(diff).mean()),max_absolute_shift=float(abs(diff).max()),improved_error_fraction=float((net<0).mean()),delta_MSE=float(net.mean()),interaction=float((2*e*diff).mean()),shift_energy=float((diff*diff).mean())))
summaries={};saves={}
for scope in ['all_rows','joint_available']:
    summaries[scope]={}
    for horizon in ['all','1h','2h','3h','4h']:
        days=sorted({key[2] for key in daily if key[:2]==(scope,horizon)});stats=np.stack([daily[scope,horizon,day] for day in days]);saves[scope+'_'+horizon]=stats
        total=stats.sum(0);points=np.sqrt(total[...,0]/total[...,1]).mean((0,1));w=np.random.default_rng(20261023).multinomial(len(days),np.ones(len(days))/len(days),5000);den=w@stats[...,1].reshape(len(days),-1);assert (den>0).all();boot=np.sqrt((w@stats[...,0].reshape(len(days),-1))/den).reshape(-1,3,3,7).mean((1,2));comparisons={}
        for j,k in [(6,4),(5,4),(4,2),(6,0),(4,0),(4,3),(6,3)]:comparisons[f'{arms[j]}_minus_{arms[k]}']=dict(delta_rmse=float(points[j]-points[k]),ci95=np.quantile(boot[:,j]-boot[:,k],[.025,.975]).tolist())
        summaries[scope][horizon]=dict(arms=dict(zip(arms,points.tolist())),comparisons=comparisons,dates=days,records=int(total[:,0,0,1].sum()))
for name,values in [('metrics',records),('changes',changes)]:
    with (P/(name+'.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
report=dict(state='CURRENT_FIXED_HEAD_MATCHED_R_RESPONSE_SCORED',summary=summaries,coverage=coverage,COT_proxy_errors=physical,claim='Observed future R is an offline out-of-distribution diagnostic, not deployable gain or strict upper bound',new_training_runs=0,official_val_new_period_outerbuffer_test_payloads_read=0)
(P/'RESPONSE_SCORES.json').write_text(json.dumps(report,indent=2));np.savez_compressed(P/'DATE_ERROR_SUMS.npz',**saves,arms=np.array(arms))
print(json.dumps({s:{'points':v['all']['arms'],'comparisons':v['all']['comparisons']} for s,v in summaries.items()},indent=2))
