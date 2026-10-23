"""Saved F_select predictions only; fixed date-cluster gates, no GHI claims."""
import csv,hashlib,json,math
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;train=P/'training_attempt1'
assert json.loads((train/'COMPLETE.json').read_text())['state']=='THREE_SEED_COT_DOMAIN_PILOT_COMPLETE'
manifest=json.loads((P/'PACK_COMPLETE.json').read_text());root=Path(manifest['path'])
fields=list(csv.DictReader((root/'fields.csv').open()));members=list(csv.DictReader((root/'members.csv').open()));selected=np.flatnonzero(np.load(root/'select.npy'));idx={int(v):i for i,v in enumerate(selected)}
arrays={name:np.array(np.load(root/(name+'.npy'),mmap_mode='r')[selected]) for name in ['base','target','persistence','score_mask']}
preds=[];seeds=[]
for seed in [42,43,44]:
    done=json.loads((train/f'seed{seed}/COMPLETE.json').read_text());meta=json.loads((train/f'seed{seed}/run_metadata.json').read_text())
    assert done['updates']==meta['updates']==600
    fn=train/f'seed{seed}/best_select_predictions.npz';assert hashlib.sha256(fn.read_bytes()).hexdigest()==done['prediction_sha256']
    with np.load(fn) as z:
        assert np.array_equal(z['field_index'],selected);preds.append(z['prediction'].astype('float64'))
    seeds.append(done)
methods=['raw_forecast','persistence','seed42','seed43','seed44'];values=[arrays['base'].astype('float64'),arrays['persistence'].astype('float64')]+preds
y=arrays['target'].astype('float64');valid=arrays['score_mask'];centers={'Niushou':(8,6),'Wangzhai':(9,8),'Mizhuang':(7,10)}
dates=sorted({fields[i]['target_time'][:8] for i in selected});di={d:i for i,d in enumerate(dates)}
den=np.zeros((48,len(dates)));sse=np.zeros((5,48,len(dates)));rows=[];sample=[];guard={};point_independent=[]
for si,(st,(cy,cx)) in enumerate(centers.items()):
    ids=np.array([idx[int(r['field_index'])] for r in members if r['station']==st and int(r['field_index']) in idx],dtype='int64');ids=ids[valid[ids,0,cy,cx]]
    leads=np.array([int(fields[selected[i]]['lead']) for i in ids]);days=np.array([di[fields[selected[i]]['target_time'][:8]] for i in ids]);truth=y[ids,0,cy,cx];persistence=values[1][ids,0,cy,cx]
    guard[st]={}
    for mi,(name,pr) in enumerate(zip(methods,values)):
        pv=pr[ids,0,cy,cx];e=pv-truth;growth=truth-persistence>3;pg=pv-persistence>3;thick=truth>=30;pt=pv>=30
        rate=lambda num,de:float(np.sum(num)/np.sum(de)) if np.sum(de) else None
        guard[st][name]=dict(n=len(ids),rmse=math.sqrt(math.fsum(float(v)**2 for v in e)/len(e)),bias=float(e.mean()),growth_POD=rate(pg&growth,growth),growth_n=int(growth.sum()),thick_POD=rate(pt&thick,thick),thick_FAR=rate(pt&~thick,pt),thick_n=int(thick.sum()))
        for li,lead in enumerate(range(15,241,15)):
            mm=leads==lead;group=si*16+li;assert mm.any()
            sse[mi,group]=np.bincount(days[mm],weights=e[mm]**2,minlength=len(dates));den[group]=np.bincount(days[mm],minlength=len(dates))
            rmse=math.sqrt(math.fsum(float(v)**2 for v in e[mm])/int(mm.sum()));rows.append(dict(station=st,lead=lead,method=name,n=int(mm.sum()),rmse=rmse,bias=float(e[mm].mean())))
            if mi==0:point_independent.append(rmse)
        for j,i in enumerate(ids):sample.append(dict(station=st,field_index=int(selected[i]),date=dates[days[j]],lead=int(leads[j]),method=name,target=float(truth[j]),prediction=float(pv[j])))
point=np.sqrt(sse.sum(-1)/den.sum(-1)[None]).mean(1)
assert abs(point[0]-math.fsum(point_independent)/48)<1e-10
w=np.random.default_rng(20261023).multinomial(len(dates),np.ones(len(dates))/len(dates),5000);bd=np.einsum('gd,bd->bg',den,w);good=(bd>0).all(1);w=w[good];bd=bd[good]
boot=np.sqrt(np.einsum('mgd,bd->bmg',sse,w)/bd[:,None]).mean(2);candidate=point[2:].mean();boot_candidate=boot[:,2:].mean(1)
comparisons={}
for mi in [0,1]:comparisons[methods[mi]]=dict(delta=float(candidate-point[mi]),ci95=np.quantile(boot_candidate-boot[:,mi],[.025,.975]).tolist())
site_nonregression=all(np.mean([guard[s][m]['rmse'] for m in methods[2:]])<=min(guard[s]['raw_forecast']['rmse'],guard[s]['persistence']['rmse']) for s in centers)
seed_improvements=int(sum(p<min(point[:2]) for p in point[2:]))
guard_ok=True;guard_failures=[]
for st in centers:
    ref=guard[st]['raw_forecast']
    for name in methods[2:]:
        row=guard[st][name]
        for key,tol,direction in [('thick_POD',.01,1),('thick_FAR',.02,-1),('growth_POD',.01,1)]:
            if ref[key] is None or row[key] is None:ok=False
            else:ok=(row[key]-ref[key])*direction>=-tol
            if not ok:guard_ok=False;guard_failures.append(dict(station=st,seed=name,metric=key,raw=ref[key],corrected=row[key]))
gate=all(r['ci95'][1]<0 for r in comparisons.values()) and site_nonregression and seed_improvements>=2 and guard_ok
full={};neighborhood={};date_rows=[]
for name,pred in zip(methods,values):
    e=(pred-y)[valid];full[name]=dict(full_patch_common_valid_pixels=len(e),rmse=float(np.sqrt(np.mean(e*e))),bias=float(e.mean()))
for st,(cy,cx) in centers.items():
    ids=np.array([idx[int(r['field_index'])] for r in members if r['station']==st and int(r['field_index']) in idx],dtype='int64');sl=(slice(cy-1,cy+2),slice(cx-1,cx+2));vm=valid[ids,0,sl[0],sl[1]];yy=y[ids,0,sl[0],sl[1]];has=vm.reshape(len(ids),-1).any(1);neighborhood[st]={}
    for name,pred in zip(methods,values):
        pp=pred[ids,0,sl[0],sl[1]];e=(pp-yy)[vm];pred_peak=np.where(vm,pp,-np.inf).reshape(len(ids),-1).max(1)[has];true_peak=np.where(vm,yy,-np.inf).reshape(len(ids),-1).max(1)[has]
        neighborhood[st][name]=dict(valid_pixels=len(e),rmse=float(np.sqrt(np.mean(e*e))),bias=float(e.mean()),local_peak_rmse=float(np.sqrt(np.mean((pred_peak-true_peak)**2))),local_peak_bias=float(np.mean(pred_peak-true_peak)))
for d,day in enumerate(dates):
    vm=den[:,d]>0
    for mi,name in enumerate(methods):date_rows.append(dict(date=day,method=name,station_lead_groups=int(vm.sum()),n=int(den[:,d].sum()),equal_available_group_rmse=float(np.sqrt(sse[mi,vm,d]/den[vm,d]).mean())))
report=dict(state='F_SELECT_CORRECTOR_GATE_PASS' if gate else 'F_SELECT_CORRECTOR_GATE_FAIL_SEAL_CANDIDATE',dates=dates,equal_station_lead_rmse=dict(zip(methods,point.tolist())),corrected_three_seed_mean=float(candidate),comparisons=comparisons,station_metrics=guard,full_patch=full,station_3x3=neighborhood,site_nonregression=site_nonregression,improving_seeds=seed_improvements,physical_guard_pass=guard_ok,guard_failures=guard_failures,bootstrap_draws=len(w),degenerate_draws=int((~good).sum()),seeds=seeds,claim='Same-fold observed-R proxy on reused development train dates, not CPP truth or GHI improvement',C_fit_C_select_corrected_predictions_read=0,outer_payloads_read=0)
(P/'CORRECTION_SCORES.json').write_text(json.dumps(report,indent=2));np.savez_compressed(P/'F_SELECT_DATE_SUMS.npz',sse=sse,den=den,dates=np.array(dates),methods=np.array(methods))
for name,data in [('correction_metrics',rows),('correction_samples',sample),('correction_dates',date_rows)]:
    with (P/(name+'.csv')).open('w',newline='') as f:ww=csv.DictWriter(f,fieldnames=list(data[0]));ww.writeheader();ww.writerows(data)
print(json.dumps({k:v for k,v in report.items() if k not in ['station_metrics','seeds','guard_failures','full_patch']},indent=2))
