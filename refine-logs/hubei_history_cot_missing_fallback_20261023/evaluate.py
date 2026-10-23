import csv,hashlib,importlib.util,json,math
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
from fallback import apply_history_missing_fallback
P=Path(__file__).resolve().parent;E=P.parent/'hubei_history_cot_confirmation_20261022';D=E/'diagnosis_20261023'
spec=importlib.util.spec_from_file_location('geometry',P.parent/'hubei_solar_geometry_repair_20261021/code_pipeline_attempt12/solar_geometry_noaa.py');geo=importlib.util.module_from_spec(spec);spec.loader.exec_module(geo)
centers={'Niushou':(111,124),'Wangzhai':(112,126),'Mizhuang':(110,128)}
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
grid=load(E/'grid_static256.npz');receipt=json.loads((E/'predictions_attempt3/COMPLETE.json').read_text());manifest=[];records=[];sums={}
out=P/'output_attempt1';out.mkdir(exist_ok=False);sites=list(centers);arms=['G0_no_COT','G1_history_COT','history_QA_fallback'];source_freeze_sha=sha(E/'MODEL_FREEZE.json')
for si,st in enumerate(sites):
    ph=load(D/f'{st}_physical.npz');pred=[];reference=None
    for seed in [42,43,44]:
        pair=[]
        for arm in ['G0','G1']:
            file=E/'predictions_attempt3'/f'{st}_seed{seed}_{arm}.npz';d=load(file)
            r=next(x for x in receipt['predictions'] if (x['station'],x['seed'],x['arm'])==(st,seed,arm))
            assert sha(file)==r['sha256']
            if reference is None:reference=d
            for key in ['ghi','clear','init_time_BJT','target_time_BJT','lead_minutes','analysis']:assert np.array_equal(d[key],reference[key])
            pair.append(d['pred_ghi']);manifest.append(dict(station=st,seed=seed,arm=arm,sha256=r['sha256'],checkpoint_sha256=r['checkpoint_sha256']))
        pred.append(pair)
    raw=np.array(pred);r=reference
    for key in ['init_time_BJT','target_time_BJT','lead_minutes']:assert np.array_equal(ph[key],r[key])
    cy,cx=centers[st];lat=grid['lat'][cy-1:cy+2,cx-1:cx+2];lon=grid['lon'][cy-1:cy+2,cx-1:cx+2]
    unique=sorted(set(r['init_time_BJT']));cache={};hist={}
    for stamp in unique:
        t=datetime.fromisoformat(str(stamp));frames=[]
        for off in range(105,-1,-15):
            target=t-timedelta(minutes=off);assert target<=t
            if target not in cache:
                z,_,day=geo.compute_solar_angles(lat,lon,target-timedelta(hours=8))
                cos=np.cos(np.deg2rad(z)).astype('float32');cache[target]=(day>.5)&(cos>np.cos(np.deg2rad(80.)))
            frames.append(cache[target][None])
        hist[stamp]=np.stack(frames)
    valid=np.stack([hist[t] for t in r['init_time_BJT']]);fraction=valid.mean((1,2,3,4))
    assert np.array_equal(fraction,ph['history_valid_fraction'])
    repaired=apply_history_missing_fallback(raw[:,0],raw[:,1],valid)
    missing=~valid.any((1,2,3,4));assert np.array_equal(repaired[:,missing],raw[:,0,missing]) and np.array_equal(repaired[:,~missing],raw[:,1,~missing])
    prediction=np.stack([raw[:,0],raw[:,1],repaired],axis=1)
    dates=np.array([str(t)[:10] for t in r['target_time_BJT']]);n=len(dates)
    np.savez_compressed(out/f'{st}.npz',prediction=prediction,truth=r['ghi'],clear=r['clear'],date=dates,analysis=r['analysis'],
        init_time_BJT=r['init_time_BJT'],target_time_BJT=r['target_time_BJT'],lead_minutes=r['lead_minutes'],history_valid=valid,missing=missing,arms=arms)
    primary=r['analysis']=='primary_shared_dates'
    for group,mask in [('primary',primary),('supplementary',~primary)]:
        if not mask.any():continue
        for sj,seed in enumerate([42,43,44]):
            for ai,arm in enumerate(arms):
                e=prediction[sj,ai]-r['ghi'];v=e[mask]
                records.append(dict(station=st,seed=seed,arm=arm,group=group,n=len(v),missing_rows=int(missing[mask].sum()),rmse=math.sqrt(math.fsum(float(x)**2 for x in v)/len(v)),mae=float(abs(v).mean()),bias=float(v.mean())))
                for day in sorted(set(dates[mask])):
                    choose=mask&(dates==day);sums.setdefault((group,day),np.zeros((3,3,3,2)))[si,sj,ai]=[math.fsum(float(x)**2 for x in e[choose]),int(choose.sum())]
    print(st,'QA_replay_pass','primary_missing',int(missing[primary].sum()),flush=True)
summary={}
for group in ['primary','supplementary']:
    days=sorted(day for g,day in sums if g==group)
    if not days:continue
    stat=np.stack([sums[group,day] for day in days]);total=stat.sum(0)
    active=np.where(total[:,0,0,1]>0)[0];stat=stat[:,active];total=total[active]
    point=np.sqrt(total[...,0]/total[...,1]);mean=point.mean((0,1));w=np.random.default_rng(20261022).multinomial(len(days),np.ones(len(days))/len(days),5000)
    flat=stat.reshape(len(days),-1,2);den=w@flat[:,:,1];assert (den>0).all()
    boot=np.sqrt((w@flat[:,:,0])/den).reshape(5000,len(active),3,3).mean((1,2));comp={}
    for j,k in [(2,0),(2,1),(1,0)]:comp[f'{arms[j]}_minus_{arms[k]}']=dict(delta_rmse=float(mean[j]-mean[k]),relative_change_percent=float((mean[j]/mean[k]-1)*100),ci95=np.quantile(boot[:,j]-boot[:,k],[.025,.975]).tolist())
    summary[group]=dict(rmse=dict(zip(arms,mean.tolist())),comparisons=comp,by_station={sites[si]:dict(zip(arms,point[i].mean(0).tolist())) for i,si in enumerate(active)},dates=days)
assert sha(E/'MODEL_FREEZE.json')==source_freeze_sha
with (P/'metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
(P/'RESULTS.json').write_text(json.dumps(dict(state='POSTHOC_MISSING_INPUT_POLICY_SCORED',summary=summary,prediction_sources=manifest,
    source_model_freeze_sha256=source_freeze_sha,grid_sha256=sha(E/'grid_static256.npz'),new_model_fits=0,independent_confirmation=False,test_payloads_read=0),indent=2))
print(json.dumps(summary,indent=2))
