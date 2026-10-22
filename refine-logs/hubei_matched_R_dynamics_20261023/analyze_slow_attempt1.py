"""Same-fold observed-R reference; causal forecast/persistence, no fitting."""
import csv,json,math
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;B=P.parent/'hubei_solar_geometry_repair_20261021';H=P.parent/'hubei_forecast_cot_increment_20261022/prepared_attempt1'
centers={'Niushou':(8,6),'Wangzhai':(9,8),'Mizhuang':(7,10)};stations=list(centers)
assert json.loads((P/'reference_attempt1/COMPLETE.json').read_text())['state']=='MATCHED_FIVE_FOLD_ACTUAL_TRAIN_COT_REFERENCE_COMPLETE'
source=list(csv.DictReader((P/'raw_attempt1/pairs.csv').open()));frames=list(csv.DictReader((P/'raw_attempt1/frames.csv').open()));frameidx={r['bjt']:int(r['frame_index']) for r in frames}
raw=np.load(P/'raw_attempt1/raw16.npy',mmap_mode='r');packrows=list(csv.DictReader((B/'GHI_pack_attempt12/rows.csv').open()));fc=np.load(B/'GHI_pack_attempt12/cot.npy',mmap_mode='r');geo=np.load(B/'GHI_pack_attempt12/geometry.npy',mmap_mode='r')
refs={};rowmeta={};labels={}
for role in [f'fold_{i}' for i in range(5)]:
    z=np.load(P/'reference_attempt1'/role/'reference.npz');refs[role]=(z,{t:i for i,t in enumerate(z['bjt'])})
for st in stations:
    rowmeta[st]=list(csv.DictReader((H/st/'rows.csv').open()));labels[st]=(np.load(H/st/'ghi.npy',mmap_mode='r'),np.load(H/st/'clear.npy',mmap_mode='r'))
kept=[];pred=[];init=[];target=[];mask=[];truth=[];clear=[];geodiff=[]
for r in source:
    if r['all_frame_metadata_available']!='True':continue
    st=r['station'];cy,cx=centers[st];sl=(slice(cy-1,cy+2),slice(cx-1,cx+2));ii=int(r['prepared_index']);hr=rowmeta[st][ii];gidx=int(hr['index']);g=packrows[gidx]
    assert (g['station'],g['init_time_bjt'],g['target_time_bjt'],g['lead_minutes'],g['source_role'])==(st,r['init_time'],r['target_time'],r['lead'],r['source_role'])
    z,lookup=refs[r['source_role']];a=lookup[r['init_time']];b=lookup[r['target_time']]
    x=np.asarray(fc[gidx,0],np.float64).reshape(9);pi=z['cot'][a,0][sl].astype('float64').reshape(9);y=z['cot'][b,0][sl].astype('float64').reshape(9)
    solar=geo[gidx,0].reshape(9)>np.cos(np.deg2rad(80.));valid=z['valid'][a,0][sl].reshape(9)&z['valid'][b,0][sl].reshape(9)&solar&np.isfinite(x)&(x>=0)&(x<=100)
    gd=float(abs(np.asarray(geo[gidx],np.float64)-raw[frameidx[r['target_time']],13:15,sl[0],sl[1]]).max());assert gd<1e-6,gd
    kept.append(r);pred.append(x);init.append(pi);target.append(y);mask.append(valid);truth.append(float(labels[st][0][ii]));clear.append(float(labels[st][1][ii]));geodiff.append(gd)
pred,init,target,mask=map(np.array,[pred,init,target,mask]);truth=np.array(truth);clear=np.array(clear);st=np.array([r['station'] for r in kept]);scope=np.array([r['scope'] for r in kept]);lead=np.array([int(r['lead']) for r in kept]);dates=np.array([r['target_time'][:8] for r in kept]);rows=[];summary={};daily=[]
def corr(x,y):return float(np.corrcoef(x,y)[0,1]) if len(x)>2 and np.std(x)>0 and np.std(y)>0 else None
def metric(x,p,y):
    e=x-y;d=x-p;dy=y-p;growth=dy>3;decay=dy< -3;pg=d>3;pd=d< -3;thick=y>=30;pt=x>=30
    rate=lambda a,b:float(a.sum()/b.sum()) if b.sum() else None
    return dict(n=len(y),rmse=float(np.sqrt(np.mean(e*e))),bias=float(e.mean()),mae=float(abs(e).mean()),increment_correlation=corr(d,dy),growth_n=int(growth.sum()),growth_POD=rate(pg&growth,growth),growth_FAR=rate(pg&~growth,pg),decay_n=int(decay.sum()),decay_POD=rate(pd&decay,decay),decay_FAR=rate(pd&~decay,pd),three_class_accuracy=float(np.mean(np.where(d>3,1,np.where(d< -3,-1,0))==np.where(dy>3,1,np.where(dy< -3,-1,0)))),thick_POD=rate(pt&thick,thick),thick_FAR=rate(pt&~thick,pt))
for sc in ['C_fit','C_select']:
    summary[sc]={}
    for label,lo,hi in [('all',15,240),('1h',15,60),('2h',75,120),('3h',135,180),('4h',195,240)]:
        sse=np.zeros((3,2,len(set(dates[scope==sc]))));den=np.zeros((3,sse.shape[-1]));days=sorted(set(dates[scope==sc]));daysix={d:i for i,d in enumerate(days)}
        for si,site in enumerate(stations):
            m=(scope==sc)&(st==site)&(lead>=lo)&(lead<=hi)&mask[:,4]
            assert m.any()
            for name,x in [('persistence',init[m,4]),('forecast',pred[m,4])]:rows.append(dict(scope=sc,station=site,horizon=label,source=name,**metric(x,init[m,4],target[m,4])))
            for day in days:
                mm=m&(dates==day);di=daysix[day];den[si,di]=mm.sum()
                for ji,x in enumerate([init,pred]):sse[si,ji,di]=float(np.sum((x[mm,4]-target[mm,4])**2))
                daily.append(dict(scope=sc,station=site,horizon=label,date=day,n=int(mm.sum()),persistence_sse=sse[si,0,di],forecast_sse=sse[si,1,di]))
        points=np.sqrt(sse.sum(-1)/den.sum(-1)[:,None]).mean(0)
        w=np.random.default_rng(20261023).multinomial(len(days),np.ones(len(days))/len(days),5000)
        boot=np.sqrt(np.einsum('sd,td->ts',den,w)[:, :,None]**-1*np.einsum('sjd,td->tsj',sse,w)).mean(1)
        ci=np.quantile(boot[:,1]-boot[:,0],[.025,.975])
        summary[sc][label]=dict(persistence_rmse=float(points[0]),forecast_rmse=float(points[1]),forecast_minus_persistence=float(points[1]-points[0]),ci95=ci.tolist(),dates=len(days),common_center_records=int(den.sum()))
        if label=='all':np.savez_compressed(P/f'{sc}_DATE_SUMS.npz',sse=sse,den=den,dates=days)
out=P/'PAIRED_TRAIN_REFERENCES.npz';np.savez_compressed(out,forecast=pred,persistence=init,observed_R=target,valid=mask,station=st,scope=scope,lead=lead,date=dates,init_time=np.array([r['init_time'] for r in kept]),target_time=np.array([r['target_time'] for r in kept]),source_role=np.array([r['source_role'] for r in kept]),ghi=truth,clear=clear)
for name,data in [('metrics',rows),('daily',daily)]:
    with (P/(name+'.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
report=dict(state='MATCHED_R_TRAIN_DYNAMICS_ANALYSIS_COMPLETE',summary=summary,candidate_pairs=len(source),paired_records=len(kept),common_center_records=int(mask[:,4].sum()),common_3x3_pixels=int(mask.sum()),target_geometry_max_difference=max(geodiff),future_observed_R_usage='label/offline reference only, not independent truth or deployable input',new_fits=0,official_val_new_period_outer_test_buffer_payloads_read=0)
(P/'ANALYSIS.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
