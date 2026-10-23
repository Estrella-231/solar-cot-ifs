"""Date-separated F_fit-only linear probe of COT correction direction."""
import csv,hashlib,json,time
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;PRE=P.parent/'hubei_forecast_domain_cot_20261023'
receipt=P/'COMPLETE.json';assert not receipt.exists()
manifest=json.loads((PRE/'PACK_COMPLETE.json').read_text());root=Path(manifest['path']);fit=np.load(root/'fit.npy');fields=list(csv.DictReader((root/'fields.csv').open()));members=list(csv.DictReader((root/'members.csv').open()));norm=json.loads((root/'norm.json').read_text());mu=np.array(norm['mean']);sd=np.array(norm['std'])
arrays={n:np.load(root/(n+'.npy'),mmap_mode='r') for n in ['x','base','target','score_mask','forecast_mask']}
centers={'Niushou':(8,6),'Wangzhai':(9,8),'Mizhuang':(7,10)};meta=[];features=[];truth=[]
for st,(cy,cx) in centers.items():
    rows=[r for r in members if r['station']==st and fit[int(r['field_index'])] and arrays['score_mask'][int(r['field_index']),0,cy,cx]]
    ids=np.array([int(r['field_index']) for r in rows],dtype='int64');sl=(slice(cy-1,cy+2),slice(cx-1,cx+2))
    xx=np.asarray(arrays['x'][ids,:,sl[0],sl[1]],dtype='float64');xx[:,:25]=xx[:,:25]*sd[None,:,None,None]+mu[None,:,None,None]
    xx[:,17:25]=np.where(xx[:,25:33]>.5,xx[:,17:25],0);xx[:,16]=np.where(arrays['forecast_mask'][ids,0,sl[0],sl[1]],xx[:,16],0)
    features.append(xx.reshape(len(ids),306));delta=arrays['target'][ids,0,cy,cx].astype('float64')-arrays['base'][ids,0,cy,cx].astype('float64');truth.append(np.where(delta>3,2,np.where(delta< -3,0,1)))
    for row,i,d in zip(rows,ids,delta):
        f=fields[i];assert f['scope']=='F_fit';meta.append(dict(station=st,field_index=int(i),prepared_index=row['prepared_index'],init_time=f['init_time'],target_time=f['target_time'],lead=int(f['lead']),date=f['target_time'][:8],source_role=f['source_role'],reference_minus_forecast=float(d)))
X=np.concatenate(features);y=np.concatenate(truth);assert X.shape==(len(meta),306) and np.isfinite(X).all()
dates=np.array([r['date'] for r in meta]);unique=sorted(set(dates));assignment={}
for month in sorted({d[:6] for d in unique}):
    ds=np.array([d for d in unique if d.startswith(month)])
    for k,block in enumerate(np.array_split(ds,3)):
        for d in block:assignment[str(d)]=k
score=np.full((len(y),3),np.nan);folds=[];start=time.perf_counter()
out=P/'models';out.mkdir(exist_ok=False)
for fold in range(3):
    vdates={d for d in unique if assignment[d]==fold};excluded=set()
    for d in vdates:
        dt=datetime.strptime(d,'%Y%m%d');excluded.update((dt+timedelta(days=o)).strftime('%Y%m%d') for o in [-1,0,1])
    vm=np.isin(dates,list(vdates));tm=~np.isin(dates,list(excluded));assert vm.any() and tm.any() and not (vm&tm).any()
    mean=X[tm].mean(0);std=np.maximum(X[tm].std(0),1e-4);tx=np.column_stack([np.ones(tm.sum()),(X[tm]-mean)/std]);ty=np.eye(3)[y[tm]]
    counts=np.bincount(y[tm],minlength=3);assert (counts>0).all();weights=tm.sum()/(3*counts[y[tm]])
    gram=tx.T@(tx*weights[:,None])/tm.sum();rhs=tx.T@(ty*weights[:,None])/tm.sum();penalty=np.eye(307)*.01;penalty[0,0]=0;coef=np.linalg.solve(gram+penalty,rhs)
    score[vm]=np.column_stack([np.ones(vm.sum()),(X[vm]-mean)/std])@coef
    np.savez_compressed(out/f'fold_{fold}.npz',mean=mean,std=std,coef=coef,train_dates=np.array(sorted(set(dates[tm]))),held_out_dates=np.array(sorted(vdates)),excluded_dates=np.array(sorted(excluded)))
    folds.append(dict(fold=fold,train_n=int(tm.sum()),held_out_n=int(vm.sum()),train_dates=sorted(set(dates[tm])),held_out_dates=sorted(vdates),excluded_dates=sorted(excluded),train_class_counts=counts.tolist(),model_sha256=hashlib.sha256((out/f'fold_{fold}.npz').read_bytes()).hexdigest()))
    print('FOLD',fold,tm.sum(),vm.sum(),time.perf_counter()-start,flush=True)
assert np.isfinite(score).all();pred=score.argmax(1)
for i,r in enumerate(meta):r.update(fold=assignment[r['date']],label=int(y[i]),prediction=int(pred[i]),score_down=float(score[i,0]),score_stable=float(score[i,1]),score_up=float(score[i,2]))
with (P/'OOF_predictions.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(meta[0]));w.writeheader();w.writerows(meta)
np.savez_compressed(P/'OOF_scores.npz',truth=y,prediction=pred,scores=score,date=dates,station=np.array([r['station'] for r in meta]),fold=np.array([assignment[d] for d in dates]))
result=dict(state='F_FIT_DATE_OOF_COT_DIRECTION_PROBE_COMPLETE',rows=len(y),dates=len(unique),features=306,folds=folds,seconds=time.perf_counter()-start,pack_manifest_sha256=hashlib.sha256((PRE/'PACK_COMPLETE.json').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),F_select_C_select_outer_payloads_read=0,GHI_values_read=0,GPU_used=False,classes=['need_down','stable','need_up'],threshold_COT=3,probe='Class-balanced ridge least squares, alpha0.01 average loss; argmax, no probability calibration or tuning')
receipt.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='folds'}),flush=True)
