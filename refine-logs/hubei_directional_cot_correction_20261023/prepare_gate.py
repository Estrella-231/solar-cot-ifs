"""F_fit OOF / F_select full-fit direction maps; only causal features at inference."""
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent;PRE=P.parent/'hubei_forecast_domain_cot_20261023';PROBE=P.parent/'hubei_cot_direction_identifiability_20261023'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert json.loads((PROBE/'INDEPENDENT_AUDIT.json').read_text())['state'].startswith('PASS_INDEPENDENT_OOF')
manifest=json.loads((PRE/'PACK_COMPLETE.json').read_text());root=Path(manifest['path']);fit=np.load(root/'fit.npy');fields=list(csv.DictReader((root/'fields.csv').open()));norm=json.loads((root/'norm.json').read_text());mu=np.array(norm['mean']);sd=np.array(norm['std']);x=np.load(root/'x.npy',mmap_mode='r');available=np.load(root/'forecast_mask.npy',mmap_mode='r');rows=list(csv.DictReader((PROBE/'OOF_predictions.csv').open()));centers={'Niushou':(8,6),'Wangzhai':(9,8),'Mizhuang':(7,10)}
out=Path('/dev/shm/hubei_directional_cot_20261023_gate_attempt1');out.mkdir(exist_ok=False);torch.set_num_threads(2);start=time.perf_counter()
features=[];labels=[]
for station,(cy,cx) in centers.items():
    rr=[r for r in rows if r['station']==station];ids=np.array([int(r['field_index']) for r in rr]);assert fit[ids].all()
    xx=np.array(x[ids,:,cy-1:cy+2,cx-1:cx+2],dtype='float64');xx[:,:25]=xx[:,:25]*sd[None,:,None,None]+mu[None,:,None,None];xx[:,17:25]=np.where(xx[:,25:33]>.5,xx[:,17:25],0);xx[:,16]=np.where(available[ids,0,cy-1:cy+2,cx-1:cx+2],xx[:,16],0)
    features.append(xx.reshape(len(ids),306));labels.append(np.array([int(r['label']) for r in rr]))
X=np.concatenate(features);y=np.concatenate(labels);mean=X.mean(0);std=np.maximum(X.std(0),1e-4);tx=np.column_stack([np.ones(len(y)),(X-mean)/std]);counts=np.bincount(y,minlength=3);w=len(y)/(3*counts[y]);penalty=np.eye(307)*.01;penalty[0,0]=0;coef=np.linalg.solve(tx.T@(tx*w[:,None])/len(y)+penalty,tx.T@(np.eye(3)[y]*w[:,None])/len(y))
np.savez_compressed(P/'full_F_fit_direction_model.npz',mean=mean,std=std,coef=coef,class_counts=counts,train_dates=np.array(sorted({r['date'] for r in rows})))
del X,tx,features
models={'full':dict(mean=mean,std=std,coef=coef)};assignment={}
for fold in range(3):
    with np.load(PROBE/'models'/f'fold_{fold}.npz') as z:
        models[str(fold)]={k:z[k] for k in ['mean','std','coef']}
        for d in z['held_out_dates']:assignment[str(d)]=str(fold)
gate=np.lib.format.open_memmap(out/'direction.npy',mode='w+',dtype='int8',shape=(len(fields),1,16,16));gate[:]=0;fallback=[];counts_by_scope={}
for key,m in models.items():
    ids=np.array([i for i,r in enumerate(fields) if (not fit[i] and key=='full') or (fit[i] and assignment.get(r['target_time'][:8])==key)],dtype='int64')
    if not len(ids):continue
    weight=torch.from_numpy((m['coef'][1:]/m['std'][:,None]).T.reshape(3,34,3,3).copy());bias=torch.from_numpy((m['coef'][0]-(m['mean']/m['std'])@m['coef'][1:]).copy())
    for j in range(0,len(ids),128):
        ii=ids[j:j+128];xx=np.array(x[ii],dtype='float64');xx[:,:25]=xx[:,:25]*sd[None,:,None,None]+mu[None,:,None,None];xx[:,17:25]=np.where(xx[:,25:33]>.5,xx[:,17:25],0);xx[:,16]=np.where(available[ii,0],xx[:,16],0)
        with torch.inference_mode():scores=F.conv2d(torch.from_numpy(xx),weight,bias).numpy()
        directions=np.argmax(scores,axis=1).astype('int8')-1;gg=np.zeros((len(ii),1,16,16),dtype='int8');gg[:,0,1:-1,1:-1]=directions;gg=np.where(available[ii],gg,0).astype('int8');gate[ii]=gg
    print('MAP',key,len(ids),time.perf_counter()-start,flush=True)
for i,r in enumerate(fields):
    if fit[i] and r['target_time'][:8] not in assignment:fallback.append(i)
gate.flush();mismatches=0
for station,(cy,cx) in centers.items():
    rr=[r for r in rows if r['station']==station];ids=np.array([int(r['field_index']) for r in rr]);expected=np.array([int(r['prediction'])-1 for r in rr]);mismatches+=int(np.sum(gate[ids,0,cy,cx]!=expected))
assert mismatches==0,mismatches
for scope,mm in [('F_fit',fit),('F_select',~fit)]:
    count=np.zeros(3,dtype='int64')
    for j in range(0,len(fields),256):count+=np.bincount((np.asarray(gate[j:j+256])[mm[j:j+256]]+1).ravel(),minlength=3)
    counts_by_scope[scope]=count.tolist()
assert (gate[:,:,0,:]==0).all() and (gate[:,:,-1,:]==0).all() and (gate[:,:,:,0]==0).all() and (gate[:,:,:,-1]==0).all()
report=dict(state='PASS_CAUSAL_OOF_DIRECTION_MAPS_AND_ALL_STATION_REPLAYS',path=str(out/'direction.npy'),sha256=sha(out/'direction.npy'),shape=list(gate.shape),dtype='int8',OOF_station_records_replayed=len(rows),OOF_label_mismatches=mismatches,fit_fields_without_OOF_date=fallback,full_F_fit_classifier_sha256=sha(P/'full_F_fit_direction_model.npz'),source_sha256=sha(__file__),pack_manifest_sha256=sha(PRE/'PACK_COMPLETE.json'),direction_counts_down_stable_up=counts_by_scope,seconds=time.perf_counter()-start,F_select_target_values_read=0,GHI_values_read=0,outer_payloads_read=0)
(P/'GATE_COMPLETE.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
