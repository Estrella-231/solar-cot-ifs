"""Independent input reconstruction witnesses; no model or GHI values."""
import csv,json,hashlib
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;B=P.parent/'hubei_solar_geometry_repair_20261021'
report=json.loads((P/'PACK_COMPLETE.json').read_text());root=Path(report['path']);norm=json.loads((root/'norm.json').read_text());mu=np.array(norm['mean']);sd=np.array(norm['std'])
fields=list(csv.DictReader((root/'fields.csv').open()));x=np.load(root/'x.npy',mmap_mode='r');fit=np.load(root/'fit.npy');sel=np.load(root/'select.npy')
assert not (fit&sel).any() and (fit|sel).all()
assert not ({r['target_time'][:8] for r,b in zip(fields,fit) if b}&{r['target_time'][:8] for r,b in zip(fields,sel) if b})
ids=sorted(set([0,len(fields)-1]+np.random.default_rng(1234).choice(len(fields),128,replace=False).tolist()));cache={};maxdiff=0.
for i in ids:
    r=fields[i];role=r['source_role'];seq=int(r['source_sequence_index']);k=int(r['lead'])//15-1
    if role not in cache:
        folder=B/'S_banks_attempt12'/role/'val'
        with np.load(P/'reference_attempt1'/role/'reference.npz') as z:ref={a:z[a] for a in z.files}
        cache[role]=(np.load(folder/'pred_agri_patch.npy',mmap_mode='r'),np.load(folder/'future_geometry_patch.npy',mmap_mode='r'),np.load(B/'COT_banks_attempt12'/f'{role}.npy',mmap_mode='r'),ref,{str(t):j for j,t in enumerate(ref['bjt'])})
    agri,geo,fc,ref,idx=cache[role];init=datetime.strptime(r['init_time'],'%Y%m%d%H%M');pastids=[idx[(init-timedelta(minutes=m)).strftime('%Y%m%d%H%M')] for m in range(105,-1,-15)]
    hist=ref['cot'][pastids,0];hv=ref['valid'][pastids,0];base=np.asarray(fc[seq,k]).reshape(1,16,16);g=geo[seq,k]
    expected=np.concatenate([agri[seq,k],g,np.log1p(base),np.where(hv,np.log1p(hist),0),hv.astype('float32'),np.full((1,16,16),int(r['lead'])/240,dtype='float32')])
    expected[:25]=(expected[:25]-mu[:,None,None])/sd[:,None,None]
    pv=(g[2:3]>.5)&(g[0:1]>np.cos(np.deg2rad(80)))&np.isfinite(base)&(base>=0)&(base<=100)
    expected[16]=np.where(pv[0],expected[16],0);expected[17:25]=np.where(hv,expected[17:25],0)
    maxdiff=max(maxdiff,float(abs(expected-x[i]).max()))
    for name,value in [('base',base),('target',ref['cot'][idx[r['target_time']]]),('persistence',hist[-1:]),('forecast_mask',pv),('mask',pv&ref['valid'][idx[r['target_time']]]),('score_mask',pv&ref['valid'][idx[r['target_time']]]&hv[-1:])]:assert np.array_equal(np.load(root/(name+'.npy'),mmap_mode='r')[i],value),(i,name)
assert maxdiff<1e-5,maxdiff
receipt=dict(state='PASS_CAUSAL_PACK_RECONSTRUCTION',witnesses=len(ids),max_abs_input_error=maxdiff,fit_fields=int(fit.sum()),select_fields=int(sel.sum()),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),GHI_values_read=0,outer_payloads_read=0)
(P/'PACK_AUDIT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
