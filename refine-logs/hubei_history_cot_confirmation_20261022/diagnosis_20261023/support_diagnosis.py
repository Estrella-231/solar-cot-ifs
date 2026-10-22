import json,csv,numpy as np
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent;out={};records=[]
groups=list(csv.DictReader((P/'groups.csv').open()));decomp=json.loads((P/'DECOMPOSITION.json').read_text())
for st in ['Niushou','Wangzhai','Mizhuang']:
    allrow=decomp[st];sub=[r for r in groups if r['station']==st];outside=next(r for r in sub if r['group']=='outside_old_init_hours');invalid=next(r for r in sub if r['group']=='history_all_invalid')
    total=allrow['delta_mse']*allrow['n'];out[st]=dict(outside_old_hours_fraction=float(outside['n'])/allrow['n'],outside_old_hours_net_delta_sse_fraction=float(outside['n'])*float(outside['delta_mse'])/total,all_invalid_fraction=float(invalid['n'])/allrow['n'],all_invalid_net_delta_sse_fraction=float(invalid['n'])*float(invalid['delta_mse'])/total)
    a=[];b=[]
    for seed in [42,43,44]:
        with np.load(B/'predictions_attempt3'/f'{st}_seed{seed}_G0.npz') as z:r={k:z[k] for k in z.files}
        a.append(r['pred_ghi']);b.append(np.load(B/'predictions_attempt3'/f'{st}_seed{seed}_G1.npz')['pred_ghi'])
    hours=np.array([float(x[11:13])+float(x[14:16])/60 for x in r['init_time_BJT']]);m=(r['analysis']=='primary_shared_dates')&(hours>=9.75)&(hours<=15.5);dates=np.array([x[:10] for x in r['target_time_BJT'][m]]);days=sorted(set(dates));idx=np.searchsorted(days,dates);den=np.bincount(idx,minlength=len(days));sse=np.array([[np.bincount(idx,weights=(x[m]-r['ghi'][m])**2,minlength=len(days)) for x in v] for v in [a,b]])
    records.append((sse,den,days))
days=sorted(set(d for _,_,dd in records for d in dd));sse=[];den=[]
for ss,nn,dd in records:
    x=np.zeros((2,3,len(days)));n=np.zeros(len(days));ix=np.searchsorted(days,dd);x[:,:,ix]=ss;n[ix]=nn;sse.append(x);den.append(n)
sse=np.array(sse);den=np.array(den);w=np.random.default_rng(20261022).multinomial(len(days),np.ones(len(days))/len(days),5000)
point=np.sqrt(sse.sum(-1)/den.sum(-1)[:,None,None]).mean(axis=(0,2));boot=np.sqrt(np.einsum('sacd,bd->sacb',sse,w)/np.einsum('sd,bd->sb',den,w)[:,None,None,:]).mean(axis=(0,2));out['old_hours_diagnostic']=dict(rmse=point.tolist(),delta=float(point[1]-point[0]),delta_ci95=np.quantile(boot[1]-boot[0],[.025,.975]).tolist(),rows=int(den.sum()),days=len(days),note='posthoc support diagnostic from actual old training init range; does not replace frozen full new-period result')
(P/'SUPPORT_DIAGNOSTIC.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
