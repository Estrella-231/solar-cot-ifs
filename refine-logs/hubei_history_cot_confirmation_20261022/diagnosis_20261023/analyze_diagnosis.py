import json,csv
import numpy as np
from pathlib import Path
P=Path(__file__).resolve().parent; BASE=P.parent
out={}; groups=[]; daily=[]
def score(x,y): return float(np.sqrt(np.mean((x-y)**2)))
for st in ['Niushou','Wangzhai','Mizhuang']:
    aa=[];bb=[];zz=[]
    for seed in [42,43,44]:
        with np.load(BASE/'predictions_attempt3'/f'{st}_seed{seed}_G0.npz') as z:r={k:z[k] for k in z.files}
        aa.append(r['pred_ghi']);bb.append(np.load(BASE/'predictions_attempt3'/f'{st}_seed{seed}_G1.npz')['pred_ghi']);zz.append(np.load(P/f'{st}_seed{seed}_counterfactual.npz')['g1_zero_cot'])
    a,b,z=map(np.array,[aa,bb,zz]);g=r['ghi'];m=r['analysis']=='primary_shared_dates'; dates=np.array([x[:10] for x in r['target_time_BJT']]); hours=np.array([int(x[11:13])+int(x[14:16])/60 for x in r['init_time_BJT']]);ph=np.load(P/f'{st}_physical.npz');hv=ph['history_valid_fraction']; validtarget=np.isfinite(ph['target_cot']);kt=g/r['clear']
    def met(mask):
        d=b[:,mask]-a[:,mask];e=a[:,mask]-g[mask];bb=b[:,mask]-g[mask];direct=b[:,mask]-z[:,mask]
        return dict(n=int(mask.sum()),rmse_A=float(np.sqrt((e*e).mean(1)).mean()),rmse_AC=float(np.sqrt((bb*bb).mean(1)).mean()),rmse_fixed_AC_zeroCOT=float(np.sqrt(((z[:,mask]-g[mask])**2).mean(1)).mean()),bias_A=float(e.mean()),bias_AC=float(bb.mean()),mean_adjustment=float(d.mean()),direct_COT_shift=float(direct.mean()),mean_abs_direct_COT_shift=float(abs(direct).mean()),improve_fraction=float((bb*bb<e*e).mean()),delta_mse=float((bb*bb-e*e).mean()),interaction=float((2*e*d).mean()),shift_energy=float((d*d).mean()))
    out[st]=met(m);out[st]['by_seed']=[dict(seed=s,rmse_A=score(a[i,m],g[m]),rmse_AC=score(b[i,m],g[m]),rmse_AC_zeroCOT=score(z[i,m],g[m])) for i,s in enumerate([42,43,44])]
    bins=[('old_init_hours_0945to1530',(hours>=9.75)&(hours<=15.5)),('outside_old_init_hours',(hours<9.75)|(hours>15.5)),('init_before08',hours<8),('init_08to12',(hours>=8)&(hours<12)),('init_after12',hours>=12),('history_all_invalid',hv==0),('history_partial',(hv>0)&(hv<1)),('history_all_valid',hv==1),('GHI_lt200',g<200),('GHI_200to500',(g>=200)&(g<500)),('GHI_500to800',(g>=500)&(g<800)),('GHI_ge800',g>=800),('kt_lt0.3',kt<.3),('kt_0.3to0.7',(kt>=.3)&(kt<.7)),('kt_ge0.7',kt>=.7)]
    bins += [(f'lead_{h}h',(r['lead_minutes']>(h-1)*60)&(r['lead_minutes']<=h*60)) for h in range(1,5)]
    for name,mask in bins:
        mm=m&mask
        if mm.any():groups.append(dict(station=st,group=name,**met(mm)))
    for day in sorted(set(dates[m])):
        mm=m&(dates==day);v=met(mm); daily.append(dict(station=st,date=day,**v,total_delta_sse=float(((b[:,mm]-g[mm])**2-(a[:,mm]-g[mm])**2).sum())))
    # block-CI per station, all seeds share dates
    days=sorted(set(dates[m])); ix=np.searchsorted(days,dates[m]);den=np.bincount(ix,minlength=len(days));sse=np.array([[np.bincount(ix,weights=(pr[m]-g[m])**2,minlength=len(days)) for pr in x] for x in [a,b,z]])
    w=np.random.default_rng(20261022).multinomial(len(days),np.ones(len(days))/len(days),5000);rm=np.sqrt(np.einsum('asd,bd->asb',sse,w)/(w@den)[None,None,:]).mean(1)
    out[st]['delta_AC_A_ci95']=np.quantile(rm[1]-rm[0],[.025,.975]).tolist();out[st]['direct_COT_delta_ci95']=np.quantile(rm[1]-rm[2],[.025,.975]).tolist()
    bylead=[]
    for h in range(1,5):
        mm=m&(r['lead_minutes']>(h-1)*60)&(r['lead_minutes']<=h*60)&validtarget&ph['last_valid']
        if mm.sum()>5:
            c0=ph['init_cot'][mm];ct=ph['target_cot'][mm]
            bylead.append(dict(hour=h,n=int(mm.sum()),median_abs_cot_change=float(np.median(abs(ct-c0))),same_R_center_COT_persistence_rmse=float(np.sqrt(((ct-c0)**2).mean())),initCOT_vs_targetkt_correlation=float(np.corrcoef(c0,kt[mm])[0,1]),targetCOT_vs_targetkt_correlation=float(np.corrcoef(ct,kt[mm])[0,1])))
    out[st]['offline_COT_changes']=bylead
for name,rows in [('groups',groups),('daily',daily)]:
    with (P/f'{name}.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(P/'DECOMPOSITION.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
