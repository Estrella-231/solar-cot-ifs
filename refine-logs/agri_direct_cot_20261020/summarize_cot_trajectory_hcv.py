"""Produce reproducible absolute metrics and a compact figure for H-only COT-trajectory CV."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

STATIONS=('sili','zhujia'); ARMS=('none','target','trajectory'); SEEDS=(42,43,44)

def load(run):
    manifest=json.loads((run/'run_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('state')!='COMPLETE_COT_TRAJECTORY_H_CV': raise RuntimeError('complete training receipt required')
    if any(manifest.get(k) is not False for k in ('test_payload_opened','buffer_payload_opened','future_observed_agri_used_as_input','cpp_used_as_input')): raise RuntimeError('forbidden data-source flag')
    d={}
    for st in STATIONS:
      for fold in range(5):
       for arm in ARMS:
        for seed in SEEDS:
         p=run/st/f'fold{fold}'/arm/f'seed{seed}'
         keys=[json.loads(x) for x in (p/'oof_keys.jsonl').read_text(encoding='utf-8').splitlines()]
         with np.load(p/'oof_predictions.npz',allow_pickle=False) as z:
          pred=z['prediction_ghi'].astype(np.float64); truth=z['truth_ghi'].astype(np.float64)
         if len(keys)!=len(pred) or len(pred)!=len(truth): raise ValueError(f'key mismatch: {st}/{fold}/{arm}/{seed}')
         d[(st,fold,arm,seed)]=(keys,pred,truth)
    return manifest,d

def metrics(pred,truth):
    e=pred-truth
    return {'rmse_wm2':float(np.sqrt(np.mean(e*e))),'mae_wm2':float(np.mean(np.abs(e))),'bias_wm2':float(np.mean(e)),'n':int(len(e))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run',type=Path,required=True); ap.add_argument('--score',type=Path,required=True); ap.add_argument('--outdir',type=Path,required=True); a=ap.parse_args()
    manifest,d=load(a.run); score=json.loads(a.score.read_text(encoding='utf-8'))
    if score.get('state')!='COMPLETE_COT_TRAJECTORY_H_CV_SCORE': raise RuntimeError('complete paired score required')
    out={'state':'COMPLETE_COT_TRAJECTORY_H_CV_SUMMARY','evidence_tier':score['evidence_tier'],'n_models':len(manifest['runs']),'stations':{},'station_seed_equal':{},'leadwise_descriptive':[],
         'paired_score':score['results'],'test_payload_opened':False,'buffer_payload_opened':False}
    full_by={arm:[] for arm in ARMS}; short_by={arm:[] for arm in ARMS}
    for st in STATIONS:
      out['stations'][st]={}
      for arm in ARMS:
       for seed in SEEDS:
        parts=[]; short=[]
        for fold in range(5):
         keys,pred,truth=d[(st,fold,arm,seed)]; parts.append((pred,truth))
         ix=np.asarray([int(k['lead_minutes'])<=60 for k in keys]); short.append((pred[ix],truth[ix]))
        p=np.concatenate([x[0] for x in parts]); y=np.concatenate([x[1] for x in parts]); ps=np.concatenate([x[0] for x in short]); ys=np.concatenate([x[1] for x in short])
        allm=metrics(p,y); shortm=metrics(ps,ys)
        out['stations'][st].setdefault(arm,{'seeds':{},'mean_seed_rmse_all':None,'mean_seed_rmse_plus15_60':None})['seeds'][str(seed)]={'all_leads':allm,'plus15_60':shortm}
        full_by[arm].append(allm); short_by[arm].append(shortm)
      for arm in ARMS:
       out['stations'][st][arm]['mean_seed_rmse_all']=float(np.mean([out['stations'][st][arm]['seeds'][str(s)]['all_leads']['rmse_wm2'] for s in SEEDS]))
       out['stations'][st][arm]['mean_seed_rmse_plus15_60']=float(np.mean([out['stations'][st][arm]['seeds'][str(s)]['plus15_60']['rmse_wm2'] for s in SEEDS]))
    out['station_seed_equal']={}
    for arm in ARMS:
      out['station_seed_equal'][arm]={'all_leads':{'rmse_wm2':float(np.mean([x['rmse_wm2'] for x in full_by[arm]])),'mae_wm2':float(np.mean([x['mae_wm2'] for x in full_by[arm]])),'bias_wm2':float(np.mean([x['bias_wm2'] for x in full_by[arm]]))},
       'plus15_60':{'rmse_wm2':float(np.mean([x['rmse_wm2'] for x in short_by[arm]])),'mae_wm2':float(np.mean([x['mae_wm2'] for x in short_by[arm]])),'bias_wm2':float(np.mean([x['bias_wm2'] for x in short_by[arm]]))}}
    for lead in range(15,241,15):
      rec={'lead_minutes':lead}
      for arm in ARMS:
       ms=[]
       for st in STATIONS:
        for seed in SEEDS:
         pp=[]; yy=[]
         for fold in range(5):
          keys,pred,truth=d[(st,fold,arm,seed)]; ix=[i for i,k in enumerate(keys) if int(k['lead_minutes'])==lead]; pp.extend(pred[ix]); yy.extend(truth[ix])
         ms.append(metrics(np.asarray(pp),np.asarray(yy))['rmse_wm2'])
       rec[arm+'_rmse_wm2']=float(np.mean(ms))
      rec['trajectory_minus_target_wm2']=rec['trajectory_rmse_wm2']-rec['target_rmse_wm2']; out['leadwise_descriptive'].append(rec)
    a.outdir.mkdir(parents=True,exist_ok=True)
    (a.outdir/'trajectory_hcv_summary.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    cols=['#667f8c','#d98b28','#008f78']; labels=['No COT','Target-frame COT','4-map COT trajectory']
    fig,(ax0,ax1)=plt.subplots(1,2,figsize=(12,5.1))
    fig.subplots_adjust(left=.08,right=.985,bottom=.19,top=.76,wspace=.30)
    x=np.arange(2); w=.23
    for j,(arm,label,color) in enumerate(zip(ARMS,labels,cols)):
      vals=[out['station_seed_equal'][arm]['all_leads']['rmse_wm2'],out['station_seed_equal'][arm]['plus15_60']['rmse_wm2']]
      ax0.bar(x+(j-1)*w,vals,width=w,color=color,label=label)
      for xx,v in zip(x+(j-1)*w,vals): ax0.text(xx,v+0.5,f'{v:.1f}',ha='center',va='bottom',fontsize=8)
    ax0.set_xticks(x,['All leads\n+15 to +240 min','Short leads\n+15 to +60 min']); ax0.set_ylabel('Station/seed-equal GHI RMSE (W m$^{-2}$)'); ax0.set_title('H-only out-of-fold prediction'); ax0.grid(axis='y',alpha=.22); ax0.set_axisbelow(True)
    leads=[r['lead_minutes'] for r in out['leadwise_descriptive']]; delta=[r['trajectory_minus_target_wm2'] for r in out['leadwise_descriptive']]
    ax1.axhline(0,color='black',lw=1); ax1.plot(leads,delta,color=cols[2],marker='o',lw=1.8); ax1.set_xticks([15,60,120,180,240]); ax1.set_xlabel('Forecast lead (min)'); ax1.set_ylabel('Trajectory − target-frame RMSE (W m$^{-2}$)'); ax1.set_title('Leadwise descriptive difference'); ax1.grid(alpha=.22); ax1.set_axisbelow(True)
    fig.suptitle('Predicted-COT trajectory does not improve GHI in H-CV',fontsize=13,y=.97)
    fig.legend(handles=[plt.Rectangle((0,0),1,1,color=c) for c in cols],labels=labels,frameon=False,loc='upper center',bbox_to_anchor=(.5,.90),ncol=3,fontsize=9)
    ci=score['results']['trajectory_minus_target']['all_leads']
    fig.text(.5,.035,f"Positive means worse. Primary paired two-day-block difference: {ci['delta_rmse_wm2']:+.2f} W m$^{{-2}}$ (95% CI {ci['ci95_low']:+.2f}, {ci['ci95_high']:+.2f}); 112 blocks, 5,000 resamples. H-only internal CV, not independent test.",ha='center',fontsize=8)
    fig.savefig(a.outdir/'trajectory_hcv_summary.png',dpi=220); fig.savefig(a.outdir/'trajectory_hcv_summary.pdf')
    print(json.dumps({'state':out['state'],'station_seed_equal':out['station_seed_equal'],'primary':score['results']['trajectory_minus_target'],'gate':score['results']['predeclared_gate']},indent=2))

if __name__=='__main__': main()
