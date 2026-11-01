"""Summarize dual-path COT H-CV from saved OOF arrays and paired scores."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

STATIONS=('sili','zhujia'); ARMS=('none','pxc','simvp_r','dual'); SEEDS=(42,43,44)
LABELS={'none':'No COT','pxc':'Direct P-XC COT','simvp_r':'SimVP→R COT','dual':'Dual-path disagreement'}

def load_arm(run,st,fold,arm,seed):
    p=run/st/f'fold{fold}'/arm/f'seed{seed}'
    keys=[json.loads(x) for x in (p/'oof_keys.jsonl').read_text(encoding='utf-8').splitlines()]
    with np.load(p/'oof_predictions.npz',allow_pickle=False) as z:
        pred=z['prediction_ghi'].astype(np.float64); truth=z['truth_ghi'].astype(np.float64)
    if len(keys)!=len(pred) or len(pred)!=len(truth): raise ValueError(f'OOF length mismatch: {st}/{fold}/{arm}/{seed}')
    return keys,pred,truth

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run',type=Path,required=True); ap.add_argument('--score',type=Path,required=True); ap.add_argument('--outdir',type=Path,required=True); a=ap.parse_args()
    manifest=json.loads((a.run/'run_manifest.json').read_text(encoding='utf-8'))
    score=json.loads(a.score.read_text(encoding='utf-8'))
    if manifest.get('state')!='COMPLETE_COT_DUALPATH_H_CV' or score.get('state')!='COMPLETE_COT_DUALPATH_H_CV_SCORE': raise RuntimeError('complete run and score receipts required')
    data={}
    for st in STATIONS:
      for arm in ARMS:
       for seed in SEEDS:
        pp=[]; yy=[]; leads=[]
        for fold in range(5):
            keys,pred,truth=load_arm(a.run,st,fold,arm,seed)
            pp.extend(pred); yy.extend(truth); leads.extend(int(k['lead_minutes']) for k in keys)
        data[(st,arm,seed)]={'pred':np.asarray(pp),'truth':np.asarray(yy),'lead':np.asarray(leads)}
    metrics={}
    for arm in ARMS:
      metrics[arm]={}
      for scope,maskfn in (('all_leads',lambda x:np.ones(len(x),dtype=bool)),('plus15_60',lambda x:x<=60)):
        rows=[]
        for st in STATIONS:
          for seed in SEEDS:
            d=data[(st,arm,seed)]; use=maskfn(d['lead']); e=d['pred'][use]-d['truth'][use]
            rows.append({'station':st,'seed':seed,'rmse_wm2':float(np.sqrt(np.mean(e*e))),'mae_wm2':float(np.mean(np.abs(e))),'bias_wm2':float(np.mean(e))})
        metrics[arm][scope]={k:float(np.mean([r[k] for r in rows])) for k in ('rmse_wm2','mae_wm2','bias_wm2')}; metrics[arm][scope]['station_seed']=rows
    out={'state':'COMPLETE_COT_DUALPATH_H_CV_SUMMARY','evidence_tier':score['evidence_tier'],'n_models':len(manifest['runs']),'station_rows':score['station_rows'],'absolute_metrics':metrics,'paired_contrasts':score['results'],'test_payload_opened':False,'buffer_payload_opened':False,'future_observed_agri_used_as_input':False,'cpp_used_as_input':False}
    a.outdir.mkdir(parents=True,exist_ok=True); (a.outdir/'dualpath_hcv_summary.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    colors=['#667f8c','#2676a8','#d98b28','#008f78']; x=np.arange(2); w=.19
    fig,(ax0,ax1)=plt.subplots(1,2,figsize=(13,5.2)); fig.subplots_adjust(left=.07,right=.985,bottom=.20,top=.75,wspace=.30)
    for j,(arm,color) in enumerate(zip(ARMS,colors)):
        vals=[metrics[arm][sc]['rmse_wm2'] for sc in ('all_leads','plus15_60')]
        ax0.bar(x+(j-1.5)*w,vals,width=w,color=color,label=LABELS[arm])
        for xx,v in zip(x+(j-1.5)*w,vals): ax0.text(xx,v+.5,f'{v:.1f}',ha='center',va='bottom',fontsize=8)
    ax0.set_xticks(x,['All leads\n+15 to +240 min','Short leads\n+15 to +60 min']); ax0.set_ylabel('Station/seed-equal GHI RMSE (W m$^{-2}$)'); ax0.set_title('H-only out-of-fold absolute error'); ax0.grid(axis='y',alpha=.22); ax0.set_axisbelow(True)
    pairs=('dual_minus_pxc','dual_minus_none','dual_minus_simvp_r'); scopes=('all_leads','plus15_60'); offsets=np.arange(3)
    for i,pair in enumerate(pairs):
        for j,scope in enumerate(scopes):
            z=score['results'][pair][scope]; y=i*3+j
            ax1.errorbar(z['delta_rmse_wm2'],y,xerr=[[z['delta_rmse_wm2']-z['ci95_low']],[z['ci95_high']-z['delta_rmse_wm2']]],fmt='o',color=colors[-1] if i==0 else '#59636a',capsize=3)
    ax1.axvline(0,color='black',lw=1); ax1.set_yticks([0,1,3,4,6,7],['Dual−P-XC, all','Dual−P-XC, +15–60','Dual−no COT, all','Dual−no COT, +15–60','Dual−SimVP→R, all','Dual−SimVP→R, +15–60']); ax1.invert_yaxis(); ax1.set_xlabel('Paired ΔRMSE (W m$^{-2}$), positive = worse'); ax1.set_title('Paired two-day-block 95% intervals'); ax1.grid(axis='x',alpha=.22); ax1.set_axisbelow(True)
    fig.suptitle('Can forecast-path disagreement help the GHI head use COT?',fontsize=13,y=.97); fig.legend([plt.Rectangle((0,0),1,1,color=c) for c in colors],[LABELS[a0] for a0 in ARMS],loc='upper center',bbox_to_anchor=(.5,.90),ncol=4,frameon=False,fontsize=9)
    p=score['results']['dual_minus_pxc']['all_leads']; fig.text(.5,.045,f"Primary dual−P-XC: {p['delta_rmse_wm2']:+.2f} W m$^{{-2}}$ (95% CI {p['ci95_low']:+.2f}, {p['ci95_high']:+.2f}); 112 two-day blocks, 5,000 resamples. H-only internal CV; not independent test.",ha='center',fontsize=8)
    fig.savefig(a.outdir/'dualpath_hcv_summary.png',dpi=220); fig.savefig(a.outdir/'dualpath_hcv_summary.pdf'); print(json.dumps({'state':out['state'],'primary':score['results']['dual_minus_pxc'],'gate':score['results']['predeclared_gate']},indent=2))

if __name__=='__main__': main()
