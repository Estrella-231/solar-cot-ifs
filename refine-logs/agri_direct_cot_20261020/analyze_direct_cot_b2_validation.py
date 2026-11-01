from __future__ import annotations
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

score=Path(sys.argv[1]); workers=Path(sys.argv[2]); out=score/'figures'; out.mkdir(parents=True,exist_ok=True)
m=json.loads((score/'direct_cot_ghi_metrics.json').read_text(encoding='utf-8'))
stations=('sili','zhujia'); arms=('none','simvp_r','p_c','p_xc'); seeds=(42,43,44)
labels={'none':'No COT','simvp_r':'SimVP→R COT','p_c':'Direct COT (P-C)','p_xc':'Direct COT (P-XC)'}
colors={'none':'#607d8b','simvp_r':'#8e6bbf','p_c':'#d98324','p_xc':'#008f83'}
# Per-lead scores: each seed is scored first; summary shows the seed mean and range.
lead_rows=[]; overall_rows=[]; strata_rows=[]
for s in stations:
  key_ref=None
  pred_by={}
  for a in arms:
    for seed in seeds:
      run=workers/f'{s}_worker'/s/a/f'seed{seed}'
      with np.load(run/'validation_predictions.npz',allow_pickle=False) as z:
        pred=np.asarray(z['prediction_ghi_wm2'],np.float64)
        truth=np.asarray(z['truth_ghi_wm2'],np.float64)
        raw=np.asarray(z['keys_jsonl_bytes'],dtype=np.uint8).tobytes()
      keys=[json.loads(line) for line in raw.decode('utf-8').splitlines() if line]
      if key_ref is None: key_ref=keys
      assert keys==key_ref and np.array_equal(truth, np.asarray([float(x) for x in truth]))
      pred_by[(a,seed)]=pred
      overall=m['point_metrics'][s][a][str(seed)]['all']['rmse_wm2']
      overall_rows.append({'station':s,'arm':a,'label':labels[a],'seed':seed,'rmse_wm2':overall,'n':len(truth)})
  leads=np.asarray([int(k['lead_minutes']) for k in key_ref])
  for lead in sorted(set(leads)):
    vals={}
    for a in arms:
      vals[a]=[]
      sel=leads==lead
      for seed in seeds:
        pred=pred_by[(a,seed)]
        vals[a].append(float(np.sqrt(np.mean((pred[sel]-truth[sel])**2))))
    for a in arms:
      lead_rows.append({'station':s,'lead_minutes':int(lead),'arm':a,'label':labels[a],
                        'rmse_seed_mean':float(np.mean(vals[a])),'rmse_seed_min':float(np.min(vals[a])),
                        'rmse_seed_max':float(np.max(vals[a])),'n':int(sel.sum())})
    for base in ('none','simvp_r'):
      d=[vals['p_xc'][i]-vals[base][i] for i in range(3)]
      lead_rows.append({'station':s,'lead_minutes':int(lead),'arm':f'p_xc-{base}','label':f'P-XC minus {labels[base]}',
                        'rmse_seed_mean':float(np.mean(d)),'rmse_seed_min':float(np.min(d)),
                        'rmse_seed_max':float(np.max(d)),'n':int(sel.sum())})
  for b in ('cot_0_1','cot_1_10','cot_10_30','cot_ge30'):
    n=m['point_metrics'][s]['p_xc']['42']['by_cpp_center_bin'][b]['n']
    ds=[]
    for seed in seeds:
      c=m['point_metrics'][s]['p_xc'][str(seed)]['by_cpp_center_bin'][b]['rmse_wm2']
      r=m['point_metrics'][s]['none'][str(seed)]['by_cpp_center_bin'][b]['rmse_wm2']
      ds.append(float(c-r))
    strata_rows.append({'station':s,'cpp_center_bin':b,'n':n,'rmse_delta_seed_mean':float(np.mean(ds)),
                        'rmse_delta_seed_min':float(np.min(ds)),'rmse_delta_seed_max':float(np.max(ds))})
for name, rows in (('overall_seed_metrics.csv',overall_rows),('leadwise_metrics.csv',lead_rows),('cpp_strata_deltas.csv',strata_rows)):
  with (out/name).open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
fig,axs=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
# A: absolute RMSE; zero baseline retained.
ax=axs[0,0]; x=np.arange(2); width=.19
for j,a in enumerate(arms):
  means=[]; sd=[]
  for s in stations:
    v=[r['rmse_wm2'] for r in overall_rows if r['station']==s and r['arm']==a]
    means.append(np.mean(v));sd.append(np.std(v,ddof=1))
  ax.bar(x+(j-1.5)*width,means,width,yerr=sd,capsize=3,color=colors[a],label=labels[a])
ax.set_xticks(x,stations);ax.set_ylabel('GHI RMSE (W m$^{-2}$)');ax.set_title('A  Full 16-lead validation RMSE')
ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.25);ax.legend(fontsize=8,ncol=2)
# B/C: delta curves, seed range is descriptive, not confidence interval.
for ax,base,title in ((axs[0,1],'none','B  P-XC minus no COT'),(axs[1,0],'simvp_r','C  P-XC minus SimVP→R COT')):
  for s,c in zip(stations,('#1769aa','#c64e4e')):
    rs=[r for r in lead_rows if r['station']==s and r['arm']==f'p_xc-{base}']
    ls=np.asarray([r['lead_minutes'] for r in rs]); mean=np.asarray([r['rmse_seed_mean'] for r in rs]); lo=np.asarray([r['rmse_seed_min'] for r in rs]); hi=np.asarray([r['rmse_seed_max'] for r in rs])
    order=np.argsort(ls);ls,mean,lo,hi=ls[order],mean[order],lo[order],hi[order]
    ax.plot(ls,mean,'o-',color=c,label=s,linewidth=1.8,markersize=4);ax.fill_between(ls,lo,hi,color=c,alpha=.12)
  ax.axhline(0,color='black',linewidth=1);ax.set_xlabel('Forecast lead (min)');ax.set_ylabel('RMSE difference (W m$^{-2}$)')
  ax.set_title(title+'  (positive = worse)');ax.grid(alpha=.25);ax.legend()
# D: stratify by CPP center reference; these are descriptive subgroup deltas.
ax=axs[1,1];bins=('cot_0_1','cot_1_10','cot_10_30','cot_ge30');names=('0–1','1–10','10–30','≥30');x=np.arange(4);w=.34
for j,(s,c) in enumerate(zip(stations,('#1769aa','#c64e4e'))):
  rr=[next(r for r in strata_rows if r['station']==s and r['cpp_center_bin']==b) for b in bins]
  vals=[r['rmse_delta_seed_mean'] for r in rr]; low=[v-r['rmse_delta_seed_min'] for v,r in zip(vals,rr)]; high=[r['rmse_delta_seed_max']-v for v,r in zip(vals,rr)]
  ax.bar(x+(j-.5)*w,vals,w,yerr=np.array([low,high]),capsize=3,color=c,label=s)
  for xx,r in zip(x+(j-.5)*w,rr): ax.text(xx,r['rmse_delta_seed_mean']+(1.0 if r['rmse_delta_seed_mean']>=0 else -1.5),f"n={r['n']}",ha='center',va='bottom' if r['rmse_delta_seed_mean']>=0 else 'top',fontsize=7,rotation=0)
ax.axhline(0,color='black',linewidth=1);ax.set_xticks(x,names);ax.set_xlabel('CPP reference COT at station center');ax.set_ylabel('P-XC minus no-COT RMSE (W m$^{-2}$)')
ax.set_title('D  CPP-supported subgroup differences\n(seed range; diagnostic, not separate significance tests)');ax.grid(axis='y',alpha=.25);ax.legend()
fig.suptitle('Forecast-COT GHI value: fixed H/V validation comparison',fontsize=14)
for ext in ('png','pdf'): fig.savefig(out/f'direct_cot_ghi_b2_validation.{ext}',dpi=240,bbox_inches='tight')
manifest={'state':'COMPLETE_DIRECT_COT_B2_VALIDATION_FIGURES','score_json_sha256':hashlib.sha256((score/'direct_cot_ghi_metrics.json').read_bytes()).hexdigest(),
          'figure_png_sha256':hashlib.sha256((out/'direct_cot_ghi_b2_validation.png').read_bytes()).hexdigest(),
          'figure_pdf_sha256':hashlib.sha256((out/'direct_cot_ghi_b2_validation.pdf').read_bytes()).hexdigest(),
          'seed_range_is_not_confidence_interval':True,'cpp_strata_are_reference_only':True}
(out/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps(manifest))
