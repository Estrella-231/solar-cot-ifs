"""Paired saved-output diagnosis, not a causal COT ablation."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]/'results/regional_station_cot_20260927/ghi_pilot'
out=root/'zhujia_diagnostic'; out.mkdir(exist_ok=True)
a={k:dict(np.load(root/f'{k}_validation_predictions.npz')) for k in ['zero','forecast','oracle']}
for k in ['forecast','oracle']:
    for field in ['row_ids','true_ghi','station','lead']:
        assert np.array_equal(a['zero'][field],a[k][field])
m=a['zero']['station']==1
y=a['zero']['true_ghi'][m]; lead=a['zero']['lead'][m]
p={k:v['pred_ghi'][m] for k,v in a.items()}
e={k:v-y for k,v in p.items()}
report={'n':len(y),'scope':'matched Zhujia validation pilot; independent trained heads, not causal input ablation','arms':{}}
for k in ['forecast','oracle']:
    d=p[k]-p['zero']; ds=e[k]**2-e['zero']**2
    order=np.argsort(ds)[::-1]
    report['arms'][k]={'mean_adjustment':float(d.mean()),'mean_abs_adjustment':float(abs(d).mean()),
      'fraction_improved':float(np.mean(ds<0)), 'delta_mse':float(ds.mean()),
      'bias_squared_change':float(e[k].mean()**2-e['zero'].mean()**2),
      'centered_error_variance_change':float(e[k].var()-e['zero'].var()),
      'baseline_overprediction_adjustment':float(d[e['zero']>0].mean()),
      'baseline_underprediction_adjustment':float(d[e['zero']<0].mean()),
      'top_5pct_delta_mse_contribution':float(ds[order[:int(np.ceil(.05*len(y)))]].sum()/len(y)),
      'leads':[{ 'minutes':15*(i+1),'n':int((lead==i).sum()),'delta_rmse':float(np.sqrt(np.mean(e[k][lead==i]**2))-np.sqrt(np.mean(e['zero'][lead==i]**2)))} for i in range(16)],
      'worst_rows':[{'row_id':int(a['zero']['row_ids'][m][j]),'lead_minutes':15*(int(lead[j])+1),'truth':float(y[j]),'zero':float(p['zero'][j]),'candidate':float(p[k][j]),'delta_squared_error':float(ds[j])} for j in order[:12]]}
(out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
fig,ax=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
for k,c in [('zero','#555555'),('forecast','#0072B2'),('oracle','#D55E00')]:
    ax[0].plot(np.arange(1,17)*15,[np.sqrt(np.mean(e[k][lead==i]**2)) for i in range(16)],marker='o',label=k,color=c)
ax[0].set(xlabel='Forecast lead (min)',ylabel='Zhujia RMSE (W/m²)'); ax[0].legend()
ax[1].scatter(e['zero'],p['forecast']-p['zero'],s=7,alpha=.25,color='#0072B2')
ax[1].axhline(0,color='k',lw=.6);ax[1].axvline(0,color='k',lw=.6)
ax[1].set(xlabel='Zero-COT prediction minus observation (W/m²)',ylabel='Forecast-COT minus zero-COT prediction (W/m²)')
fig.suptitle('Actual paired Zhujia validation outputs; independently trained heads')
fig.savefig(out/'paired_diagnosis.png',dpi=160); plt.close(fig)
print(json.dumps(report,indent=2))
