from pathlib import Path
import csv
import matplotlib.pyplot as plt
p=Path(__file__).parent
with (p/'lead_metrics_fp64.csv').open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
colors={'sili':'#0072B2','zhujia':'#D55E00'}
labels={'sili':'Sili','zhujia':'Zhujia'}
fig,ax=plt.subplots(figsize=(8.4,4.6),layout='constrained')
for s in ('sili','zhujia'):
 q=sorted((r for r in rows if r['station']==s),key=lambda r:int(r['lead_min']))
 x=[int(r['lead_min']) for r in q];y=[float(r['delta_rmse']) for r in q]
 marker='o' if s=='sili' else 's'
 ax.plot(x,y,color=colors[s],marker=marker,markersize=4.4,linewidth=1.8,label=labels[s])
ax.axhline(0,color='#333333',linewidth=1.0,linestyle='--')
ax.set(xlim=(10,245),xticks=[15,45,75,105,135,165,195,225,240],xlabel='Forecast lead from history end (min)',ylabel='Δ GHI RMSE (W m⁻²)\nforecast COT − same-time R-COT',title='Fixed-head sensitivity to forecast COT under oracle AGRI')
ax.grid(True,color='#d9d9d9',linewidth=.6,alpha=.75)
ax.legend(frameon=False,ncol=2,loc='upper left')
ax.text(.99,.03,'Validation only · 7,005 overlapping forecast instances · 8 BJT dates\nActual target AGRI and geometry held fixed; not deployable GHI skill',transform=ax.transAxes,ha='right',va='bottom',fontsize=8.4,color='#444444')
fig.savefig(p/'forecast_cot_fixed_head_lead_sensitivity.png',dpi=220,facecolor='white')
fig.savefig(p/'forecast_cot_fixed_head_lead_sensitivity.svg',facecolor='white')
print('saved',p/'forecast_cot_fixed_head_lead_sensitivity.png')

