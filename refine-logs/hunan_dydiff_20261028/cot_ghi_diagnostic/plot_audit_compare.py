"""Explain the coverage mismatch behind the questioned5percent comparison."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).parent;m=json.loads((B/'NUMERIC_AUDIT.json').read_text())['coverage_comparison']
keys=['full_validation_observed_same_heads','same_432_observed','same_432_forecast'];labels=['Observed images\nFull validation: 4,226','Observed images\nSelected 432','DyDiff forecasts\nSame selected 432']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
fig,ax=plt.subplots(figsize=(8.6,4.7),layout='constrained');x=np.arange(3);width=.31
for offset,col,field,label in [(-width/2,'#607D8B','before_RMSE','Without COT'),(width/2,'#00897B','after_RMSE','With COT')]:
 vals=[m[k][field] for k in keys];bars=ax.bar(x+offset,vals,width,label=label,color=col)
 for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+1,f'{v:.2f}',ha='center',va='bottom',fontsize=9)
for i,k in enumerate(keys):ax.text(i,max(m[k]['before_RMSE'],m[k]['after_RMSE'])+10,f"{m[k]['relative_reduction_pct']:.2f}% lower",ha='center',fontsize=10)
ax.set(xticks=x,xticklabels=labels,ylabel='GHI RMSE (W m$^{-2}$)',ylim=(0,165));ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
ax.legend(frameon=False,loc='upper left');ax.set_title('Same heads, different coverage: the percentages are not interchangeable',fontsize=11)
fig.supxlabel('Selected windows: 15 of 16 initializations at 08:00 BJT.\nThe observed-vs-forecast COT-effect difference is unresolved (paired date CI crosses zero).',fontsize=8)
out=B/'attempt1/figures'
for ext in ('png','pdf'):fig.savefig(out/f'audit_same_heads_different_coverage.{ext}',bbox_inches='tight')
print(out)
