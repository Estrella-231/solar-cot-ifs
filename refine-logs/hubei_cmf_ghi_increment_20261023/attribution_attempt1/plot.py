"""Descriptive equal-site/equal-seed means; no uncertainty inferred from seeds."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
r=json.loads((P/'ANALYSIS.json').read_text());groups=list(csv.DictReader((P/'groups.csv').open()))
out=P/'figures';out.mkdir(exist_ok=True)
with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42}):
    fig,axs=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
    colors=['#0072B2','#D55E00'];roles=['C_fit','C_select'];labels=['Fitting dates','Selection dates']
    for i,(role,label) in enumerate(zip(roles,labels)):
        x=np.arange(2)+(-.18 if i==0 else .18)
        vals=[r['summary'][role][a]['rmse'] for a in ['A','M']]
        axs[0].bar(x,vals,.34,label=label,color=colors[i],hatch='' if i==0 else '//',alpha=.85)
    axs[0].set(xticks=[0,1],xticklabels=['Frozen baseline','+ COT correction'],ylabel='GHI RMSE (W m$^{-2}$)',ylim=(0,140),title='(a) Improvement does not transfer')
    axs[0].legend(frameon=False,fontsize=9)
    x=np.arange(2)
    for j,(arm,label) in enumerate([('A','Baseline error'),('M','Corrected error')]):
        vals=[r['summary'][role][arm]['bias'] for role in roles]
        axs[1].bar(x+(-.18 if j==0 else .18),vals,.34,color=colors[j],label=label,hatch='' if j==0 else '//')
    axs[1].axhline(0,color='black',lw=.8)
    axs[1].set(xticks=x,xticklabels=labels,ylabel='Mean prediction minus observation (W m$^{-2}$)',ylim=(-65,85),title='(b) Baseline bias changes sign')
    axs[1].legend(frameon=False,fontsize=9)
    bins=['cot_0to3','cot_3to10','cot_10to30','cot_30to101'];x=np.arange(4)
    for role,label,color,marker in zip(roles,labels,colors,['o','s']):
        def means(key):
            return [np.mean([float(v[key]) for v in groups if v['role']==role and v['arm']=='M' and v['group']==g]) for g in bins]
        axs[2].plot(x,means('base_bias'),color=color,marker=marker,label=label+': baseline bias')
        axs[2].plot(x,means('mean_adjustment'),color=color,marker=marker,ls='--',label=label+': correction')
    axs[2].axhline(0,color='black',lw=.8)
    axs[2].set(xticks=x,xticklabels=['0–3','3–10','10–30','≥30'],xlabel='Predicted station-centre COT',ylabel='GHI bias or adjustment (W m$^{-2}$)',ylim=(-65,85),title='(c) Similar COT, different error sign')
    axs[2].legend(frameon=False,fontsize=7,loc='upper left')
    for ax in axs:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Frozen coefficients: fitting dates vs held-out selection dates',fontsize=13)
    fig.savefig(out/'fit_select_transfer.png',dpi=180)
    fig.savefig(out/'fit_select_transfer.pdf')
    plt.close(fig)
(out/'FIGURE_SCOPE.txt').write_text('Descriptive means over three stations and three seeds, each scored separately. C_fit/C_select are reused training-development dates, not independent test. COT bins are predictions, not CPP-reference classes. No samples removed; no smoothing. Lines join bin summaries only. See groups.csv for support counts and per-site scores.\n')
print(out/'fit_select_transfer.png')
