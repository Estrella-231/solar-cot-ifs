"""Training/development descriptors, not a controlled generalization estimate."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_cot_event_timing import read_csv,sha

B=Path(__file__).resolve().parent
D=B/'results/u_train_fit_diagnostic_v1_20261021'
rr=read_csv(D/'u_v_descriptive_comparison.csv')
uu=read_csv(D/'u_first_difference_metrics.csv')
vv=read_csv(B/'results/dense_r_field_event_diagnostic_v1_20261021/first_difference_metrics.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained',sharex=True)
for col,st in enumerate(('sili','zhujia')):
    for part,label,color,style,marker in [('U_training_sample','Training U, fixed sample','#0072B2','-','o'),
                                         ('V_development','Development V','#D55E00','--','s')]:
        rows=sorted([r for r in rr if r['partition']==part and r['station']==st and r['arm']=='cot_p_xc'
                     and r['scale']=='station3' and r['lead_minutes']!='all'],key=lambda r:int(r['lead_minutes']))
        x=[int(r['lead_minutes']) for r in rows]
        for row,key in enumerate(('rmse','thick_pod')):
            axs[row,col].plot(x,[float(r[key]) for r in rows],style,color=color,marker=marker,ms=3,label=label)
    axs[0,col].set(title=st.title(),ylabel='COT RMSE vs R reference',ylim=(0,18))
    axs[1,col].set(ylabel='Thick-cloud POD (COT ≥ 30)',ylim=(0,1),xlabel='Lead from history end (min)',xticks=[15,60,120,180,240])
    for ax in axs[:,col]:ax.grid(alpha=.2)
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=2)
fig.suptitle('Existing direct-COT model: training fit and development descriptors\n'
             'Same epoch-12 weights; different date/support distributions. This is not independent confirmation.',fontsize=12)
fig.savefig(D/'u_v_lead_descriptors.png',dpi=170);fig.savefig(D/'u_v_lead_descriptors.pdf');plt.close(fig)

rows=[r for r in uu if r['station']=='both' and r['scale']=='station3' and r['arm']=='cot_p_xc']
rows += [r for r in vv if r['station']=='both' and r['scale']=='station3' and r['arm']=='cot_p_xc']
fig,axs=plt.subplots(1,2,figsize=(9,4.2),layout='constrained')
labels=['Training U sample','Development V']
for ax,key,title in zip(axs,('sd_ratio','corr'),('15-min change amplitude / reference','15-min change correlation')):
    yy=[float(r[key]) for r in rows]
    bars=ax.bar(labels,yy,color=['#0072B2','#D55E00'],hatch=['','//'],edgecolor='white')
    ax.set(ylim=(0,1),title=title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    for bar,y in zip(bars,yy):ax.text(bar.get_x()+bar.get_width()/2,y+.015,f'{y:.3f}',ha='center')
fig.suptitle('Temporal changes are damped within the training sample too\n'
             'Descriptive pixel pairs; no GHI score or new model training',fontsize=12)
fig.savefig(D/'u_v_temporal_changes.png',dpi=170);fig.savefig(D/'u_v_temporal_changes.pdf');plt.close(fig)
receipt={'state':'GENERATED_U_FIT_DESCRIPTIVE_FIGURES','script_sha256':sha(Path(__file__)),
         'source_sha256':{p.name:sha(p) for p in [D/'u_v_descriptive_comparison.csv',D/'u_first_difference_metrics.csv',
                         B/'results/dense_r_field_event_diagnostic_v1_20261021/first_difference_metrics.csv']},
         'scope':'descriptive U/V; different support distributions; no unbiased generalization or GHI claim'}
(D/'figure_provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
