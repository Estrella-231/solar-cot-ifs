"""Secondary row-weighted scoring and fixed true-cloud case visualization."""
import csv,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

here=Path(__file__).resolve().parent;repo=here.parents[1]
score=json.loads((here/'analysis/independent_score.json').read_text())
with np.load(here/'analysis/paired_validation_scoring.npz',allow_pickle=False) as data:
    z={key:data[key] for key in data.files}
p=z['prediction'];y=z['truth'];st=z['station'];date=z['dates'];arms=score['arms']
groups={'all':np.ones(len(y),bool),'high_transmittance':(z['clear']>=100)&(z['kt']>=.9),'low_kt':(z['clear']>=100)&(z['kt']<.3)}
days=np.unique(date);di=np.searchsorted(days,date);D=len(days)
w=np.random.default_rng(20261016).multinomial(D,np.ones(D)/D,2000)
results={}
for group,mask in groups.items():
    # This implementation sums each row directly, rather than using date SSE bins.
    site_points=[];site_boot=[]
    for station in ('sili','zhujia'):
        ids=np.flatnonzero(mask&(st==station));err=p[:,:,ids]-y[ids]
        point=np.array([[math.sqrt(math.fsum(float(v*v) for v in err[s,a])/len(ids)) for a in range(4)] for s in range(3)])
        site_points.append(point)
        row_weights=w[:,di[ids]].astype('f8');sums=row_weights@((err*err).reshape(12,-1)).T
        site_boot.append(np.sqrt(sums/row_weights.sum(1)[:,None]).reshape(2000,3,4))
    point=np.mean(site_points,axis=(0,1));boot=np.mean(site_boot,axis=(0,2))
    max_point=max(abs(point[i]-score['groups'][group]['RMSE'][a]) for i,a in enumerate(arms));assert max_point<1e-8
    max_ci=0
    for cand,ctrl in ((1,0),(3,2),(2,0),(3,1)):
        ci=np.quantile(boot[:,cand]-boot[:,ctrl],[.025,.975]);ref=score['groups'][group]['contrasts'][arms[cand]+' - '+arms[ctrl]]['CI95']
        max_ci=max(max_ci,float(np.max(np.abs(ci-ref))))
    assert max_ci<1e-8
    results[group]=dict(max_RMSE_difference=max_point,max_date_CI_difference=max_ci)
(here/'analysis/secondary_recompute.json').write_text(json.dumps(dict(state='PASS_SECONDARY_ROW_WEIGHTED_FP64_AND_DATE_BOOTSTRAP',groups=results,test_payloads_read=0),indent=2)+'\n')

out=here/'figures';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
labels=['All validation','High transmittance','GHI >= 800','Low kt','Middle kt']
for arm,col,mk,offset in [('A_bright2x - A_uniform','#0072B2','o',-.1),('AC_bright2x - AC_uniform','#D55E00','s',.1)]:
    vals=[score['groups'][g]['contrasts'][arm] for g in ['all','high_transmittance','high_GHI_800','low_kt','middle_kt']]
    d=np.array([v['delta_RMSE'] for v in vals]);ci=np.array([v['CI95'] for v in vals])
    axs[0].errorbar(d,np.arange(5)+offset,xerr=np.stack([d-ci[:,0],ci[:,1]-d]),fmt=mk,color=col,capsize=3,label='AGRI+COT' if mk=='s' else 'AGRI')
axs[0].axvline(0,c='.4',lw=1);axs[0].set(yticks=range(5),yticklabels=labels,xlabel='2x loss minus uniform RMSE (W m$^{-2}$)',title='High-value recovery trades against low/middle values');axs[0].invert_yaxis();axs[0].legend()
for ai,col,mk,label in [(2,'#0072B2','o','AGRI+COT, uniform'),(3,'#D55E00','s','AGRI+COT, 2x loss')]:
    mask=y>=800;axs[1].scatter(y[mask],p[:,ai,mask].mean(0),s=15,alpha=.45,c=col,marker=mk,label=label)
axs[1].plot([0,1400],[0,1400],'k--',lw=.8);axs[1].set(xlim=(0,1400),ylim=(0,1400),xlabel='Observed 15-min GHI (W m$^{-2}$)',ylabel='Three-seed mean predicted GHI (W m$^{-2}$)',title='All 219 validation targets with GHI >= 800');axs[1].legend(fontsize=8)
fig.supxlabel('Left: paired 95% BJT-date bootstrap intervals; 52 validation dates, 3 seeds, 2 stations. Validation-selected pilot; test closed.',fontsize=8)
fig.savefig(out/'loss_tradeoff.png',dpi=160);fig.savefig(out/'loss_tradeoff.pdf');plt.close(fig)

diag=repo/'experiments/cot_underestimate_forensics_20261016'
selection=json.loads((diag/'results/selected_cases.json').read_text(encoding='utf-8'))
with np.load(diag/'payload/real_cloud_cases_and_gradients.npz',allow_pickle=False) as data:
    clouds={k:data[k] for k in data.files if k.endswith('_rgb') or k.endswith('_cot_corrected')}
stretch=json.loads((diag/'figures/figure_transformations.json').read_text());lo=np.array(stretch['RGB_low']);hi=np.array(stretch['RGB_high'])
fail=[c for c in selection['cases'] if c['role']=='severe_under_drop']
fig,axs=plt.subplots(6,4,figsize=(14,15),layout='constrained',gridspec_kw={'width_ratios':[1,1,1,2.2]})
for row,case in enumerate(fail):
    ci=case['case_index'];j=case['validation_index']
    assert st[j]==case['station'] and y[j]==case['ghi']
    for t,offset in enumerate((-30,-15,0)):
        ax=axs[row,t];im=np.moveaxis(clouds[f'c{ci}_t{t}_rgb'],0,-1);ax.imshow(np.clip((im-lo)/(hi-lo),0,1),interpolation='nearest')
        ax.add_patch(Rectangle((6.5,6.5),3,3,fill=False,ec='cyan',lw=1));ax.plot(8,8,'+',c='cyan');ax.set_xticks([]);ax.set_yticks([])
        ax.set_title(f'{case["station"].title()} {case["bjt"][:16]}\n{offset:+d} min, R-COT center {clouds[f"c{ci}_cot_corrected"][t,1,1]:.2f}',fontsize=8)
    ax=axs[row,3];means=p[:,:,j].mean(0);low=p[:,:,j].min(0);high=p[:,:,j].max(0)
    for ai,col,mk in [(0,'#0072B2','o'),(1,'#0072B2','s'),(2,'#D55E00','o'),(3,'#D55E00','s')]:
        ax.errorbar(means[ai],ai,xerr=[[means[ai]-low[ai]],[high[ai]-means[ai]]],fmt=mk,c=col,capsize=3)
        ax.annotate(f'{means[ai]:.1f}',(means[ai],ai),xytext=(5,5),textcoords='offset points',fontsize=8)
    ax.axvline(y[j],c='black',ls='--',label=f'Observed {y[j]:.1f}');ax.set_xlim(0,1300);ax.set_ylim(3.7,-.7);ax.set_yticks(range(4),['AGRI uniform','AGRI 2x','AGRI+COT uniform','AGRI+COT 2x'],fontsize=8);ax.grid(axis='x',alpha=.2);ax.legend(fontsize=8,loc='lower right')
fig.suptitle('Same six frozen failure cases: loss change raises estimates but does not resolve underprediction',fontsize=13)
fig.supxlabel('True AGRI C03/C02/C01 composite with fixed shared stretch; cyan = unchanged 3x3 GHI input. Right: 3-seed mean and min-max (not CI), W m$^{-2}$.\n16x16 images are the existing R context only. Cases selected on prior errors are regression checks, not representative validation skill.',fontsize=9)
fig.savefig(out/'six_real_cloud_loss_comparison.png',dpi=140);fig.savefig(out/'six_real_cloud_loss_comparison.pdf');plt.close(fig)
(out/'figure_provenance.json').write_text(json.dumps(dict(RGB_source=str((diag/'payload/real_cloud_cases_and_gradients.npz').relative_to(repo)),RGB_transform=stretch,case_selection='same six prior failure anchors; no new selection',error_bars='case plots: min-max of 3 seeds; tradeoff: paired BJT date bootstrap 95% intervals',test_payloads_read=0),indent=2)+'\n')
print('PASS_SECONDARY_RECOMPUTE_AND_REAL_CLOUD_FIGURES')
