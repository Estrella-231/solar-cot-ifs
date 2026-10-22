import csv,json,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
rows=list(csv.DictReader((P/'groups.csv').open()));fig,ax=plt.subplots(1,3,figsize=(14,4.5),layout='constrained');sts=['Niushou','Wangzhai','Mizhuang'];colors=['#0072B2','#D55E00','#009E73'];detail=json.loads((P/'SUPPORT_DIAGNOSTIC.json').read_text())
for k,st in enumerate(sts):
    data=np.load(B/'predictions_attempt3'/f'{st}_seed42_G0.npz');m=data['analysis']=='primary_shared_dates';h=np.array([int(x[11:13])+int(x[14:16])/60 for x in data['init_time_BJT'][m]])
    hist,edges=np.histogram(h,bins=np.arange(0,24.5,.5));ax[0].step((edges[1:]+edges[:-1])/2,hist/hist.sum()*100,where='mid',label=st,color=colors[k])
    for j,g in enumerate(['old_init_hours_0945to1530','outside_old_init_hours']):
        r=next(x for x in rows if x['station']==st and x['group']==g);d=float(r['rmse_AC'])-float(r['rmse_A']);ax[1].bar(k+(j-.5)*.32,d,.3,color=['#009E73','#D55E00'][j],label=['Within old init hours','Outside old init hours'][j] if k==0 else None)
    frac=100*detail[st]['all_invalid_fraction'];ax[2].bar(k,frac,color=colors[k]);ax[2].text(k,frac+.6,f'{frac:.1f}%',ha='center')
ax[0].axvspan(9.75,15.5,color='.65',alpha=.3,label='Old train/val init range');ax[0].set(xlim=(0,20),xlabel='Initialization hour (BJT)',ylabel='New-period forecast records (%)',title='Initialization domain expanded');ax[0].legend(fontsize=8)
ax[1].axhline(0,color='black',lw=.8);ax[1].set_xticks(range(3),sts);ax[1].set(ylabel='RMSE change (COT minus no COT; W m$^{-2}$)',title='Most station differences arise outside old hours');ax[1].legend(fontsize=8)
ax[2].set_xticks(range(3),sts);ax[2].set(ylim=(0,27),ylabel='Records with all 8 history COT frames invalid (%)',title='All-zero COT was absent from old train/val');
for a in ax:a.grid(axis='y',alpha=.18)
fig.suptitle('Input timing audit: correct timestamps, different initialization coverage');fig.supxlabel('Post-hoc diagnostics only. Full new-period scores remain unchanged; no observations were removed or models retrained.',fontsize=9)
fig.savefig(P/'figures/input_domain_diagnosis.png',dpi=180);fig.savefig(P/'figures/input_domain_diagnosis.pdf');plt.close(fig)
# exact case-level fixed-head evidence
out=[]
for c in json.loads((P/'cases.json').read_text()):
    for st in sts:
        ph=np.load(P/f'{st}_physical.npz');r=np.load(B/'predictions_attempt3'/f'{st}_seed42_G0.npz');m=r['init_time_BJT']==c['init'];a=[];b=[];z=[]
        for s in [42,43,44]:
            a.append(np.load(B/'predictions_attempt3'/f'{st}_seed{s}_G0.npz')['pred_ghi'][m]);b.append(np.load(B/'predictions_attempt3'/f'{st}_seed{s}_G1.npz')['pred_ghi'][m]);z.append(np.load(P/f'{st}_seed{s}_counterfactual.npz')['g1_zero_cot'][m])
        a,b,z=map(np.array,[a,b,z]);out.append(dict(init=c['init'],station=st,history_valid_fraction=float(ph['history_valid_fraction'][m].mean()),mean_G1_minus_G0=float((b-a).mean()),mean_G1_minus_fixed_zero=float((b-z).mean()),max_abs_G1_minus_fixed_zero=float(abs(b-z).max())))
(P/'CASE_COUNTERFACTUAL.json').write_text(json.dumps(out,indent=2))
