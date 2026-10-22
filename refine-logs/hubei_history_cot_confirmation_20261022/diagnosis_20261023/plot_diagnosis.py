import json,csv
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from pathlib import Path
from datetime import datetime
P=Path(__file__).resolve().parent;B=P.parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':180})
C={'Niushou':(8,6),'Wangzhai':(9,8),'Mizhuang':(7,10)};labels={'Niushou':'Niushou','Wangzhai':'Wangzhai','Mizhuang':'Mizhuang'}
meta=json.loads((P/'DECOMPOSITION.json').read_text());cases=json.loads((P/'ALIGNMENT_AUDIT.json').read_text())['cases'];figdir=P/'figures';figdir.mkdir(exist_ok=True)
def export(fig,name):
    fig.savefig(figdir/(name+'.png'),dpi=180);fig.savefig(figdir/(name+'.pdf'));plt.close(fig)
def rgb(arr):
    # fixed all-case false colour: NIR=C03, red=C02, blue=C01, NOT a true RGB image
    x=np.moveaxis(arr[[2,1,0]],0,-1);return np.sqrt(np.clip(x/0.7,0,1))
def points(ax):
    for st,(y,x) in C.items():
        ax.plot(x,y,'*',ms=8,mec='black',mew=.7,mfc='white');ax.text(x+.5,y+.2,st[0] if st!='Mizhuang' else 'M',color='white',fontsize=8,bbox=dict(fc='black',alpha=.4,pad=.2,ec='none'))
def seq(st,init):
    aa=[];bb=[];zz=[]
    for seed in [42,43,44]:
        with np.load(B/'predictions_attempt3'/f'{st}_seed{seed}_G0.npz') as z:r={k:z[k] for k in z.files}
        aa.append(r['pred_ghi']);bb.append(np.load(B/'predictions_attempt3'/f'{st}_seed{seed}_G1.npz')['pred_ghi']);zz.append(np.load(P/f'{st}_seed{seed}_counterfactual.npz')['g1_zero_cot'])
    mask=r['init_time_BJT']==init;idx=r['lead_minutes'][mask]//15-1
    out=[np.full(16,np.nan) for _ in range(4)]
    for dest,vals in zip(out,[r['ghi'],np.array(aa).mean(0),np.array(bb).mean(0),np.array(zz).mean(0)]):dest[idx]=vals[mask]
    return out,mask,r,np.array(aa),np.array(bb)
case_metrics=[]
for case in cases:
    if case['available_future_frames']==0:continue
    f=case['file'];z=np.load(P/f);obs=z['observed_agri'];cot=z['observed_R_cot'].copy();valid=z['cot_valid'];cot[~valid]=np.nan;offs=z['offset_minutes'];pred=z['forecast_agri'];init=case['init']
    fig=plt.figure(figsize=(15,12.7),layout='constrained');outer=fig.add_gridspec(5,1,height_ratios=[1,1,1,1.1,1.45]);fig.suptitle(f'Observed clouds, history COT and GHI forecasts | init {init} BJT',fontsize=15)
    sels=[7,11,15,19,23];names=['init','+60 min','+120 min','+180 min','+240 min'];clim=Normalize(0,30)
    for row in range(3):
        gs=outer[row].subgridspec(1,5)
        for col,(j,nam) in enumerate(zip(sels,names)):
            ax=fig.add_subplot(gs[0,col]);ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.imshow(rgb(obs[j]),interpolation='nearest');ax.set_title(nam+'\n'+(datetime.fromisoformat(init)+__import__('datetime').timedelta(minutes=int(offs[j]))).strftime('%H:%M'),fontsize=10)
            elif row==1:ax.imshow(rgb(obs[7] if col==0 else pred[j-8]),interpolation='nearest')
            else:im=ax.imshow(cot[j],cmap=plt.colormaps['magma'].with_extremes(bad='.75'),norm=clim,interpolation='nearest')
            points(ax)
            if col==0:ax.set_ylabel(['Observed AGRI\nC03/C02/C01','SimVP AGRI\ninit image reused','Observed AGRI -> R\nCOT (snapshot)'][row])
        if row==2:fig.colorbar(im,ax=fig.axes[-5:],label='R-retrieved COT',shrink=.85,extend='max',pad=.015)
    gs=outer[3].subgridspec(1,3)
    for k,(st,(y,x)) in enumerate(C.items()):
        ax=fig.add_subplot(gs[0,k]);ax.plot(offs,cot[:,y,x],'-o',ms=3,color='#009E73',label='Observed AGRI -> frozen R')
        ax.axvline(0,color='black',lw=.9);ax.axvspan(-105,0,color='.7',alpha=.2);ax.set(xlim=(-105,240),ylim=(0,max(3,float(np.nanmax(cot[:,y,x]))*1.2)),title=st,xlabel='Minutes from initialization',ylabel='Station COT');ax.grid(alpha=.18);ax.set_xticks([-105,0,60,120,180,240]);ax.legend(fontsize=7,loc='upper right')
    gs=outer[4].subgridspec(1,3)
    for k,(st,center) in enumerate(C.items()):
        ax=fig.add_subplot(gs[0,k]);v,m,r,a,b=seq(st,init);xx=np.arange(15,241,15)
        for ar,co,sty,lab in zip(v,['black','#0072B2','#D55E00','.55'],['o-','s-','^-','--'],['Observed 15-min mean','No COT head','History-COT head','Same COT head, zero input']):ax.plot(xx,ar,sty,color=co,ms=3,lw=1.7 if co!='.55' else 1,label=lab)
        ax.set(xlim=(10,245),ylim=(0,1100),xlabel='Forecast lead (min)',ylabel='GHI (W m$^{-2}$)');ax.grid(alpha=.18);ax.legend(fontsize=7,loc='upper left')
        delta=float(np.mean((b[:,m]-r['ghi'][m])**2-(a[:,m]-r['ghi'][m])**2));ax.set_title(f'{st}: mean squared-error change {delta:+.0f}',fontsize=10)
        case_metrics.append(dict(init=init,station=st,n=int(m.sum()),delta_mse=delta,observed_max=float(np.nanmax(v[0])),no_cot_max=float(np.nanmax(v[1])),history_cot_max=float(np.nanmax(v[2]))))
    fig.supxlabel('Diagnostic cases chosen for station contrast; GHI curves show seed means (not an ensemble score). Grey COT = invalid.\nAGRI is a fixed-stretch false-colour composite; COT snapshots are not interval GHI truth. Future observed images/COT are offline only.',fontsize=9)
    export(fig,Path(f).stem)
# Aggregate diagnosis: seed dependence, mask-conditioned changes, bias
fig,axs=plt.subplots(1,3,figsize=(15,4.5),layout='constrained');sts=list(C)
for j,st in enumerate(sts):
    d=meta[st];ds=np.array([r['rmse_AC']-r['rmse_A'] for r in d['by_seed']]);axs[0].plot(np.full(3,j)+np.array([-.12,0,.12]),ds,'o',ms=7,label=st)
    for xx,yy,seed in zip(j+np.array([-.12,0,.12]),ds,[42,43,44]):axs[0].annotate(str(seed),(xx,yy),xytext=(0,6),textcoords='offset points',ha='center',fontsize=8)
    vals=[d['bias_A'],d['bias_AC']];axs[2].bar(j-.16,vals[0],width=.3,color='#0072B2');axs[2].bar(j+.16,vals[1],width=.3,color='#D55E00')
rows=list(csv.DictReader((P/'groups.csv').open()));colors=['.5','#E69F00','#009E73']
for k,name in enumerate(['history_all_invalid','history_partial','history_all_valid']):
    vals=[float(next(r for r in rows if r['station']==st and r['group']==name)['delta_mse']) for st in sts];axs[1].bar(np.arange(3)+(k-1)*.24,vals,.22,color=colors[k],label=['All history COT invalid','Partly valid','All valid'][k])
for ax in axs:ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(3),sts);ax.grid(axis='y',alpha=.2)
axs[0].set(title='Seed dependence',ylabel='RMSE change (COT minus no COT; W m$^{-2}$)');axs[1].set(title='Conditional error contribution',ylabel='Mean squared-error change (W$^2$ m$^{-4}$)');axs[1].legend(fontsize=8)
axs[2].set(title='Mean GHI bias',ylabel='Prediction minus observation (W m$^{-2}$)');axs[2].legend(handles=[Line2D([0],[0],color='#0072B2',lw=8,label='No COT'),Line2D([0],[0],color='#D55E00',lw=8,label='History COT')],fontsize=8)
fig.suptitle('New-period diagnosis on all primary paired samples | positive error change = worse');export(fig,'all_sample_diagnosis')
(P/'CASE_METRICS.json').write_text(json.dumps(case_metrics,indent=2))
print('PLOTS_COMPLETE',len(case_metrics)//3)
