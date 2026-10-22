import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;out=P/'figures';out.mkdir(exist_ok=True)
r=json.loads((P/'ANALYSIS.json').read_text())
with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42}):
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained',sharey=True)
    for ax,sc,title in zip(axs,['C_fit','C_select'],['50 fitting dates (49 at Wangzhai)','12 reused selection dates']):
        for key,label,c,marker in [('persistence_rmse','Persistence','#777777','s'),('forecast_rmse','SimVP to matched R','#0072B2','o')]:
            ax.plot(range(1,5),[r['summary'][sc][f'{h}h'][key] for h in range(1,5)],color=c,marker=marker,label=label)
        ax.set(xticks=range(1,5),xlabel='Forecast horizon bin (h)',ylim=(0,16),title=title);ax.grid(alpha=.2);ax.legend(frameon=False)
    axs[0].set_ylabel('COT RMSE vs same-fold observed-R reference')
    fig.savefig(out/'matched_R_horizons.png',dpi=180);fig.savefig(out/'matched_R_horizons.pdf');plt.close(fig)
    z=np.load(P/'CALENDAR_CASE.npz');fig=plt.figure(figsize=(14,14));gs=fig.add_gridspec(6,12,height_ratios=[1,1,1,1,1.3,1.3],left=.08,right=.90,bottom=.05,top=.94,hspace=.65,wspace=1.4);lead_indices=[3,7,11,15]
    centers=[(8,6),(9,8),(7,10)];names=['Niushou','Wangzhai','Mizhuang']
    for row in range(4):
        for col,idx in enumerate(lead_indices):
            ax=fig.add_subplot(gs[row,col*3:(col+1)*3]);ax.set_xticks([]);ax.set_yticks([])
            if row<2:
                x=z['observed'][idx+1,:3] if row==0 else z['forecast'][idx,:3]
                rgb=np.sqrt(np.clip(np.moveaxis(x[[2,1,0]],0,-1)/.7,0,1));ax.imshow(rgb,interpolation='nearest')
            else:
                v=z['observed_R'][idx+1,0].copy() if row==2 else z['forecast_cot'][idx,0].copy()
                if row==2:v=np.where(z['observed_valid'][idx+1,0],v,np.nan)
                im=ax.imshow(v,vmin=0,vmax=60,cmap='magma',interpolation='nearest')
            for (cy,cx),letter in zip(centers,['N','W','M']):ax.scatter(cx,cy,marker='*',s=45,c='white',edgecolors='black',linewidths=.5);ax.text(cx+.4,cy,letter,color='white',fontsize=7)
            if row==0:ax.set_title(f'+{(idx+1)*15} min')
            if col==0:ax.set_ylabel(['Actual AGRI\nC03/C02/C01','Forecast AGRI\nC03/C02/C01','Actual AGRI to R\n(offline reference)','Forecast AGRI to R'][row])
    for si,(name,(cy,cx)) in enumerate(zip(names,centers)):
        ax=fig.add_subplot(gs[4,si*4:(si+1)*4]);x=np.arange(15,241,15)
        actual=z['observed_R'][1:,0,cy,cx];actual=np.where(z['observed_valid'][1:,0,cy,cx],actual,np.nan)
        ax.plot(x,actual,color='#009E73',marker='o',ms=3,label='Actual AGRI to R (offline)');ax.plot(x,z['forecast_cot'][:,0,cy,cx],color='#0072B2',marker='s',ms=3,label='Forecast AGRI to R')
        ax.axhline(z['observed_R'][0,0,cy,cx],color='#777777',ls='--',label='Init persistence');ax.set_ylim(0,max(1,1.1*max(np.nanmax(actual),np.max(z['forecast_cot'][:,0,cy,cx]))));ax.set(title=name,ylabel='Station-centre COT',xlabel='Lead (min)',xticks=[60,120,180,240]);ax.grid(alpha=.15)
        if si==0:ax.legend(frameon=False,fontsize=7)
        ax=fig.add_subplot(gs[5,si*4:(si+1)*4]);ax.plot(x,z['ghi_truth'][si],color='black',marker='o',ms=3,label='Observed 15-min GHI')
        for k,label,c,ls in [(0,'Frozen history-COT base','#777777','--'),(1,'No future COT','#0072B2','-'),(2,'With future COT','#D55E00','-')]:ax.plot(x,z['ghi_prediction'][si,k],color=c,ls=ls,label=label)
        ax.set(ylabel='GHI (W m$^{-2}$)',xlabel='Lead (min)',xticks=[60,120,180,240]);ax.grid(alpha=.15)
        if si==0:ax.legend(frameon=False,fontsize=7)
    fig.suptitle('Calendar-selected case | init '+str(z['init'])+' BJT | '+str(z['source_role'])+' R',fontsize=13)
    fig.colorbar(im,cax=fig.add_axes([.92,.43,.015,.48]),label='COT (values above 60 saturated)',extend='max')
    fig.savefig(out/'calendar_cloud_COT_GHI.png',dpi=160);fig.savefig(out/'calendar_cloud_COT_GHI.pdf');plt.close(fig)
print('FIGURES_WRITTEN')
