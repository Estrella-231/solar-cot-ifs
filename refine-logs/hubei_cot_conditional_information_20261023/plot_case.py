import csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
with np.load(P/'CASE_MATERIAL.npz') as z:case={k:z[k] for k in z.files}
init=str(case['init']);sites=['Niushou','Wangzhai','Mizhuang'];centers=[(8,6),(9,8),(7,10)]
def rgb(x):return np.sqrt(np.clip(np.moveaxis(x[[2,1,0]],0,-1)/.7,0,1))
with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42}):
    fig=plt.figure(figsize=(17,13));gs=fig.add_gridspec(4,15,height_ratios=[1.15,1.15,.9,1.2],left=.06,right=.985,top=.91,bottom=.14,wspace=1.4,hspace=.5)
    for j,k in enumerate([0,4,8,12,16]):
        for row in [0,1]:
            ax=fig.add_subplot(gs[row,3*j:3*j+3]);image=case['observed_AGRI'][k] if row==0 or k==0 else case['forecast_AGRI'][k-1]
            ax.imshow(rgb(image),interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
            for (cy,cx),label in zip(centers,['N','W','M']):ax.scatter(cx,cy,marker='*',s=65,facecolor='white',edgecolor='black',linewidth=.7);ax.text(cx+.4,cy-.3,label,color='black',fontsize=8,bbox=dict(facecolor='white',alpha=.7,edgecolor='none',pad=.2))
            if row==0:ax.set_title(('Init' if k==0 else f'+{k*15} min')+'\n'+str(case['times'][k])[8:10]+':'+str(case['times'][k])[10:12])
            if j==0:ax.set_ylabel('Observed AGRI' if row==0 else 'SimVP AGRI\n(init observed reused)')
    table=[];legend=None;cot_axes=[];ghi_axes=[]
    for si,(st,(cy,cx)) in enumerate(zip(sites,centers)):
        ax=fig.add_subplot(gs[2,5*si:5*si+5]);actual=case['actual_R_COT'][1:,0,cy,cx].astype('float64');actual[~case['actual_valid'][1:,0,cy,cx]]=np.nan;pred=case['forecast_R_COT'][:,0,cy,cx]
        lead=np.arange(15,241,15);ax.plot(lead,actual,'--o',color='#00856A',ms=3,label='Actual AGRI to R (offline)');ax.plot(lead,pred,'-s',color='#D55E00',ms=3,label='SimVP AGRI to R');ax.set_title(st);ax.set(ylabel='R-COT proxy',ylim=(0,None));ax.grid(alpha=.2);ax.legend(fontsize=8);cot_axes.append(ax)
        values=[]
        for seed in [42,43,44]:
            with np.load(P/'cross_source_attempt1'/st/f'seed{seed}_OOF.npz') as z:d={k:z[k] for k in z.files}
            pp=np.full((3,16),np.nan);truth=np.full(16,np.nan)
            for li,l in enumerate(lead):
                idx=np.flatnonzero((d['init_time']==init)&(d['lead']==l));assert len(idx)<=1
                if len(idx):pp[:,li]=d['prediction'][[0,4,3],idx[0]];truth[li]=d['truth'][idx[0]]
            values.append(pp)
        values=np.array(values);count=np.isfinite(values).sum(0);mean=np.divide(np.nansum(values,0),count,out=np.full((3,16),np.nan),where=count>0);ax=fig.add_subplot(gs[3,5*si:5*si+5]);ax.plot(lead,truth,'-o',color='black',ms=3,label='Observed 15-min GHI');ax.plot(lead,mean[0],'-',color='#0072B2',label='Existing G1');ax.plot(lead,mean[1],'-s',color='#D55E00',ms=3,label='Reference-trained head, forecast COT');ax.plot(lead,mean[2],'--',color='#00856A',label='Same head, actual R-COT (offline)');ax.set(ylabel='GHI (W m$^{-2}$)',xlabel='Lead from history end (min)',ylim=(0,None));ax.grid(alpha=.2);legend=ax.get_legend_handles_labels()
        ghi_axes.append(ax)
        for i,l in enumerate(lead):table.append(dict(station=st,init=init,lead=int(l),observed_R_COT=float(actual[i]),forecast_R_COT=float(pred[i]),observed_GHI=float(truth[i]),G1=float(mean[0,i]),reference_trained_forecast_COT_GHI=float(mean[1,i]),reference_COT_GHI_OFFLINE=float(mean[2,i])))
    cotmax=np.nanmax([[r['observed_R_COT'],r['forecast_R_COT']] for r in table]);ghimax=np.nanmax([[r[k] for k in ['observed_GHI','G1','reference_trained_forecast_COT_GHI','reference_COT_GHI_OFFLINE']] for r in table])
    for ax in cot_axes:ax.set_ylim(0,20*np.ceil(cotmax/20));ax.set_xlim(10,245)
    for ax in ghi_axes:ax.set_ylim(0,100*np.ceil(ghimax/100));ax.set_xlim(10,245)
    fig.suptitle(f'Real cloud event and the COT source penalty | init {init[:4]}-{init[4:6]}-{init[6:8]} {init[8:10]}:{init[10:12]} BJT',fontsize=15)
    fig.legend(*legend,loc='lower center',bbox_to_anchor=(.5,.065),ncol=4,fontsize=9)
    fig.text(.5,.035,'Train-date OOF diagnostic. Actual future imagery / R-COT are offline; R-COT is not independent truth. GHI gaps are unscored labels; curves show three-base means.',ha='center',fontsize=9)
    out=P/'figures';out.mkdir(exist_ok=True);fig.savefig(out/'calendar_cloud_COT_GHI.png',dpi=180,facecolor='white');fig.savefig(out/'calendar_cloud_COT_GHI.pdf',facecolor='white');plt.close(fig)
with (P/'case_values.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
(P/'FIGURE_PROVENANCE.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256((P/'CASE_MATERIAL.npz').read_bytes()).hexdigest(),selection=json.loads((P/'CASE_SCOPE.json').read_text()),display='Fixed C03/C02/C01 square-root 0..0.7 false colour; native 16x16 pixels; no enhancement/smoothing',GHI='Right-ending 15min means; three-base means only for plotting, not ensemble scoring',limits='Raw R-COT can exceed100; it is not independent truth. Actual future image/COT paths are offline. One calendar case does not estimate overall gain.',destination='Provisional research diagnostic, no journal-specific export claim'),indent=2))
