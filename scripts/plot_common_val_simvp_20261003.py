"""Render native-grid paired AGRI and COT from real validation inference."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    report=json.loads((a.results/'complete.json').read_text())
    assert report['state']=='COMPLETE_COMMON_VALIDATION_DIAGNOSTIC' and not report['test_payloads_read']
    leads=np.arange(1,17)*15
    colors={'observed':'#0072B2','old':'#D55E00','new':'#009E73','cpp':'#333333'}
    labels={'observed':'R(real AGRI), proxy','old':'Old SimVP -> R','new':'New SimVP -> R','cpp':'CPP reference'}
    generated=[]
    for case in report['cases']:
        rec=case['record'];st=rec['station'];sr,sc=(163,148) if st=='sili' else (73,150)
        stamp=rec['init_bjt'];name=Path(case['file']).stem
        with np.load(a.results/case['file']) as z:
            # All frame intensities use the identical fixed physical reflectance mapping.
            fig,axs=plt.subplots(3,8,figsize=(20,8),layout='constrained')
            ids=(0,2,4,6,8,10,12,15)
            for row,key in enumerate(('observed','old','new')):
                for col,t in enumerate(ids):
                    ax=axs[row,col];ax.imshow(z[key+'_rgb'][t],interpolation='nearest')
                    ax.plot(sc,sr,'*',ms=8,mfc='white',mec='black',mew=.6)
                    ax.add_patch(Rectangle((sc-32-.5,sr-32-.5),64,64,fill=False,ec='yellow',lw=.8))
                    ax.set_xticks([]);ax.set_yticks([])
                    if row==0:ax.set_title('+'+str(leads[t])+' min',fontsize=10)
                    if col==0:ax.set_ylabel({'observed':'Real AGRI','old':'Old SimVP','new':'New SimVP'}[key],fontsize=12)
            fig.suptitle(f'{st.upper()} | init {stamp} BJT | common validation\nC03/C02/C01 false-colour proxy: fixed reflectance 0..0.8, gamma 0.6; star=station; box=64x64',fontsize=13)
            dst=a.output/(name+'_agri.png');fig.savefig(dst,dpi=140,facecolor='white');plt.close(fig);generated.append(dst.name)
            fig=plt.figure(figsize=(26,12),layout='constrained')
            gs=fig.add_gridspec(6,16,height_ratios=[1,1,1,1,1.3,1.3])
            axes=[];cmap=plt.get_cmap('magma').copy();cmap.set_bad('#bdbdbd')
            norm=Normalize(0,50)
            for row,key in enumerate(('observed','old','new','cpp')):
                values=z[key+'_cot'].copy()
                if key!='cpp':values[~z['valid']]=np.nan
                for t in range(16):
                    ax=fig.add_subplot(gs[row,t]);axes.append(ax)
                    im=ax.imshow(values[t],cmap=cmap,norm=norm,interpolation='nearest')
                    ax.plot(32,32,'*',ms=6,mfc='white',mec='black',mew=.5)
                    if np.isfinite(values[t]).any():
                        rr,cc=np.unravel_index(np.nanargmax(values[t]),values[t].shape)
                        ax.plot(cc,rr,'o',ms=4,mfc='none',mec='#56B4E9',mew=.8)
                    ax.set_xticks([]);ax.set_yticks([])
                    if row==0:ax.set_title('+'+str(leads[t]),fontsize=9)
                    if t==0:ax.set_ylabel(labels[key],fontsize=10)
            fig.colorbar(im,ax=axes,shrink=.8,pad=.005,extend='max',label='Physical COT (dimensionless); grey=invalid/unavailable')
            curve=fig.add_subplot(gs[4:,0:6]);peak=fig.add_subplot(gs[4:,6:11]);pos=fig.add_subplot(gs[4:,11:16])
            for key in ('observed','old','new','cpp'):
                v=rec['series'][key]
                curve.plot(leads,v['station_cot'],color=colors[key],marker='o' if key=='observed' else None,
                           ls='--' if key=='old' else '-',ms=3,label=labels[key])
                if key!='cpp':
                    peak.plot(leads,v['peak5_cot'],color=colors[key],ls='--' if key=='old' else '-',label=labels[key])
                    xy=np.array([[np.nan if q is None else q for q in pair] for pair in v['peak64_offset_rc']],float)
                    pos.plot(xy[:,1],xy[:,0],color=colors[key],marker='o',ms=3,ls='--' if key=='old' else '-',label=labels[key])
                    for i in (0,3,7,11,15):
                        if np.isfinite(xy[i]).all():pos.annotate(str(i+1), (xy[i,1],xy[i,0]),fontsize=7,color=colors[key])
            curve.axhline(10,color='#999999',lw=.7,ls=':');curve.set_title('Station COT; dotted line=10')
            peak.set_title('5x5 neighbourhood peak COT');pos.set_title('64x64 maximum location sequence\nlabels=lead index; not cloud-object tracking')
            for ax in (curve,peak):ax.set_xlabel('Lead from history end (min)');ax.set_ylabel('Physical COT');ax.grid(alpha=.2);ax.legend(fontsize=8)
            pos.plot(0,0,'k*',ms=10);pos.set_xlabel('Column offset (pixels)');pos.set_ylabel('Row offset (pixels)');pos.invert_yaxis();pos.grid(alpha=.2)
            fig.suptitle(f'{st.upper()} | init {stamp} BJT | same frozen R, same 64x64 grid, same masks\nReal-AGRI inversion is a proxy; CPP is a retrieval reference. Selected by observed peak, not model gain.',fontsize=14)
            dst=a.output/(name+'_cot.png');fig.savefig(dst,dpi=140,facecolor='white');plt.close(fig);generated.append(dst.name)
    manifest={'source_complete_json':str((a.results/'complete.json').resolve()),'figures':generated,'common_windows':report['common_windows'],
              'evaluated_windows':report['evaluated_windows'],'test_payloads_read':False,'style':'native grid, fixed reflectance mapping and COT scale; no image smoothing'}
    (a.output/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Figures:',len(generated))

if __name__=='__main__':main()
