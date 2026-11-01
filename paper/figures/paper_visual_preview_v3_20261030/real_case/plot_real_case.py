"""Native-image, fixed-event figure. No fitting, inference or support changes."""
from pathlib import Path
import importlib.util
import csv
import hashlib
import json
import platform
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
OLD=ROOT/'paper/figures/redesign_20261030/real_case/plot_real_case.py'
spec=importlib.util.spec_from_file_location('source_case_reader',OLD)
source=importlib.util.module_from_spec(spec); spec.loader.exec_module(source)
case,scope,data,records,paths=source.load_case()
COLORS={'observed':'#80549A','forecast':'#C06100','baseline':'#586976','truth':'#181818'}
MM=1/25.4
STYLE=dict(source.STYLE)
STYLE.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':8,
              'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
              'lines.linewidth':1.2})
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def save(fig,stem):
    for ext in ['pdf','svg','png']: fig.savefig(OUT/f'{stem}.{ext}',dpi=300)
    fig.savefig(OUT/f'{stem}_print_preview.png',dpi=150)
    plt.close(fig)

def time_axis(ax):
    ax.set_xlim(12.15,16.08)
    ax.set_xticks([12.25,13,14,15,16],['12:15','13:00','14:00','15:00','16:00'])
    ax.grid(axis='y',color='#DDE1E5',linewidth=.5,zorder=0)
    ax.tick_params(length=2.5,width=.6)
    for h in [12.25,14,16]: ax.axvline(h,color='#ADB5BB',lw=.65,ls=':',zorder=0)

def trajectories(ax,station,kind):
    d=data[station]; hours=12+np.arange(15,241,15)/60
    if kind=='cot':
        ax.plot(hours,d['actual'],'--o',color=COLORS['observed'],ms=2.5,label='Observed-image COT (offline)')
        ax.plot(hours,d['forecast'],'-s',color=COLORS['forecast'],ms=2.5,label='Forecast-image COT')
        ax.set(ylim=(0,120),yticks=[0,50,100],ylabel='Station-pixel COT')
    else:
        ax.plot(hours,d['truth'],'-o',color=COLORS['truth'],ms=2.5,label='Observed 15-min GHI',zorder=5)
        ax.plot(hours,d['prediction'][0],'-D',color=COLORS['baseline'],ms=1.8,label='AGRI + historical COT')
        ax.plot(hours,d['prediction'][1],'-s',color=COLORS['forecast'],ms=2.3,label='Fixed mapping: forecast COT')
        ax.plot(hours,d['prediction'][2],'--o',color=COLORS['observed'],ms=2.3,label='Fixed mapping: observed COT (offline)')
        ax.set(ylim=(0,800),yticks=[0,400,800],ylabel='GHI (W m$^{-2}$)')
    time_axis(ax)

def main_figure():
    fig=plt.figure(figsize=(177*MM,175*MM))
    fig.text(.035,.981,'a  Cloud evolution at matched target times',fontweight='bold',fontsize=9.2,va='top')
    image_gs=fig.add_gridspec(4,3,left=.25,right=.895,bottom=.345,top=.932,wspace=.08,hspace=.075)
    cm=plt.get_cmap('magma').copy();cm.set_bad('#B9BEC3')
    row_labels=['Observed AGRI\n(offline)','Observed-image\nretrieved COT','SimVP-predicted\nAGRI','Forecast-image\nretrieved COT']
    for ri in range(4):
        for col,k in enumerate([1,8,16]):
            ax=fig.add_subplot(image_gs[ri,col])
            if ri==0: array=source.rgb(case['observed_AGRI'][k])
            elif ri==1: array=np.where(case['actual_valid'][k,0],case['actual_R_COT'][k,0],np.nan)
            elif ri==2: array=source.rgb(case['forecast_AGRI'][k-1])
            else: array=case['forecast_R_COT'][k-1,0]
            kw={} if ri in [0,2] else dict(cmap=cm,vmin=0,vmax=140)
            im=ax.imshow(array,origin='upper',interpolation='nearest',**kw)
            for color,lw in [('#202020',1.5),('white',.7)]:
                ax.add_patch(Rectangle((4.5,6.5),3,3,fill=False,ec=color,lw=lw,ls='--'))
            ax.plot(6,8,'+',color='white',ms=4,mew=.85)
            ax.set_xticks([]);ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(True);sp.set_color('#6A7176');sp.set_linewidth(.4)
            if ri==0: ax.set_title(['+15 min  |  12:15','+120 min  |  14:00','+240 min  |  16:00'][col],fontsize=8,pad=5)
            if col==0:
                ax.text(-.20,.5,row_labels[ri],transform=ax.transAxes,ha='right',va='center',fontsize=8.2,
                        color=COLORS['observed'] if ri<2 else COLORS['forecast'])
    cb=fig.colorbar(im,cax=fig.add_axes([.92,.365,.012,.39]),ticks=[0,40,80,120,140])
    cb.ax.tick_params(labelsize=8,length=2,width=.5);cb.outline.set_linewidth(.5)
    cb.ax.set_title('COT',fontsize=8,pad=7)
    fig.text(.25,.326,'+ Niushou station pixel    Dashed box: 3 × 3 GHI input',fontsize=8)
    cot=fig.add_axes([.14,.222,.79,.080]);ghi=fig.add_axes([.14,.112,.79,.080],sharex=cot)
    trajectories(cot,'Niushou','cot');trajectories(ghi,'Niushou','ghi')
    cot.tick_params(labelbottom=False)
    cot.set_ylabel('COT',labelpad=5)
    ghi.set_ylabel('GHI\n(W m$^{-2}$)',labelpad=5)
    ghi.set_xlabel('Target time on 6 August 2025 (BJT)',labelpad=2)
    for ax,letter in [(cot,'b'),(ghi,'c')]:
        ax.text(-.13,1.02,letter,transform=ax.transAxes,fontweight='bold',fontsize=9.2)
    handles,labels=ghi.get_legend_handles_labels()
    labels=['Observed 15-min GHI','AGRI + historical COT','Forecast-image COT mapping','Observed-image COT mapping (offline)']
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.52,.010),ncol=2,
               frameon=False,columnspacing=1.5,handlelength=2.6,labelspacing=.45,fontsize=8)
    save(fig,'fig_cloud_source_mechanism')

def supplement():
    fig,axes=plt.subplots(2,3,figsize=(177*MM,116*MM),sharex=True,sharey='row',layout='constrained')
    fig.set_constrained_layout_pads(w_pad=2*MM,h_pad=2*MM,wspace=.04,hspace=.03)
    for c,s in enumerate(source.SITES):
        for r,k in enumerate(['cot','ghi']):
            ax=axes[r,c];trajectories(ax,s,k)
            ax.text(.015,.96,chr(97+3*r+c),transform=ax.transAxes,va='top',fontweight='bold',fontsize=9)
            if c: ax.set_ylabel('')
            if r==0: ax.set_title(s,fontsize=9);ax.set_xlabel('')
            else: ax.set_xlabel('Target time (BJT)')
            ax.set_xticks([13,14,15,16],['13:00','14:00','15:00','16:00'])
    handles,labels=axes[1,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False,handlelength=2.6,columnspacing=1.1)
    save(fig,'fig_three_station_source_response')

with plt.rc_context(STYLE): main_figure();supplement()
with (OUT/'plotted_timeseries.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
shutil.copyfile(source.SRC/'CASE_MATERIAL.npz',OUT/'image_source_arrays.npz')
inputs=[OLD,source.SRC/'CASE_MATERIAL.npz',source.SRC/'CASE_SCOPE.json',source.SRC/'RESULT_REPORT.md']+paths
manifest={
    'selection':scope,'physical_sizes_mm':{'main':[177,175],'supplement':[177,116]},
    'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in inputs},
    'snapshot_indices':[1,8,16],'station':'Niushou','station_row_col_zero_based':[8,6],
    'RGB':'C03/C02/C01; sqrt(clip(reflectance,0,1)); shared transform, no per-image adjustment',
    'RGB_outside_0_1_fraction':{k:float(np.mean((case[k][:,:3]<0)|(case[k][:,:3]>1))) for k in ['observed_AGRI','forecast_AGRI']},
    'COT':'R proxy, not independent CPP truth; shared linear 0..140; finite outputs above training range retained',
    'image_rendering':'Native16x16, nearest, no smoothing, no geographic scale inferred',
    'GHI':'Right-ending15-min means; three baseline-source curve means for display, not ensemble scores; gaps retained',
    'scope':'Development-date OOF explanatory source substitution. Future observations only offline.',
    'test_read':False,'training_performed':False,
    'software':{'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__},
    'script_sha256':sha(Path(__file__)),
    'outputs':{p.name:sha(p) for p in OUT.glob('fig_*.*')},
}
(OUT/'PROVENANCE.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'files':list(manifest['outputs']),'RGB_outside':manifest['RGB_outside_0_1_fraction']},indent=2))
