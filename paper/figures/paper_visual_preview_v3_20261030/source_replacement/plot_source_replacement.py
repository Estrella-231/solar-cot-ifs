"""Compact source-replacement diagnosis, exact audited summary values retained."""
from pathlib import Path
import json, shutil, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
SRC=ROOT/'paper/figures/revision_v1_20261030/source_replacement'
S=json.loads((SRC/'summary.json').read_text())
STYLE={'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
       'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'xtick.labelsize':8,
       'ytick.labelsize':8,'axes.linewidth':.65,'axes.spines.top':False,
       'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none',
       'mathtext.fontset':'stix','savefig.facecolor':'white','figure.facecolor':'white'}

with plt.rc_context(STYLE):
    fig=plt.figure(figsize=(177/25.4,108/25.4))
    ax=fig.add_axes([.16,.565,.78,.28])
    keys=['forecast_no_COT','forecast_forecast_COT','forecast_observed_COT_OFFLINE',
          'observed_no_COT_OFFLINE','observed_forecast_COT_OFFLINE','observed_observed_COT_OFFLINE']
    grid=np.array([S['rmse'][k] for k in keys]).reshape(2,3)
    ax.imshow(grid,cmap='cividis_r',norm=Normalize(80,110),aspect='auto')
    ax.set_xticks([0,1,2],['No future COT','Forecast-image COT','Observed-image COT*'])
    ax.xaxis.tick_top();ax.tick_params(length=0,pad=7)
    ax.set_yticks([0,1],['Forecast AGRI','Observed AGRI*'])
    for i in range(2):
        for j in range(3):
            ax.text(j,i,f'{grid[i,j]:.2f}',ha='center',va='center',fontsize=13,
                color='white' if grid[i,j]>97 else '#17252A',fontweight='bold')
    ax.set_xticks(np.arange(-.5,3),minor=True);ax.set_yticks(np.arange(-.5,2),minor=True)
    ax.grid(which='minor',color='white',lw=2);ax.tick_params(which='minor',length=0)
    ax.spines[:].set_visible(False)
    fig.text(.035,.962,'a  GHI error under fixed-mapping source substitution',fontsize=9.2,fontweight='bold')
    fig.text(.94,.51,'RMSE (W m$^{-2}$)',ha='right',fontsize=8)
    fig.text(.16,.51,'8,111 matched rows  |  12 target dates  |  3 stations',fontsize=8)
    # Intentionally distinct scales: a large image-source effect and small COT effects.
    a=fig.add_axes([.10,.17,.35,.205])
    c=S['comparisons']['replace_AGRI_no_future_COT'];x=c['delta'];lo,hi=c['ci95']
    a.axvline(0,color='#737D86',lw=.75)
    a.errorbar(x,0,xerr=[[x-lo],[hi-x]],fmt='D',ms=5.5,color='#80549A',capsize=3,lw=1.3)
    a.set(xlim=(-25,2),ylim=(-.75,.75),yticks=[],xticks=[-20,-10,0])
    a.spines['left'].set_visible(False);a.grid(axis='x',color='#E5E8EC',lw=.6)
    a.text(.5,.91,f'{x:+.2f}  [{lo:+.2f}, {hi:+.2f}]',ha='center',transform=a.transAxes,fontsize=8)
    a.set_xlabel('$\Delta$RMSE (W m$^{-2}$)',labelpad=4)
    fig.text(.035,.435,'b  Replace forecast AGRI with observed AGRI*',fontweight='bold',fontsize=8.6)
    b=fig.add_axes([.65,.17,.29,.205])
    b.axvline(0,color='#737D86',lw=.75)
    for y,name,color,mark in [(1,'forecast_COT_increment','#C06100','s'),(0,'observed_COT_increment_OFFLINE','#80549A','D')]:
        c=S['comparisons'][name];x=c['delta'];lo,hi=c['ci95']
        b.errorbar(x,y,xerr=[[x-lo],[hi-x]],fmt=mark,ms=5,color=color,capsize=3,lw=1.3)
        b.text(.76,y+.18,f'{x:+.2f}',ha='right',va='bottom',fontsize=8,color=color)
    b.set(yticks=[1,0],yticklabels=['Forecast','Observed*'],xlim=(-.8,.8),ylim=(-.55,1.55),xticks=[-.5,0,.5])
    b.tick_params(axis='y',length=0);b.spines['left'].set_visible(False)
    b.grid(axis='x',color='#E5E8EC',lw=.6);b.set_xlabel('$\Delta$RMSE (W m$^{-2}$)',labelpad=4)
    fig.text(.54,.435,'c  Add COT from the same image source',fontweight='bold',fontsize=8.6)
    fig.text(.035,.035,'Paired 95% date-block intervals; negative values improve.  *Observed future imagery: offline only.',fontsize=8,color='#434C52')
    for ext in ['pdf','svg','png']: fig.savefig(OUT/f'source_replacement.{ext}',dpi=300)
    fig.savefig(OUT/'source_replacement_print_preview.png',dpi=150)
    plt.close(fig)

for name in ['summary.json','source_metrics.csv']:
    shutil.copy2(SRC/name,OUT/name)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((SRC/'MANIFEST.json').read_text())
manifest.update({'size_mm':[177,108],'source_summary_sha256':sha(SRC/'summary.json'),
                 'summary_copy_sha256':sha(OUT/'summary.json'),'script_sha256':sha(Path(__file__)),
                 'transformation_this_revision':'layout only; exact existing summary and per-source scores copied',
                 'historical_COT':'present in all arms; no future COT means no extra future correction'})
assert sha(SRC/'summary.json')==sha(OUT/'summary.json')
(OUT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Saved source replacement figure; audited summary copied without changes.')
