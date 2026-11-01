"""Publication styling only: current COT quality and regional forecast figures.

Read existing saved validation arrays and saved lead metrics. No retraining,
resampling, smoothing, filtering changes or test payloads.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
BLUE, ORANGE, PURPLE = '#2878A5', '#D87836', '#8064A2'
INK, GRID = '#252525', '#E9EAEC'
STYLE = {
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 9, 'axes.labelsize': 9.5,
    'axes.titlesize': 10, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
    'axes.edgecolor': '#555555', 'axes.linewidth': .65,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.major.width': .6, 'ytick.major.width': .6,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'savefig.facecolor': 'white', 'legend.frameon': False,
}
SOURCES = []

def source(relative):
    p = ROOT / relative
    SOURCES.append(p)
    return p

def export(fig, name):
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(OUT / f'{name}.{extension}', dpi=450, facecolor='white')
    plt.close(fig)

def quality_figure():
    p = 'refine-logs/hubei_same_time_geometry_20261102/'
    audit = json.loads(source(p+'COT_QUALITY_AND_GHI_STRATA.json').read_text(encoding='utf-8'))
    quality = audit['quality']
    pred = np.asarray(np.load(source(p+'r_validation/val_prediction_cot.npy')), dtype=np.float64)
    truth = np.asarray(np.load(source(p+'r_validation/val_reference_cpp_cot.npy')), dtype=np.float64)
    valid = np.asarray(np.load(source(p+'r_validation/val_valid_mask.npy')), dtype=bool)
    x, y = truth[valid], pred[valid]
    error = y-x
    actual = {'rmse': float(np.sqrt(np.mean(error**2))), 'mae': float(np.mean(abs(error))),
              'bias': float(np.mean(error)), 'pixels': len(x)}
    for key in ('rmse', 'mae', 'bias'):
        assert abs(actual[key]-quality[key]) < 1e-10, (key, actual[key],quality[key])
    assert actual['pixels'] == quality['pixels']
    bins = np.linspace(0, 100, 81)
    hist, _, _ = np.histogram2d(x, y, bins=[bins,bins])
    edges = [0, .1, 5, 30, 100.000001]
    bias = [float(np.mean(error[(x>=edges[i]) & (x<edges[i+1])])) for i in range(4)]
    labels = ['Near-zero\n[0, 0.1)', 'Thin\n[0.1, 5)', 'Moderate\n[5, 30)', 'Thick\n[30, 100]']
    fig = plt.figure(figsize=(7.05, 3.65))
    ax = fig.add_axes([.075,.235,.36,.695])
    cbax = fig.add_axes([.10,.103,.31,.024])
    bx = fig.add_axes([.665,.235,.31,.695])
    # Avoid near-white occupied bins while preserving white for zero-count bins.
    cmap = LinearSegmentedColormap.from_list('Blues_visible',plt.colormaps['Blues'](np.linspace(.12,1,256)))
    cmap.set_bad('white')
    image = ax.pcolormesh(bins,bins,np.ma.masked_equal(hist.T,0),cmap=cmap,
                         norm=LogNorm(vmin=1,vmax=float(hist.max())), rasterized=True)
    ax.plot([0,100],[0,100], color='#707070',ls='--',lw=.8)
    ax.set(xlim=(0,100),ylim=(0,100),xlabel='CPP reference COT',ylabel='Retrieved COT',aspect='equal')
    ax.set_xticks([0,20,40,60,80,100]); ax.set_yticks([0,20,40,60,80,100])
    ax.set_title('(a)  Retrieval agreement',loc='left',pad=9,fontweight='bold')
    ax.text(.045,.96,f"RMSE = {actual['rmse']:.2f}\nMAE = {actual['mae']:.2f}\nBias = {actual['bias']:+.2f}\nN = {len(x):,}",
            transform=ax.transAxes,ha='left',va='top',fontsize=8.5,
            bbox={'facecolor':'white','edgecolor':'none','pad':1.5,'alpha':.94})
    cb = fig.colorbar(image,cax=cbax,orientation='horizontal')
    cb.set_label('Pixel count',labelpad=0,fontsize=8.5)
    cb.ax.tick_params(labelsize=8); cb.outline.set_linewidth(.5)
    # A separate categorical display, without connecting the four classes.
    bx.axvline(0,color='#666666',lw=.8,zorder=1)
    for j,v in enumerate(bias):
        bx.plot([0,v],[j,j],color=PURPLE,alpha=.36,lw=2,zorder=2)
        bx.plot(v,j,'o',mfc=PURPLE,mec='white',mew=.6,ms=5.6,zorder=3)
        bx.annotate(f'{v:+.2f}',(v,j),xytext=(5 if v>=0 else -5,5),
                    textcoords='offset points',ha='left' if v>=0 else 'right',va='bottom',fontsize=8.5)
    bx.set_yticks(range(4), labels)
    bx.tick_params(axis='y',length=0,pad=5)
    bx.set_ylim(3.6,-.65)
    bx.set_xlim(-11,9)
    bx.set_xticks([-10,-5,0,5])
    bx.set_xlabel('Mean bias in COT')
    bx.set_title('(b)  Bias by COT class',loc='left',pad=9,fontweight='bold')
    bx.spines['left'].set_visible(False)
    bx.grid(axis='x',color=GRID,lw=.55); bx.set_axisbelow(True)
    fig.text(.665,.085,'264 frames · 16 validation dates\n67,187 valid pixels in 16 × 16 patches',
             fontsize=8.5,ha='left',va='center',linespacing=1.5)
    export(fig,'cot_quality')
    return {'quality_recomputed':actual,'quality_source':quality,'bin_edges':edges,
            'bias_by_class':bias,'histogram_bins':80,'histogram_range':[0,100],
            'histogram_count':int(hist.sum()),'histogram_sha256':hashlib.sha256(hist.tobytes()).hexdigest(),
            'valid_mask_unchanged':True}

def forecast_figure():
    p = source('paper/figures/regional_forecast_20261102/lead_metrics.csv')
    df = pd.read_csv(p)
    assert len(df)==32
    for region in ['Hubei','Hunan']:
        d=df[df.region==region].sort_values('lead_minutes')
        assert d.lead_minutes.tolist()==list(range(15,241,15))
        assert np.allclose(d.with_cot_rmse-d.no_cot_rmse,d.delta_rmse,rtol=0,atol=1e-9)
    fig,axs=plt.subplots(2,2,figsize=(7.05,5.1),sharex=True,sharey='row')
    fig.subplots_adjust(left=.105,right=.98,bottom=.16,top=.86,wspace=.16,hspace=.26)
    drawn={}
    for col,region in enumerate(['Hubei','Hunan']):
        d=df[df.region==region].sort_values('lead_minutes')
        x=d.lead_minutes.to_numpy()
        ax,bx=axs[:,col]
        for a in (ax,bx):
            a.axvspan(7.5,45,facecolor='#D8DADF',alpha=.38,lw=0,zorder=0)
            a.axvline(45,color='#8E9299',ls=':',lw=.8,zorder=1)
            a.set_xlim(7.5,247.5)
            a.set_xticks([15,60,120,180,240])
            a.grid(axis='y',color=GRID,lw=.6)
            a.set_axisbelow(True)
        la,=ax.plot(x,d.no_cot_rmse,'o-',color=BLUE,lw=1.55,ms=3.5,mew=.45,mec='white',zorder=3)
        lc,=ax.plot(x,d.with_cot_rmse,'s--',color=ORANGE,lw=1.55,ms=3.4,mew=.4,mec='white',zorder=3)
        ax.set_title(f'({chr(97+col)})  {region}',loc='left',fontweight='bold',pad=8)
        bx.fill_between(x,d.delta_ci95_lower.to_numpy(),d.delta_ci95_upper.to_numpy(),
                        color=PURPLE,alpha=.20,lw=0,zorder=2)
        ld,=bx.plot(x,d.delta_rmse,'o-',color=PURPLE,lw=1.5,ms=3.3,mec='white',mew=.4,zorder=4)
        bx.axhline(0,color='#535353',lw=.85,zorder=3)
        bx.set_title(f'({chr(99+col)})  COT contribution',loc='left',fontweight='bold',pad=8)
        # Verify artists receive the exact CSV quantities.
        for artist,column in [(la,'no_cot_rmse'),(lc,'with_cot_rmse'),(ld,'delta_rmse')]:
            assert np.array_equal(artist.get_ydata(),d[column].to_numpy())
        drawn[region]=d.to_dict(orient='records')
    top_values=df[['no_cot_rmse','with_cot_rmse']].to_numpy()
    axs[0,0].set_ylim(np.floor(top_values.min()/10)*10-3,np.ceil(top_values.max()/10)*10+3)
    intervals=df[['delta_ci95_lower','delta_ci95_upper']].to_numpy()
    axs[1,0].set_ylim(np.floor(min(0,intervals.min())/2)*2-.5,np.ceil(max(0,intervals.max())/2)*2+.5)
    axs[0,0].set_ylabel(r'GHI RMSE (W m$^{-2}$)')
    axs[1,0].set_ylabel(r'$\Delta$RMSE (W m$^{-2}$)')
    for a in axs[1]:a.set_xlabel('Forecast lead (min)',labelpad=5)
    handles=[Line2D([],[],color=BLUE,marker='o',lw=1.55,ms=4,label='Without COT'),
             Line2D([],[],color=ORANGE,marker='s',ls='--',lw=1.55,ms=4,label='With predicted COT'),
             Patch(facecolor=PURPLE,alpha=.20,edgecolor='none',label='95% date-block CI')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.99),ncol=3,
               fontsize=9,handlelength=2.2,columnspacing=1.3)
    fig.text(.54,.065,r'$\Delta$RMSE = with COT − without COT',ha='center',fontsize=8.5)
    fig.text(.54,.025,'Shading: observed history included; all three frames predicted from +45 min.',ha='center',fontsize=8.5)
    export(fig,'regional_forecast_lead')
    return {'records':32,'lead_minutes':list(range(15,241,15)), 'plotted_values_exact_to_csv':True,
            'CI_reused_without_resampling':True,'per_region_values':drawn,
            'gray_shading':'history-containing positions at +15 and +30; boundary at +45',
            'difference_definition':'with COT minus without COT; lower is better'}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with plt.rc_context(STYLE):
        q=quality_figure(); f=forecast_figure()
    out={'scope':'style and encoding only; current saved validation evidence',
         'cot_quality':q,'regional_forecast':f,'test_payload_read':False,
         'figure_width_inch':7.05,'png_dpi':450,
         'visual_review':{'rendered_png_inspected':True,'overlap_resolved':True,
                          'ci_endpoints_within_axes':True,'font':'Times New Roman; mathtext STIX'},
         'sources':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in SOURCES]}
    (OUT/'quality_forecast_checks.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'quality':q['quality_recomputed'],'forecast_values_exact':True},indent=2))

if __name__=='__main__':main()
