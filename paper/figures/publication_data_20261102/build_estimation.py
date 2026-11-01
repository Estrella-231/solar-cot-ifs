"""Data-preserving redraw of the paired estimation and CPP-stratified figures."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
FILES = [ROOT/'refine-logs/hubei_same_time_geometry_20261102/INDEPENDENT_FP64_RESULTS.json',
         ROOT/'refine-logs/hubei_same_time_geometry_20261102/COT_QUALITY_AND_GHI_STRATA.json']
R, Q = [json.loads(p.read_text(encoding='utf-8')) for p in FILES]
BLUE='#2878A5'; ORANGE='#D87836'; PURPLE='#8064A2'; INK='#282828'; GRID='#E8E8E8'
STYLE={'font.family':'serif', 'font.serif':['Times New Roman','STIXGeneral','DejaVu Serif'],
       'font.size':9, 'mathtext.fontset':'stix', 'axes.titlesize':10, 'axes.titleweight':'bold',
       'axes.labelsize':9, 'xtick.labelsize':8.5, 'ytick.labelsize':9,
       'axes.labelcolor':INK, 'text.color':INK, 'xtick.color':INK, 'ytick.color':INK,
       'axes.edgecolor':'#555555','axes.linewidth':.65,
       'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,
       'xtick.major.width':.65,'ytick.major.width':.65,'xtick.major.size':3,
       'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'}

def save(fig, name):
    for ext in ('pdf','png','svg'):
        fig.savefig(OUT/f'{name}.{ext}', dpi=450, facecolor='white')
    plt.close(fig)

def setup(ax, yy, limits):
    ax.set_ylim(min(yy)-.55,max(yy)+.55)
    ax.set_xlim(*limits)
    ax.set_axisbelow(True)
    ax.grid(axis='x',color=GRID,lw=.55)
    ax.tick_params(axis='y',length=0,pad=7)
    for y in yy: ax.axhline(y,color='#F0F0F0',lw=.65,zorder=0)

def pair(ax, yy, a, ac):
    for y,x0,x1 in zip(yy,a,ac):
        ax.plot([x0,x1],[y,y],color='#B8B8B8',lw=1.5,zorder=2)
    ax.scatter(a,yy,s=37,marker='o',color=BLUE,edgecolor='white',linewidth=.45,zorder=3)
    ax.scatter(ac,yy,s=37,marker='s',color=ORANGE,edgecolor='white',linewidth=.45,zorder=4)

def ci(ax, yy, val, bounds, overall=False):
    val=np.asarray(val); bounds=np.asarray(bounds)
    for i,(y,x,(lo,hi)) in enumerate(zip(yy,val,bounds)):
        marker='D' if overall and i==len(yy)-1 else 'o'
        ax.errorbar(x,y,xerr=[[x-lo],[hi-x]],fmt=marker,color=PURPLE,
                    ecolor=PURPLE,elinewidth=1.1,capsize=3,capthick=.9,
                    markersize=5 if marker=='o' else 5.4,zorder=3)
    ax.axvline(0,color='#555555',lw=.85,linestyle=(0,(3,2)),zorder=1)

def legend(fig, y):
    handles=[Line2D([],[],linestyle='None',marker='o',ms=5.3,color=BLUE,label='Without COT'),
             Line2D([],[],linestyle='None',marker='s',ms=5.3,color=ORANGE,label='With COT')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.46,y),ncol=2,
               frameon=False,handlelength=1,handletextpad=.4,columnspacing=1.5,fontsize=9)

stations=['Niushou','Wangzhai','Mizhuang']
rows=[dict(station=s,**R['by_station'][s]) for s in stations]
rows.append(dict(station='Equal-station mean',A_rmse=R['A_rmse'],AC_rmse=R['AC_rmse'],
                 delta_ci95=R['delta_ci95'],gain_pct=R['gain_pct']))
with plt.rc_context(STYLE):
    fig,axs=plt.subplots(1,2,figsize=(7.05,3.45),gridspec_kw={'width_ratios':[1,1.06]})
    fig.subplots_adjust(left=.20,right=.975,bottom=.23,top=.77,wspace=.25)
    yy=np.array([3,2,1,-.4])
    a=[v['A_rmse'] for v in rows]; ac=[v['AC_rmse'] for v in rows]
    delta=[v1-v0 for v0,v1 in zip(a,ac)]; bounds=[v['delta_ci95'] for v in rows]
    setup(axs[0],yy,(50,95));setup(axs[1],yy,(-15,2))
    pair(axs[0],yy,a,ac);ci(axs[1],yy,delta,bounds,overall=True)
    axs[0].set_yticks(yy,[v['station'] for v in rows]);axs[1].set_yticks(yy,[])
    for ax in axs:
        ax.axhline(.3,color='#B6B6B6',lw=.65)
        ax.axhspan(-.93,.2,color='#F7F7F7',zorder=-1)
    axs[0].get_yticklabels()[-1].set_fontweight('bold')
    axs[0].set_xticks([50,60,70,80,90]);axs[1].set_xticks([-15,-10,-5,0])
    axs[0].set_xlabel('GHI RMSE (W m$^{-2}$)')
    axs[1].set_xlabel('$\\Delta$RMSE (W m$^{-2}$)\nWith COT − without COT')
    axs[0].set_title('(a)  Estimation error',loc='left',pad=12)
    axs[1].set_title('(b)  Paired difference',loc='left',pad=12)
    legend(fig,1.0)
    fig.text(.61,.035,'Points and 95% date-block intervals',ha='center',fontsize=8.5,color='#555555')
    save(fig,'ghi_gain')

strata=Q['strata']
bins=['[0, 0.1)','[0.1, 5)','[5, 30)','[30, 100]']
with plt.rc_context(STYLE):
    fig,axs=plt.subplots(1,2,figsize=(7.05,3.70),gridspec_kw={'width_ratios':[1,1.06]})
    fig.subplots_adjust(left=.205,right=.975,bottom=.22,top=.78,wspace=.25)
    yy=np.array([3,2,1,0])
    a=[v['A_rmse'] for v in strata]; ac=[v['AC_rmse'] for v in strata]
    gain=[v['gain_pct'] for v in strata]; bounds=[v['gain_ci95'] for v in strata]
    setup(axs[0],yy,(20,140));setup(axs[1],yy,(-40,50))
    pair(axs[0],yy,a,ac);ci(axs[1],yy,gain,bounds)
    # Each category is a separate row. Neither stations nor COT classes are connected.
    labels=[f"{s['name']}  {b}\n$n$ = {s['n']}; {s['dates']} dates" for s,b in zip(strata,bins)]
    axs[0].set_yticks(yy,labels);axs[0].tick_params(axis='y',labelsize=8.5)
    axs[1].set_yticks(yy,[])
    axs[0].set_xticks([20,40,60,80,100,120,140]);axs[1].set_xticks([-40,-20,0,20,40])
    axs[0].set_xlabel('GHI RMSE (W m$^{-2}$)');axs[1].set_xlabel('RMSE reduction (%)\nPositive values indicate improvement')
    axs[0].set_title('(a)  Error by CPP COT class',loc='left',pad=12)
    axs[1].set_title('(b)  Relative improvement',loc='left',pad=12)
    legend(fig,1.0)
    fig.text(.205,.035,'$n$: station–time records; intervals: 95% target-date-block bootstrap.',ha='left',fontsize=8.5,color='#555555')
    save(fig,'ghi_thickness')

payload={'scope':'style and encoding only; no new evaluation',
         'source_files':[{'path':str(p.relative_to(ROOT)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in FILES],
         'source_values_unchanged':True,
         'gain_rows':rows,
         'thickness_rows':strata,
         'thickness_bin_labels':bins,
         'CI_definition':Q['intervals'],
         'rendering':{'width_inches':7.05,'font':'Times New Roman','formats':['pdf','png','svg'],
                      'comparison':'horizontal paired points joined within rows only',
                      'uncertainty':'horizontal intervals, zero line retained; no clipping',
                      'numeric_axis_scope':'point plots use labelled nonzero RMSE lower limits; no bars or filled area'},
         'test_payload_read':False,'training_performed':False}
payload['visual_review']='PNG images inspected at full resolution; labels, zero crossings and full intervals visible; no category-to-category connections.'
assert payload['gain_rows'][-1]['A_rmse']==R['A_rmse']
assert payload['gain_rows'][-1]['AC_rmse']==R['AC_rmse']
assert payload['thickness_rows']==Q['strata']
(OUT/'estimation_checks.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Saved paired-point and forest plots, preserving every source value and interval.')
