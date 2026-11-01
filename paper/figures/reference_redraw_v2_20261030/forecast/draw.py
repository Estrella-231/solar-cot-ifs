"""Reference-based redesign. Reuses V1 verified metrics without rescoring."""
from pathlib import Path
import csv,json,hashlib,shutil
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
PAPER=HERE.parents[2]
OLD=PAPER/'figures/revision_v1_20261030/evidence'
REF=PAPER/'reviews/20261030_figure_atlas/amt/fig11_original.png'

def read(name):
    with (OLD/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    leads=read('lead_metrics.csv');pairs=read('paired_comparisons.csv')
    x=np.array([float(r['lead_minutes']) for r in leads])
    assert np.array_equal(x,np.arange(15,241,15))
    C={'none':'#252525','hist':'#007C70','control':'#416BA5','forecast':'#CB6500'}
    cfg={'font.family':'Arial','font.size':8.4,'axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,
         'axes.linewidth':.6,'grid.color':'#D9DDE1','grid.linewidth':.45,'lines.linewidth':1.1,
         'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
         'axes.labelcolor':'#171717','xtick.color':'#171717','ytick.color':'#171717','mathtext.fontset':'stix'}
    with mpl.rc_context(cfg):
        fig=plt.figure(figsize=(177/25.4,112/25.4),facecolor='white')
        gs=fig.add_gridspec(2,2,left=.115,right=.982,top=.94,bottom=.19,wspace=.31,hspace=.64,height_ratios=[1.05,1])
        axs=np.array([[fig.add_subplot(gs[i,j]) for j in range(2)] for i in range(2)])
        ax=axs[0,0]
        lines=[]
        for key,col,marker,ls,label in [
            ('no_cot_rmse',C['none'],'o','-','No COT'),
            ('history_cot_rmse',C['hist'],'s','-','Historical COT'),
            ('history_control_rmse',C['control'],'^','--','History + matched control'),
            ('history_forecast_cot_rmse',C['forecast'],'x',':','History + forecast COT')]:
            line,=ax.plot(x,[float(r[key]) for r in leads],color=col,marker=marker,ms=3,markevery=1,ls=ls,label=label)
            lines.append(line)
        ax.set(xlim=(8,247),ylim=(87,115),xticks=[15,60,120,180,240],yticks=[90,100,110],ylabel='RMSE (W m$^{-2}$)',xlabel='Forecast horizon (min)')
        ax.text(.02,1.07,'(a) GHI forecast error',transform=ax.transAxes,fontweight='bold')
        ax.yaxis.set_minor_locator(MultipleLocator(5))
        ax=axs[0,1]
        y=np.array([float(r['forecast_increment_delta']) for r in leads])
        lo=np.array([float(r['forecast_increment_ci95_low']) for r in leads])
        hi=np.array([float(r['forecast_increment_ci95_high']) for r in leads])
        ax.fill_between(x,lo,hi,color=C['forecast'],alpha=.15,lw=0)
        ax.plot(x,y,color=C['forecast'],marker='s',ms=3)
        ax.axhline(0,color='#3E3E3E',ls='--',lw=.8)
        ax.set(xlim=(8,247),ylim=(-.018,.073),xticks=[15,60,120,180,240],yticks=[0,.02,.04,.06],ylabel='$\Delta$RMSE (W m$^{-2}$)',xlabel='Forecast horizon (min)')
        ax.text(.02,1.07,'(b) Additional forecast-COT effect',transform=ax.transAxes,fontweight='bold')
        ax.yaxis.set_minor_locator(MultipleLocator(.01))
        stations=['Niushou','Wangzhai','Mizhuang','Equal-station mean']
        labels=['Niushou','Wangzhai','Mizhuang','Mean']
        for j,(comparison,col,title,xlim,xticks) in enumerate([
          ('Historical COT - no COT',C['hist'],'(c) Historical COT − no COT',(-8,4),[-8,-4,0,4]),
          ('Forecast COT head - control head',C['forecast'],'(d) Forecast COT − control',(-.04,.125),[-.04,0,.04,.08,.12])]):
            ax=axs[1,j]
            for i,st in enumerate(stations):
                r=next(r for r in pairs if r['comparison']==comparison and r['station']==st)
                point=float(r['delta_rmse_wm2']);low=float(r['ci95_low']);high=float(r['ci95_high'])
                ax.errorbar(point,i,xerr=[[point-low],[high-point]],fmt='D',ms=4,color=col,capsize=2.5,lw=1)
                offsets=[-.10,0,.10]
                for off,s in zip(offsets,[42,43,44]):
                    ax.plot(float(r[f'seed{s}_delta']),i+off,'o',ms=3,mfc='white',mec=col,mew=.9,zorder=3)
            ax.axvline(0,color='#3E3E3E',ls='--',lw=.8)
            ax.axhline(2.5,color='#BCC4CB',ls=':',lw=.6)
            ax.set(xlim=xlim,xticks=xticks,ylim=(3.45,-.45),yticks=range(4),yticklabels=labels if j==0 else [],xlabel='$\Delta$RMSE (W m$^{-2}$)')
            ax.text(.02,1.09,title,transform=ax.transAxes,fontweight='bold')
            ax.grid(axis='x',which='major')
        for ax in axs[0]:
            ax.grid(which='major',axis='both');ax.grid(which='minor',axis='y',alpha=.35)
        for ax in axs.ravel():
            for spine in ax.spines.values():spine.set_color('#777E84')
            ax.tick_params(length=2.4,width=.6,pad=2)
            ax.set_axisbelow(True)
        fig.legend(lines,[h.get_label() for h in lines],loc='lower center',bbox_to_anchor=(.53,.018),ncol=2,frameon=False,columnspacing=1.9,handlelength=2.2,handletextpad=.6,labelspacing=.45)
        fig.savefig(HERE/'forecast_ghi_v2.pdf')
        fig.savefig(HERE/'forecast_ghi_v2.svg')
        fig.savefig(HERE/'forecast_ghi_v2.png',dpi=300)
        plt.close(fig)
    for name in ['lead_metrics.csv','paired_comparisons.csv','sample_support.csv']:
        shutil.copy2(OLD/name,HERE/name)
    shutil.copy2(REF,HERE/'reference_amt_fig11.png')
    # Unchanged original screenshot and previous version are shown next to the redesign.
    items=[('原论文截图 · AMT (2024), Fig.11 · PDF p.17',REF),
           ('上一版 · 177 × 132 mm · 标题、外部标签与空白占比过高',OLD/'forecast_ghi.png'),
           ('按原图重排 · 177 × 112 mm · 同一数据、同一配对区间',HERE/'forecast_ghi_v2.png')]
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',24)
    small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    width=1180;padding=30; rendered=[]
    for title,path in items:
        im=Image.open(path).convert('RGB');im.thumbnail((1120,820),Image.Resampling.LANCZOS)
        pane=Image.new('RGB',(width,im.height+75),'white');d=ImageDraw.Draw(pane)
        d.text((padding,12),title,font=font,fill='#171717')
        pane.paste(im,((width-im.width)//2,60));rendered.append(pane)
    canvas=Image.new('RGB',(width,sum(im.height for im in rendered)+50),'#E3E6E9');y0=0
    for pane in rendered:canvas.paste(pane,(0,y0));y0+=pane.height
    d=ImageDraw.Draw(canvas);d.text((padding,y0+12),'Source: Papachristopoulou et al., AMT 17, 1851-1877 (2024), Fig.11. Comparison only.',font=small,fill='#333333')
    canvas.save(HERE/'reference_old_new_comparison.png')
    (HERE/'MANIFEST.json').write_text(json.dumps({'reference':'https://amt.copernicus.org/articles/17/1851/2024/','reference_figure':11,'reference_pdf_page':17,'reference_printed_page':1867,'reference_license':'CC BY 4.0, original authors; screenshot unmodified except display resizing','reference_image_sha256':sha(REF),'old_figure_sha256':sha(OLD/'forecast_ghi.png'),'source_metrics':{n:sha(OLD/n) for n in ['lead_metrics.csv','paired_comparisons.csv','sample_support.csv']},'evaluation_changed':False,'bootstrap_intervals_changed':False,'input_frame_scope_changed':False,'size_mm':[177,112],'font_min_pt':8,'outputs':{p.name:sha(p) for p in HERE.glob('forecast_ghi_v2.*')}},indent=2),encoding='utf-8')
    print('Rendered unchanged numerical evidence using AMT Fig.11 layout reference.')

if __name__=='__main__':main()
