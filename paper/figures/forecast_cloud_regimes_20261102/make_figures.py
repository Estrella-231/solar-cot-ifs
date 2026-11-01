"""Manuscript layouts from frozen, versioned forecast diagnostics. No inference."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from PIL import Image
from pypdf import PdfReader

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
SOURCE=ROOT/'refine-logs/forecast_thick_cloud_events_20261102'
NO='#2878A5'; YES='#D87836'; OBS='#202830'; RET='#8064A2'; LIGHT='#E6EBEE'
STYLE={'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
       'axes.titlesize':9,'xtick.labelsize':7.2,'ytick.labelsize':7.2,
       'axes.spines.top':False,'axes.spines.right':False,
       'axes.linewidth':.65,'pdf.fonttype':42,'ps.fonttype':42,
       'svg.fonttype':'none','savefig.facecolor':'white','text.color':OBS,
       'axes.labelcolor':OBS,'xtick.color':OBS,'ytick.color':OBS}
plt.rcParams.update(STYLE)
checks=[]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(fig,name):
    for ext in ('pdf','svg','png'):
        fig.savefig(OUT/f'{name}.{ext}',dpi=400,facecolor='white')
    p=OUT/f'{name}.png'
    with Image.open(p) as image:
        if image.mode=='RGBA':assert image.getchannel('A').getextrema()==(255,255)
        image.convert('RGB').save(p,dpi=(400,400))
    pdf=PdfReader(OUT/f'{name}.pdf');box=pdf.pages[0].mediabox
    checks.append({'file':name,'width_mm':float(box.width)*25.4/72,
                   'height_mm':float(box.height)*25.4/72,
                   'png_mode':Image.open(p).mode,'png_size':list(Image.open(p).size),
                   'pdf_pages':len(pdf.pages),'pdf_sha256':sha(OUT/f'{name}.pdf')})
    plt.close(fig)

def regimes():
    records=json.loads((SOURCE/'RESULTS.json').read_text())['metrics']
    bins=['Near-zero','Thin','Moderate','Thick']; ranges=['[0, 0.1)','[0.1, 5)','[5, 30)','[30, 100]']
    fig=plt.figure(figsize=(180/25.4,112/25.4))
    # A row is a cloud regime; distinct panels separate absolute errors and effects.
    fig.text(.016,.977,'Forecast GHI by reference cloud optical thickness',va='top',fontsize=11,weight='bold')
    fig.legend(handles=[Line2D([],[],color=NO,marker='o',ls='',label='Without COT'),Line2D([],[],color=YES,marker='s',ls='',label='With forecast COT')],loc='upper right',bbox_to_anchor=(.995,.945),ncol=2,frameon=False,fontsize=7.4,columnspacing=1.2,handletextpad=.3)
    for ri,reg in enumerate(['Hubei','Hunan']):
        top=.855-ri*.405; bottom=top-.285
        a=fig.add_axes([.245,bottom,.295,.285]);b=fig.add_axes([.655,bottom,.315,.285])
        a.set_title(f'({chr(97+2*ri)}) {reg}: absolute error',loc='left',pad=6,weight='bold',fontsize=8.5)
        b.set_title(f'({chr(98+2*ri)}) Paired COT effect',loc='left',pad=6,weight='bold',fontsize=8.5)
        rs=[next(q for q in records if q['region']==reg and q['scope']=='all' and q['stratum']==s) for s in bins]
        for j,q in enumerate(rs):
            y=3-j
            for ax in (a,b):
                if j==3:ax.axhspan(y-.43,y+.43,color='#EFF6F4',zorder=-3)
            a.plot([q['A_rmse'],q['AC_rmse']],[y,y],color='#B2BEC5',lw=2.3,zorder=1)
            a.plot(q['A_rmse'],y,'o',ms=5,color=NO,zorder=2)
            a.plot(q['AC_rmse'],y,'s',ms=4.8,color=YES,zorder=3)
            lo,hi=q['delta_ci95'];v=q['delta_rmse']
            b.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt='D',color=YES,ms=4.2,lw=1.35,capsize=2.6)
            ypos=bottom+(.5+y)/4*.285
            fig.text(.018,ypos+.009,bins[j],fontsize=8.5,weight='bold' if j==3 else 'normal',va='center')
            fig.text(.018,ypos-.018,ranges[j],fontsize=6.9,color='#58646C',va='center')
            fig.text(.195,ypos+.008,f"{q['n_station_targets']}",ha='right',va='center',fontsize=8)
            fig.text(.195,ypos-.018,f"{q['n_dates']} dates",ha='right',va='center',fontsize=6.7,color='#58646C')
        fig.text(.018,top+.012,'CPP COT',fontsize=7.4,color='#58646C')
        fig.text(.195,top+.012,'Targets',ha='right',fontsize=7.4,color='#58646C')
        for ax in (a,b):
            ax.set_ylim(-.5,3.5);ax.set_yticks([]);ax.spines['left'].set_visible(False);ax.tick_params(length=3);ax.grid(axis='x',color=LIGHT,lw=.6,zorder=-5)
        a.set_xlim(0,200);a.set_xticks([0,50,100,150,200])
        b.set_xlim(-32,28);b.set_xticks([-30,-15,0,15]);b.axvline(0,color='#7C878F',lw=.8,ls='--')
        if ri==1:
            a.set_xlabel('GHI RMSE (W m$^{-2}$)',labelpad=4)
            b.set_xlabel('With − without COT (W m$^{-2}$)',labelpad=4)
    fig.text(.018,.020,'Points: equal-station / equal-seed RMSE. Whiskers: 95% paired target-date bootstrap intervals.',fontsize=7,color='#58646C')
    save(fig,'forecast_regimes')

def rgb(z):
    image=np.moveaxis(z[[2,1,0]],0,-1)
    valid=np.isfinite(image).all(-1)
    image=np.sqrt(np.clip(np.nan_to_num(image),0,1));image[~valid]=.83
    return image

def tile(ax,arr,center,rgb_image=False):
    n=arr.shape[-1];r,c=center
    if rgb_image:im=ax.imshow(rgb(arr),interpolation='nearest')
    else:im=ax.imshow(arr,cmap=plt.colormaps['viridis'].with_extremes(bad='#D9DDE1'),norm=Normalize(0,80),interpolation='nearest')
    ax.plot(c,r,'+',ms=5.4,mew=1.8,color='black');ax.plot(c,r,'+',ms=5.4,mew=.75,color='white')
    if n==16:ax.add_patch(Rectangle((c-1.5,r-1.5),3,3,fill=False,ec='white',lw=.6,ls='--'))
    ax.set_xticks([]);ax.set_yticks([])
    for sp in ax.spines.values():sp.set_visible(True);sp.set_color('#B4BEC5');sp.set_linewidth(.4)
    return im

def cases():
    metadata=json.loads((SOURCE/'CASE_SELECTION.json').read_text())
    cs=[metadata['cases'][0],metadata['cases'][3]]
    fig=plt.figure(figsize=(180/25.4,190/25.4))
    fig.text(.018,.981,'Cloud evolution and the direction of the GHI correction',fontsize=11,weight='bold',va='top')
    fig.text(.018,.957,'Observed future fields are shown only for diagnosis.',fontsize=7.5,color='#58646C',va='top')
    audit=[]
    for i,c in enumerate(cs):
        upper=.925-i*.425;lower=upper-.365
        z=np.load(SOURCE/'cases'/f"{c['event_id']}.npz")
        times=pd.to_datetime(z['times_bjt'],format='%Y%m%d%H%M')
        q=pd.read_csv(SOURCE/'cases'/f"{c['event_id']}_timeseries.csv")
        assert len(q)==len(times)
        assert (pd.to_datetime(q.iloc[:,0]).to_numpy()==times.to_numpy()).all()
        ctr=z['center'];fc=z['forecast_center'];native=z['forecast_cot'].shape[-1]
        cpp=np.where(z['cpp_mask'],z['cpp'],np.nan)
        selected=[int(np.where(times==pd.Timestamp(t))[0][0]) for t in c['display_times']]
        title=f"({chr(97+i)}) {c['station'].capitalize()}, {c['region']}  ·  {pd.Timestamp(c['init']):%d %b %Y}"
        fig.text(.018,upper,title,va='top',weight='bold',fontsize=9.2)
        fig.text(.98,upper,f"{c['direction'].capitalize()} segment",ha='right',va='top',fontsize=8.2,color=YES if i==0 else '#9E533A')
        fig.text(.018,upper-.023,f"Initialized {pd.Timestamp(c['init']):%H:%M} BJT",fontsize=7.3,color='#58646C',va='top')
        # Left: three native-pixel snapshots. Right: physical trajectories.
        gx=.17;tile_w=.095;tile_h=tile_w*180/190;gapx=.016;gapy=.009
        top=upper-.079-tile_h
        names=['Observed AGRI\n16 × 16','CPP reference\n16 × 16',f'Forecast COT\n{native} × {native}'+(' crop' if native==3 else '')]
        for row in range(3):
            y=top-row*(tile_h+gapy)
            fig.text(.018,y+tile_h/2,names[row],va='center',fontsize=7.1,color=OBS)
            for col,k in enumerate(selected):
                ax=fig.add_axes([gx+col*(tile_w+gapx),y,tile_w,tile_h])
                arr=[z['observed_agri'],cpp,z['forecast_cot']][row]
                im=tile(ax,arr[k],fc if row==2 else ctr,row==0)
                if row==0:
                    lead=int((times[k]-pd.Timestamp(c['init'])).total_seconds()/60)
                    ax.set_title(f'{times[k]:%H:%M}\n+{lead} min',fontsize=7.2,pad=4)
        x=.595;w=.383
        ca=fig.add_axes([x,upper-.191,w,.114]);ga=fig.add_axes([x,upper-.365,w,.114],sharex=ca)
        cpp_s=cpp[:,ctr[0],ctr[1]];forecast=z['forecast_cot'][:,fc[0],fc[1]];retr=z['observed_cot'][:,ctr[0],ctr[1]]
        ca.plot(times,cpp_s,color=OBS,marker='o',ms=2.8,lw=.9,label='CPP reference')
        ca.plot(times,retr,color=RET,ls=':',lw=1.1,marker='.',ms=2.8,label='R(observed AGRI)')
        ca.plot(times,forecast,color=YES,marker='s',ms=2.3,lw=1.2,label='Forecast COT')
        ca.axhline(30,color='#8D969D',ls='--',lw=.6)
        ca.set_ylim(0,70);ca.set_yticks([0,30,60]);ca.set_ylabel('Station COT',labelpad=3)
        ca.set_title('COT at the station pixel',loc='left',fontsize=8,pad=4)
        ca.tick_params(labelbottom=False)
        ga.plot(times,q.truth,color=OBS,marker='o',ms=2.7,lw=1.15,label='Observed GHI')
        ga.plot(times,q.A42,color=NO,ls='--',lw=1.2,marker='^',ms=3.0,mfc='white',mew=.7,label='Without COT')
        ga.plot(times,q.AC42,color=YES,lw=1.35,marker='s',ms=2.7,label='With forecast COT')
        ga.set_ylabel('GHI (W m$^{-2}$)',labelpad=3);ga.set_xlabel('Target interval end (BJT)',labelpad=3)
        ga.set_ylim(0,160 if i==0 else 800);ga.set_yticks([0,40,80,120,160] if i==0 else [0,200,400,600,800])
        ga.set_title('15-minute mean irradiance',loc='left',fontsize=8,pad=4)
        for ax in [ca,ga]:
            ax.set_xlim(times[0]-pd.Timedelta(minutes=3),times[-1]+pd.Timedelta(minutes=3))
            ax.axvspan(pd.Timestamp(c['start']),pd.Timestamp(c['end']),color='#E9C46A',alpha=.18,zorder=-3)
            for k in selected:
                ax.axvline(times[k],color='#B3BAC1',ls=':',lw=.65,zorder=.5)
            ax.grid(axis='y',color=LIGHT,lw=.55,zorder=-5)
            ax.xaxis.set_major_locator(mdates.HourLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.tick_params(labelsize=7,length=2.5,pad=2)
        m=times.to_series().between(pd.Timestamp(c['start']),pd.Timestamp(c['end'])).to_numpy()&q.truth.notna().to_numpy()
        score=[float(np.sqrt(np.mean((q[key].to_numpy()[m]-q.truth.to_numpy()[m])**2))) for key in ['A42','AC42']]
        fig.text(.018,lower-.014,f"Shown segment RMSE: {score[0]:.1f} → {score[1]:.1f} W m$^{{-2}}$",fontsize=7.4,va='top')
        audit.append({'case':c['event_id'],'display_times':c['display_times'],'data_sha256':sha(SOURCE/'cases'/f"{c['event_id']}.npz"),'native_forecast_shape':list(z['forecast_cot'].shape),'shown_seed42_rmse':score})
    cbax=fig.add_axes([.17,.064,.317,.009]);cbar=fig.colorbar(im,cax=cbax,orientation='horizontal',extend='max',ticks=[0,30,60,80]);cbar.ax.tick_params(labelsize=7,length=2,pad=2);fig.text(.17,.079,'Cloud optical thickness',fontsize=7.4)
    fig.legend(handles=[Line2D([],[],color=OBS,marker='o',ms=3,lw=1,label='CPP / observed GHI'),Line2D([],[],color=RET,lw=1.2,ls=':',marker='.',ms=2.8,label='R(observed AGRI)'),Line2D([],[],color=YES,marker='s',ms=3,lw=1.2,label='Forecast COT / GHI + COT'),Line2D([],[],color=NO,lw=1.2,ls='--',marker='^',ms=3,mfc='white',mew=.7,label='GHI without COT')],loc='lower right',bbox_to_anchor=(.99,.01),ncol=1,frameon=False,fontsize=6.8,handlelength=2,columnspacing=.5,labelspacing=.45)
    fig.text(.018,.017,'+ station pixel; dashed box: 3 × 3 input. Grey: missing CPP.\nShading: thick-reference segment. GHI: seed 42; y-scales differ.',fontsize=6.7,color='#58646C',va='bottom')
    save(fig,'forecast_events')
    return audit

if __name__=='__main__':
    regimes();audit=cases()
    manifest={'state':'RENDERED','source_directory':str(SOURCE),'journal':'not selected; general manuscript layout',
              'outputs':checks,'case_checks':audit,'source_hashes':{p.name:sha(p) for p in [SOURCE/'RESULTS.json',SOURCE/'CASE_SELECTION.json',SOURCE/'CASE_SOURCE_CHECK.json']},
              'statistics':'Equal station and seed RMSE; 5000 paired target-date bootstrap draws. No rescore or sample changes.',
              'images':'Actual saved native arrays. Observed AGRI RGB=C03/C02/C01; sqrt(clip(reflectance,0,1)); COT common 0..80 with upper extension; no smoothing. Hunan 3x3 forecast crop is enlarged for display, not spatially interpolated.',
              'display_selection':'Same frozen 3 times per case: CPP peak minus 30 min / CPP peak / final GHI-valid target.',
              'case_uncertainty':'Cases selected conditional on improvement/degradation; not independent weather frequency evidence.',
              'software_citation':'Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065 (latest v2 record checked 2026-11-02).'}
    (OUT/'FIGURE_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))
