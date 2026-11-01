"""Restyle the two already-selected observed-image cases; no reselection."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize

OUT=Path(__file__).resolve().parent
PAPER=OUT.parents[1]
SOURCE=PAPER/'figures/current_events_20261102'
BLUE='#2878A5'; ORANGE='#D87836'; PURPLE='#8064A2'
df=pd.read_csv(SOURCE/'event_trajectories.csv', dtype={'time_bjt':str,'date':str})
fields=np.load(SOURCE/'selected_snapshot_fields.npz')
audit=json.loads((SOURCE/'EVENT_SELECTION_AND_PROVENANCE.json').read_text(encoding='utf-8'))
style={'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
       'font.size':9,'mathtext.fontset':'stix','axes.labelsize':9,'axes.titlesize':10,
       'xtick.labelsize':8.5,'ytick.labelsize':8.5,
       'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
       'xtick.major.width':.6,'ytick.major.width':.6,
       'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
       'text.color':'#222222','axes.labelcolor':'#222222','axes.edgecolor':'#555555'}
checks=[]
with plt.rc_context(style):
    fig=plt.figure(figsize=(7.05,6.35))
    outer=fig.add_gridspec(2,2,left=.075,right=.985,bottom=.145,top=.92,
                          width_ratios=[1.05,1.40],wspace=.29,hspace=.48)
    ghi_axes=[]
    for row,case in enumerate(audit['cases']):
        date=case['date']; sub=df.loc[df.date==date].copy()
        ts=pd.to_datetime(sub.time_bjt,format='%Y%m%d%H%M')
        assert len(sub)==17
        imgrid=outer[row,0].subgridspec(2,3,hspace=.17,wspace=.055)
        for j,k in enumerate(case['snapshots_bjt']):
            raw=fields[f'{date}_{k[-4:]}_agri']
            cot=fields[f'{date}_{k[-4:]}_cot']
            rgb=np.sqrt(np.clip(raw[[2,1,0]].transpose(1,2,0),0,1))
            for r,arr in [(0,rgb),(1,cot)]:
                ax=fig.add_subplot(imgrid[r,j])
                if r==0:ax.imshow(arr,interpolation='nearest')
                else: m=ax.imshow(arr,cmap='magma',norm=Normalize(0,70),interpolation='nearest')
                ax.plot(6,8,'+',color='white',ms=5.5,mew=.9)
                ax.add_patch(Rectangle((4.5,6.5),3,3,fill=False,edgecolor='white',linestyle='--',lw=.65))
                ax.set_xticks([]);ax.set_yticks([])
                for sp in ax.spines.values():sp.set_visible(True);sp.set_color('#777777');sp.set_linewidth(.35)
                if r==0:ax.set_title(pd.Timestamp(k[:8]+' '+k[8:10]+':'+k[10:]).strftime('%H:%M'),fontsize=9,pad=4)
                if j==0:ax.set_ylabel('AGRI' if r==0 else 'COT',labelpad=5)
            checks.append({'key':k,'native_grid':list(cot.shape),'rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'cot_sha256':hashlib.sha256(cot.tobytes()).hexdigest()})
        grid=outer[row,1].subgridspec(2,1,height_ratios=[.8,1.15],hspace=.20)
        ca=fig.add_subplot(grid[0]);ga=fig.add_subplot(grid[1],sharex=ca);ghi_axes.append(ga)
        ca.plot(ts,sub.COT,'o-',color=PURPLE,lw=1.25,ms=2.6,label='Retrieved')
        ca.plot(ts,sub.CPP_COT,linestyle='None',marker='x',color='#555555',ms=3.7,mew=.8,label='CPP reference')
        ca.set_ylim(0,70);ca.set_yticks([0,30,60]);ca.set_ylabel('Station COT',labelpad=4)
        ca.tick_params(axis='x',labelbottom=False)
        if row==0:
            ca.legend(loc='lower left',frameon=False,ncol=2,fontsize=8,handlelength=1.3,columnspacing=.9,borderpad=.1)
        for key,c,mk,ls,label in [('truth','#222222','o','-','Observed'),('A',BLUE,'s','--','Without COT'),('AC',ORANGE,'^','-','With COT')]:
            ga.plot(ts,sub[key],color=c,marker=mk,ls=ls,lw=1.2,ms=2.7,label=label)
        ga.set_ylim(0,150 if row==0 else 500)
        ga.set_yticks([0,50,100,150] if row==0 else [0,250,500])
        ga.set_ylabel('GHI (W m$^{-2}$)',labelpad=3)
        ga.set_xlabel('Interval end (BJT)',labelpad=3)
        ga.xaxis.set_major_locator(mdates.HourLocator())
        ga.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ga.set_xlim(ts.iloc[0]-pd.Timedelta(minutes=5),ts.iloc[-1]+pd.Timedelta(minutes=5))
        for ax in (ca,ga):
            ax.grid(axis='y',color='#e6e6e6',lw=.5);ax.set_axisbelow(True)
            for snap in case['snapshots_bjt']:
                ax.axvline(pd.to_datetime(snap,format='%Y%m%d%H%M'),color='#dddddd',lw=.55,ls=':',zorder=0)
        bbox=outer[row,:].get_position(fig)
        fig.text(bbox.x0,bbox.y1+.035,f"({'ab'[row]})  Niushou · {pd.to_datetime(date,format='%Y%m%d'):%d %B %Y}",ha='left',va='bottom',fontsize=10,fontweight='bold')
        fig.text(bbox.x1,bbox.y1+.035,f"RMSE: {case['A_rmse_seedmean']:.1f} → {case['AC_rmse_seedmean']:.1f} W m$^{{-2}}$",ha='right',va='bottom',fontsize=9)
    cbax=fig.add_axes([.098,.058,.32,.013])
    cb=fig.colorbar(m,cax=cbax,orientation='horizontal',ticks=[0,20,40,60,70])
    cb.set_label('Retrieved COT',fontsize=8.5,labelpad=2)
    cb.ax.tick_params(labelsize=8,pad=2,length=2);cb.outline.set_linewidth(.4)
    handles,labels=ghi_axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower right',bbox_to_anchor=(.99,.02),ncol=3,frameon=False,fontsize=8.4,handlelength=1.8,columnspacing=.9)
    for ext in ('pdf','png','svg'):fig.savefig(OUT/f'current_observed_events.{ext}',dpi=400,facecolor='white')
    for ga in ghi_axes:ga.set_ylim(0,550);ga.set_yticks([0,250,500])
    for ext in ('pdf','png'):fig.savefig(OUT/f'current_observed_events_common_ghi_axis.{ext}',dpi=400,facecolor='white')
    plt.close(fig)

sources=['event_trajectories.csv','selected_snapshot_fields.npz','EVENT_SELECTION_AND_PROVENANCE.json']
(OUT/'events_checks.json').write_text(json.dumps({
    'source_hashes':{f:hashlib.sha256((SOURCE/f).read_bytes()).hexdigest() for f in sources},
    'selected_dates_unchanged':[c['date'] for c in audit['cases']],
    'snapshots':checks,'source_trajectories_unchanged':True,
    'RGB_transform':'sqrt(clip(C03/C02/C01,0,1)) unchanged; nearest native pixels',
    'COT_limits':[0,70], 'GHI_limits':[[0,150],[0,500]],
    'missing_GHI_rows':[int(df.loc[df.date==c['date'],'truth'].isna().sum()) for c in audit['cases']],
    'CPP_reference':'unconnected markers',
    'layout_change':'dates are rows; images left, COT/GHI trajectories right; common colourbar',
    'test_read':False,'training':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Saved current events in PDF, PNG and editable SVG; sources unchanged.')
