"""Current observed-three-frame paired GHI cases, with actual AGRI and R-COT."""
from pathlib import Path
import json,hashlib,csv
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize

P=Path(__file__).parent;ROOT=P.parents[2]
df=pd.read_csv(P/'current_validation_predictions.csv',dtype={'time_bjt':str,'date':str})
candidates=pd.read_csv(P/'event_candidates.csv',dtype={'date':str})
pool=candidates[(candidates.station=='Niushou') & candidates.eligible].sort_values('date')
chosen=[pool[pool.gain_pct>0].iloc[0],pool[pool.gain_pct<=0].iloc[0]]
images=np.load(P/'validation_target_images.npz')
lut={t:i for i,t in enumerate(images['times_bjt'])}
rv=ROOT/'refine-logs/hubei_same_time_geometry_20261102/r_validation'
rkeys=pd.read_csv(rv/'val_keys.csv',dtype={'time_bjt':str});rlut={t:i for i,t in enumerate(rkeys.time_bjt)}
ref=np.load(rv/'val_reference_cpp_cot.npy');rmask=np.load(rv/'val_valid_mask.npy')
BLUE='#0072B2';ORANGE='#D55E00';PURPLE='#6A3D9A'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,
    'axes.spines.right':False,'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6,
    'pdf.fonttype':42,'savefig.facecolor':'white'})
fig=plt.figure(figsize=(7.1,6.6))
outer=fig.add_gridspec(1,2,left=.075,right=.975,bottom=.115,top=.915,wspace=.20)
metadata=[];export=[];rawsave={};axes_ghi=[];axes_cot=[]
for col,rec in enumerate(chosen):
    date=rec.date;st=rec.station;y,x=8,6
    idx=pd.date_range(pd.Timestamp(date)+pd.Timedelta(hours=10,minutes=30),periods=17,freq='15min')
    key=idx.strftime('%Y%m%d%H%M');g=df[(df.station==st)&(df.date==date)].copy()
    g.index=pd.to_datetime(g.time_bjt,format='%Y%m%d%H%M');g=g.reindex(idx)
    # COT can remain observable when the GHI scoring row is missing; retain
    # its independently available imagery rather than imposing the label mask.
    g['COT']=[float(np.expm1(images['cot_log1p'][lut[k],y,x])) if k in lut else np.nan for k in key]
    gs=outer[col].subgridspec(5,3,height_ratios=[1.02,1.02,.12,1.12,1.75],hspace=.37,wspace=.045)
    letter='ab'[col]
    fig.text(.075+col*.492,.976,f'({letter}) Niushou · {idx[0]:%d %b %Y}',ha='left',va='top',fontsize=10,fontweight='bold')
    snaps=[0,8,16]
    cotmax=0
    for j,t in enumerate(snaps):
        k=key[t]
        if k not in lut:
            raise ValueError(f'selected nominal image time unavailable: {k}')
        raw=images['raw'][lut[k]];cot=np.expm1(images['cot_log1p'][lut[k]]).astype(float)
        rgb=np.sqrt(np.clip(raw[[2,1,0]].transpose(1,2,0),0,1))
        rawsave[f'{date}_{k[-4:]}_agri']=raw;rawsave[f'{date}_{k[-4:]}_cot']=cot
        cotmax=max(cotmax,float(cot.max()))
        for row,v in [(0,rgb),(1,cot)]:
            ax=fig.add_subplot(gs[row,j])
            if row==0:ax.imshow(v,interpolation='nearest')
            else:img=ax.imshow(v,vmin=0,vmax=70,cmap='magma',interpolation='nearest')
            ax.plot(x,y,'+',color='white',ms=5,mew=.85)
            ax.add_patch(Rectangle((x-1.5,y-1.5),3,3,fill=False,edgecolor='white',linestyle='--',linewidth=.7))
            ax.set_xticks([]);ax.set_yticks([])
            for sp in ax.spines.values():sp.set_visible(True);sp.set_linewidth(.4)
            if row==0:ax.set_title(idx[t].strftime('%H:%M'),fontsize=8,pad=4)
            if j==0:ax.set_ylabel('AGRI composite' if row==0 else 'Retrieved COT',fontsize=8,labelpad=5)
    cbax=fig.add_subplot(gs[2,:]);cb=fig.colorbar(img,cax=cbax,orientation='horizontal',ticks=[0,20,40,60,70]);cb.ax.tick_params(labelsize=7,pad=1,length=2);cb.outline.set_linewidth(.4)
    ca=fig.add_subplot(gs[3,:]);axes_cot.append(ca)
    ca.plot(idx,g.COT,color=PURPLE,lw=1.2,marker='o',ms=2.7,label='Retrieved COT')
    refv=[]
    for k in key:
        q=rlut.get(k)
        refv.append(float(ref[q,0,y,x]) if q is not None and bool(rmask[q,0,y,x]) else np.nan)
    # Do not connect sparse CPP reference times as though they were continuous.
    if np.isfinite(refv).any():ca.plot(idx,refv,linestyle='None',marker='x',ms=4,color='.35',label='CPP reference')
    ca.set_ylim(0,70);ca.set_ylabel('Station COT');ca.grid(axis='y',alpha=.17);ca.tick_params(axis='x',labelbottom=False)
    ca.legend(loc='lower left',fontsize=6.6,ncol=2,frameon=False,handlelength=1.4,borderpad=.1,columnspacing=1.)
    ga=fig.add_subplot(gs[4,:],sharex=ca);axes_ghi.append(ga)
    ga.plot(idx,g.truth,'o-',color='.15',lw=1.25,ms=3,label='Observed')
    ga.plot(idx,g.A,'s--',color=BLUE,lw=1.15,ms=2.5,label='Without COT')
    ga.plot(idx,g.AC,'^-',color=ORANGE,lw=1.15,ms=3,label='With COT')
    ga.set_ylabel('GHI (W m$^{-2}$)');ga.set_xlabel('Target interval end (BJT)');ga.grid(axis='y',alpha=.17)
    # Separate GHI scales expose the actual low-irradiance dynamics without an inset;
    # the caption states this explicitly. An additional common-axis version is saved.
    hi=np.ceil(max(g.truth.max(),g.A.max(),g.AC.max())/50)*50
    ga.set_ylim(0,hi);ga.xaxis.set_major_locator(mdates.HourLocator());ga.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ga.set_xlim(idx[0]-pd.Timedelta(minutes=5),idx[-1]+pd.Timedelta(minutes=5))
    ga.tick_params(axis='x',labelsize=7.3)
    ga.text(.03,.96,f'RMSE: {rec.A_rmse:.1f} → {rec.AC_rmse:.1f} W m$^{{-2}}$',ha='left',va='top',transform=ga.transAxes,fontsize=7.3,bbox={'facecolor':'white','alpha':.85,'edgecolor':'none','pad':1})
    g['CPP_COT']=refv;g['station']='Niushou';g['time_bjt']=key;g['date']=date
    export.append(g)
    metadata.append({'station':st,'date':date,'n':int(rec.n),'A_rmse_seedmean':float(rec.A_rmse),'AC_rmse_seedmean':float(rec.AC_rmse),'gain_pct':float(rec.gain_pct),'COT_min':float(rec.cot_min),'COT_max':float(rec.cot_max),'snapshot_COT_max':cotmax,'display_mean_A_bias':float((g.A-g.truth).mean()),'display_mean_AC_bias':float((g.AC-g.truth).mean()),'snapshots_bjt':[str(key[t]) for t in snaps],'missing_GHI_times':[str(key[i]) for i in range(len(g)) if not np.isfinite(g.truth.iloc[i])],'CPP_reference_points':int(np.isfinite(refv).sum())})
handles,labels=axes_ghi[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.53,.005),ncol=3,frameon=False,fontsize=8)
for ext in ['pdf','png']:fig.savefig(P/f'current_observed_events.{ext}',dpi=300)
for ga in axes_ghi:ga.set_ylim(0,550)
for ext in ['pdf','png']:fig.savefig(P/f'current_observed_events_common_ghi_axis.{ext}',dpi=300)
plt.close(fig)
pd.concat(export).to_csv(P/'event_trajectories.csv',index=False)
np.savez_compressed(P/'selected_snapshot_fields.npz',**rawsave)
audit={'cases':metadata,'candidate_rule':{'station':'Niushou, first station in fixed study order','time_bjt':'10:30–14:30, fixed 17-target grid','minimum_GHI_targets':12,'retrieved_station_COT_peak_min':30,'retrieved_station_COT_range_min':5,'selection':'earliest calendar date in each positive and non-positive paired-RMSE direction group; not extrema','scope':'illustrative conditional selection, not prevalence or independent weather replication','initial_completeness_attempt':'17/17 yielded no candidates; threshold relaxed to 12/17 based on missingness before direction-based selection'},'eligible_station_dates':pool.to_dict(orient='records'),'display_predictions':'mean of three independent seed predictions; displayed RMSE computed per seed then averaged, not ensemble RMSE','raw_model_route':'observed T-30/T-15/T AGRI → corrected fixed R → same-time paired GHI heads','RGB':'C03/C02/C01, square root of clip(reflectance,0,1), common mapping, native16x16 nearest-neighbour display','COT':'expm1(current frozen R log1p output); common0–70 scale; no pixel above70 in chosen snapshots','spatial':'Niushou centre(row8,col6); dashed3x3 model crop; Rcontext16x16','gaps':'15-min reindex with NaN, no gap bridging; sparse CPP reference markers not connected','normal_GHI_axes':'independent0–150 /0–500 Wm-2; alternative common0–550 export also saved','test_payloads_read':0,'training_performed':False}
(P/'EVENT_SELECTION_AND_PROVENANCE.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(audit['cases'],indent=2))
