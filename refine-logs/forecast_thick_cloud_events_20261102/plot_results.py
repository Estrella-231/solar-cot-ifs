from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from PIL import Image
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.backends.backend_pdf import PdfPages
import analyze

P=Path(__file__).resolve().parent;F=P/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.6,'savefig.facecolor':'white'})
BLUE='#0072B2';ORANGE='#D55E00';GREEN='#009E73';DARK='#202830'
records=json.loads((P/'RESULTS.json').read_text());d=analyze.load();cases=json.loads((P/'CASE_SELECTION.json').read_text())['cases'];overview=[r for r in records['metrics'] if r['scope']=='all' and r['stratum'] in analyze.BINS]

def save(fig,name):
 png=F/(name+'.png');fig.savefig(png,dpi=280)
 with Image.open(png) as raster:
  if raster.mode=='RGBA':assert raster.getchannel('A').getextrema()==(255,255)
  raster.convert('RGB').save(png,dpi=(280,280))
 fig.savefig(F/(name+'.pdf'));fig.savefig(F/(name+'.svg'));return fig

fig,axs=plt.subplots(2,2,figsize=(7.1,5.2),gridspec_kw={'height_ratios':[1,1.15]},layout='constrained')
labels=['Near-zero','Thin','Moderate','Thick']
for ci,reg in enumerate(['Hubei','Hunan']):
 r=[next(q for q in overview if q['region']==reg and q['stratum']==s) for s in labels];x=np.arange(4)
 a=axs[0,ci];a.plot(x,[q['A_rmse'] for q in r],'o--',color=BLUE,lw=1.15,ms=4,label='Without COT');a.plot(x,[q['AC_rmse'] for q in r],'s-',color=ORANGE,lw=1.35,ms=4,label='With forecast COT');a.set_title(f'({chr(97+ci)}) {reg}: GHI error',loc='left',weight='bold',fontsize=9);a.set_ylim(0,205);a.set_ylabel('GHI RMSE (W m$^{-2}$)');a.legend(frameon=False,fontsize=7,loc='lower left')
 a=axs[1,ci]
 for j,q in enumerate(r):
  lo,hi=q['delta_ci95'];v=q['delta_rmse'];a.errorbar(j,v,yerr=[[v-lo],[hi-v]],fmt='o',color=ORANGE if j==3 else BLUE,capsize=3,lw=1,ms=4)
  a.text(j,.94,f"{q['n_station_targets']} targets\n{q['n_dates']} days",transform=a.get_xaxis_transform(),ha='center',va='top',fontsize=6.8,color=DARK)
 a.axhline(0,lw=.8,color='.4');a.set_ylim(-40,45);a.axhspan(-40,0,color=GREEN,alpha=.045);a.set_title(f'({chr(99+ci)}) Paired COT effect',loc='left',weight='bold',fontsize=9);a.set_ylabel('With COT − without COT\nRMSE (W m$^{-2}$)')
 for a in axs[:,ci]:a.set_xticks(x,labels,rotation=12);a.grid(axis='y',alpha=.15);a.set_xlim(-.45,3.45)
fig.suptitle('Forecast GHI error by reference cloud optical thickness',fontsize=11,weight='bold');save(fig,'forecast_cpp_strata');plt.close(fig)

# Event diagnostic: a point is a station CPP reference segment, not an independent cloud.
e=pd.read_csv(P/'events.csv');fig,axs=plt.subplots(1,2,figsize=(7.1,3.1),layout='constrained')
for a,reg in zip(axs,['Hubei','Hunan']):
 q=e[e.region.eq(reg)]
 for mult,marker,label,col in [(False,'x','Single reference time','#8A929A'),(True,'o','Consecutive reference times',BLUE)]:
  ss=q[q.n_targets.ge(2).eq(mult)];a.scatter(ss.peak_cpp,ss.delta_rmse_target_balanced,s=16,marker=marker,label=label,color=col,alpha=.8,linewidths=.6)
 a.axhline(0,color='.4',lw=.8);a.set(xlabel='Peak station-pixel CPP COT',ylabel='Segment ΔRMSE (W m$^{-2}$)',title=reg);a.grid(alpha=.12);a.legend(frameon=False,fontsize=6.5)
fig.suptitle('Reference segments: improvement and degradation both occur',fontsize=10,weight='bold');save(fig,'reference_segment_effects');plt.close(fig)

def rgb(z):
 im=np.moveaxis(z[[2,1,0]],0,-1);valid=np.isfinite(im).all(-1);im=np.sqrt(np.clip(np.nan_to_num(im),0,1));im[~valid]=.78;return im

def image_panel(ax,arr,center,isrgb=False,native=16):
 extent=(-.5,15.5,15.5,-.5) if native==16 else (6.5,9.5,9.5,6.5)
 if isrgb:im=ax.imshow(rgb(arr),interpolation='nearest',extent=extent)
 else:im=ax.imshow(arr,cmap=plt.colormaps['viridis'].with_extremes(bad='#D9DDE1'),norm=Normalize(0,80),interpolation='nearest',extent=extent)
 y,x=center
 if native==3:y=x=8
 ax.plot(x,y,marker='+',mec='black',mew=2,ms=7);ax.plot(x,y,marker='+',mec='white',mew=.8,ms=7)
 if native==16:ax.add_patch(Rectangle((x-1.5,y-1.5),3,3,fill=False,edgecolor='white',ls='--',lw=.7))
 for sp in ax.spines.values():sp.set_visible(True);sp.set_linewidth(.4);sp.set_color('#A9B0B7')
 ax.set_xticks([0,8,15] if native==16 else [7,8,9]);ax.set_yticks([]);ax.tick_params(axis='x',labelsize=5.5,length=2,pad=1)
 return im

checks=[];case_metrics=[]
with PdfPages(F/'four_real_cloud_cases.pdf') as pdf:
 for num,c in enumerate(cases,1):
  z=np.load(P/'cases'/(c['event_id']+'.npz'));times=pd.to_datetime(z['times_bjt'],format='%Y%m%d%H%M');center=z['center'];fc=z['forecast_center'];native=z['forecast_agri'].shape[-1]
  q=d[d.region.eq(c['region'])&d.station.eq(c['station'])&d.init.eq(pd.Timestamp(c['init']))].sort_values('target').set_index('target').reindex(times)
  # Exact saved-COT agreement at all available GHI endpoints.
  cseries=z['forecast_cot'][:,fc[0],fc[1]];valid=q.cot.notna().to_numpy();difference=float(np.nanmax(abs(q.cot.to_numpy()[valid]-cseries[valid])));assert difference<1e-4
  cpp=np.where(z['cpp_mask'],z['cpp'],np.nan);cppseries=cpp[:,center[0],center[1]];valid=q.cpp.notna().to_numpy();refdiff=float(np.nanmax(abs(q.cpp.to_numpy()[valid]-cppseries[valid])));assert refdiff<1e-4
  selected=[int(np.where(times==pd.Timestamp(t))[0][0]) for t in c['display_times']]
  assert len(selected)==3
  fig=plt.figure(figsize=(7.1,9.55));gs=fig.add_gridspec(8,4,width_ratios=[1,1,1,.055],height_ratios=[1,1,1,1,1,.10,1.22,1.45],left=.18,right=.92,top=.915,bottom=.125,hspace=.28,wspace=.13)
  title=f"{c['region']} · {c['station'].capitalize()} · {pd.Timestamp(c['init']):%d %b %Y}"
  fig.text(.035,.977,title,weight='bold',fontsize=12,color=DARK,va='top');fig.text(.035,.949,f"{c['direction'].capitalize()} reference segment  |  initialized {pd.Timestamp(c['init']):%H:%M} BJT",fontsize=8.5,color='.28')
  names=[f'Observed AGRI\n16 × 16',f'Forecast AGRI\n{native} × {native}'+('\narchived crop' if native==3 else ''),'CPP reference\n16 × 16','R(observed AGRI)\n16 × 16',f'R(forecast AGRI)\n{native} × {native}']
  for row in range(5):
   for col,k in enumerate(selected):
    a=fig.add_subplot(gs[row,col]);source=[z['observed_agri'],z['forecast_agri'],cpp,z['observed_cot'],z['forecast_cot']][row];im=image_panel(a,source[k],center,row<2,native if row in (1,4) else 16)
    if row==0:a.set_title(f"{times[k]:%H:%M}  (+{int((times[k]-pd.Timestamp(c['init'])).total_seconds()/60)} min)",fontsize=8,pad=5)
    if col==0:fig.text(.014,a.get_position().y0+a.get_position().height/2,names[row],va='center',fontsize=7.7,color=DARK)
  ca=fig.add_subplot(gs[2:5,3]);cb=fig.colorbar(im,cax=ca,extend='max',ticks=[0,30,60,80]);cb.set_label('Cloud optical thickness',fontsize=7,labelpad=5);cb.ax.tick_params(labelsize=7)
  cotax=fig.add_subplot(gs[6,:3]);ghiax=fig.add_subplot(gs[7,:3],sharex=cotax)
  cotax.plot(times,cppseries,'ko-',ms=3,lw=.85,label='CPP reference');cotax.plot(times,z['observed_cot'][:,center[0],center[1]],color=GREEN,lw=1.05,ls=':',label='R(observed AGRI)');cotax.plot(times,cseries,color=ORANGE,lw=1.25,marker='s',ms=2.5,label='R(forecast AGRI)');cotax.axhline(30,color='.45',lw=.65,ls='--');cotax.set_ylabel('Station COT');cotax.set_ylim(bottom=0);cotax.legend(frameon=False,ncol=3,fontsize=6,loc='upper right',bbox_to_anchor=(1,1.30));cotax.tick_params(labelbottom=False)
  ghiax.plot(times,q.truth,'o-',color=DARK,lw=1.25,ms=3,label='Observed');ghiax.plot(times,q.A42,'--',color=BLUE,lw=1.15,label='Without COT');ghiax.plot(times,q.AC42,'-',color=ORANGE,lw=1.35,label='With forecast COT');ghiax.set_ylabel('15-min GHI\n(W m$^{-2}$)');ghiax.set_xlabel('Target interval end (BJT)');ghiax.set_ylim(bottom=0);ghiax.legend(frameon=False,ncol=3,fontsize=6.5,loc='upper left',bbox_to_anchor=(0,1.19))
  for ax in [cotax,ghiax]:
   ax.axvspan(pd.Timestamp(c['start']),pd.Timestamp(c['end']),color='#E9C46A',alpha=.18,zorder=-2);ax.grid(axis='y',alpha=.15);ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0,30]));ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'));ax.tick_params(labelsize=6.8)
  m=q.index.to_series().between(pd.Timestamp(c['start']),pd.Timestamp(c['end'])).to_numpy()&q.truth.notna().to_numpy();a=float(np.sqrt(np.mean((q.A42.to_numpy()[m]-q.truth.to_numpy()[m])**2)));b=float(np.sqrt(np.mean((q.AC42.to_numpy()[m]-q.truth.to_numpy()[m])**2)))
  fig.text(.92,.052,f'Shown origin, seed 42, thick-target RMSE: {a:.1f} → {b:.1f} W m$^{{-2}}$',ha='right',va='bottom',fontsize=6.5,color=DARK)
  fig.text(.035,.018,'Native pixels; RGB = C03/C02/C01, identical square-root display. Grey = missing reference.\nShading: selected CPP-thick endpoint segment. Future observations are offline diagnostics; GHI lines: seed 42.',fontsize=6.5,color='.35',va='bottom')
  save(fig,f'case_{num:02d}_{c["event_id"]}');pdf.savefig(fig);plt.close(fig)
  q.assign(cpp_exported=cppseries,forecast_cot_exported=cseries).to_csv(P/'cases'/(c['event_id']+'_timeseries.csv'),index_label='target_time_bjt')
  checks.append({'case':c['event_id'],'forecast_native_grid':native,'saved_forecast_cot_max_diff':difference,'cpp_center_max_diff':refdiff,'display_times':c['display_times'],'data_sha256':hashlib.sha256((P/'cases'/(c['event_id']+'.npz')).read_bytes()).hexdigest()});case_metrics.append({'event_id':c['event_id'],'region':c['region'],'station':c['station'],'start':c['start'],'end':c['end'],'init':c['init'],'direction':c['direction'],'all_seed_event_target_balanced_delta':c['delta_rmse_target_balanced'],'shown_seed42_origin_thick_A':a,'shown_seed42_origin_thick_AC':b})
pd.DataFrame(case_metrics).to_csv(P/'case_display_metrics.csv',index=False)
(P/'CASE_SOURCE_CHECK.json').write_text(json.dumps({'state':'PASS','checks':checks,'no_synthetic_images':True,'no_prediction_upsampling':'Hunan exact archived 3x3 crop shown at native grid; Hubei saved 16x16','no_new_model_inference':True},indent=2))
(P/'FIGURE_PROVENANCE.json').write_text(json.dumps({'intended_width_mm':180,'journal':'not selected; general manuscript preview','exports':['PDF','PNG 280 dpi','SVG'],'RGB':'C03/C02/C01, fixed sqrt(clip(reflectance,0,1)); native nearest-neighbour pixels','COT':'common [0,80] scale; upper triangle for >80; missing grey','case_GHI':'seed42 lines; primary tables are equal station/3-seed scores','skills_citation':'Kassis T., Agarwal V., He Y., Patel D., Brueckner A.M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. arXiv:2609.00065, latest record v2 checked 2026-11-02. https://doi.org/10.48550/arXiv.2609.00065'},indent=2))
print('Rendered overview, segment diagnostic and four real-data case figures.')
