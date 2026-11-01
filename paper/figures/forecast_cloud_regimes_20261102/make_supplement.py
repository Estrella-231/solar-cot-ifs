# Preserves source data; presentation-only revision of full-case panels.
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

P=Path(__file__).resolve().parents[3]/"refine-logs/forecast_thick_cloud_events_20261102"
F=Path(__file__).resolve().parent/"supplement"
F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.6,'savefig.facecolor':'white'})
BLUE='#2878A5';ORANGE='#D87836';GREEN='#8064A2';DARK='#202830'
cases=json.loads((P/'CASE_SELECTION.json').read_text())['cases']

def save(fig,name):
 png=F/(name+'.png');fig.savefig(png,dpi=280)
 with Image.open(png) as raster:
  if raster.mode=='RGBA':assert raster.getchannel('A').getextrema()==(255,255)
  raster.convert('RGB').save(png,dpi=(280,280))
 fig.savefig(F/(name+'.pdf'));fig.savefig(F/(name+'.svg'));return fig

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
  q=pd.read_csv(P/'cases'/(c['event_id']+'_timeseries.csv'));q.index=pd.to_datetime(q.pop('target_time_bjt'));q=q.reindex(times)
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
  cotax.plot(times,cppseries,'ko-',ms=3,lw=.85,label='CPP reference');cotax.plot(times,z['observed_cot'][:,center[0],center[1]],color=GREEN,lw=1.05,ls=':',marker='o',ms=1.8,label='R(observed AGRI)');cotax.plot(times,cseries,color=ORANGE,lw=1.25,marker='s',ms=2.5,label='R(forecast AGRI)');cotax.axhline(30,color='.45',lw=.65,ls='--');cotax.set_ylabel('Station COT');cotax.set_ylim(bottom=0);cotax.legend(frameon=False,ncol=3,fontsize=6,loc='upper right',bbox_to_anchor=(1,1.30));cotax.tick_params(labelbottom=False)
  ghiax.plot(times,q.truth,'o-',color=DARK,lw=1.25,ms=3,label='Observed');ghiax.plot(times,q.A42,'--',color=BLUE,lw=1.15,marker='^',mfc='white',ms=3,label='Without COT');ghiax.plot(times,q.AC42,'-',color=ORANGE,lw=1.35,marker='s',ms=2.7,label='With forecast COT');ghiax.set_ylabel('15-min GHI\n(W m$^{-2}$)');ghiax.set_xlabel('Target interval end (BJT)');ghiax.set_ylim(bottom=0);ghiax.legend(frameon=False,ncol=3,fontsize=6.5,loc='upper left',bbox_to_anchor=(0,1.19))
  for ax in [cotax,ghiax]:
   ax.axvspan(pd.Timestamp(c['start']),pd.Timestamp(c['end']),color='#E9C46A',alpha=.18,zorder=-2);ax.grid(axis='y',alpha=.15);ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0,30]));ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'));ax.tick_params(labelsize=6.8)
  m=q.index.to_series().between(pd.Timestamp(c['start']),pd.Timestamp(c['end'])).to_numpy()&q.truth.notna().to_numpy();a=float(np.sqrt(np.mean((q.A42.to_numpy()[m]-q.truth.to_numpy()[m])**2)));b=float(np.sqrt(np.mean((q.AC42.to_numpy()[m]-q.truth.to_numpy()[m])**2)))
  fig.text(.92,.052,f'Shown origin, seed 42, thick-target RMSE: {a:.1f} → {b:.1f} W m$^{{-2}}$',ha='right',va='bottom',fontsize=6.5,color=DARK)
  fig.text(.035,.018,'Native pixels; RGB = C03/C02/C01, identical square-root display. Grey = missing reference.\nShading: selected CPP-thick endpoint segment. Future observations are offline diagnostics; GHI lines: seed 42.',fontsize=6.5,color='.35',va='bottom')
  save(fig,f'case_{num:02d}_{c["event_id"]}');pdf.savefig(fig);plt.close(fig)
  # Source arrays and case CSVs are read-only; markers retain isolated finite points.
  checks.append({'case':c['event_id'],'forecast_native_grid':native,'saved_forecast_cot_max_diff':difference,'cpp_center_max_diff':refdiff,'display_times':c['display_times'],'data_sha256':hashlib.sha256((P/'cases'/(c['event_id']+'.npz')).read_bytes()).hexdigest()});case_metrics.append({'event_id':c['event_id'],'region':c['region'],'station':c['station'],'start':c['start'],'end':c['end'],'init':c['init'],'direction':c['direction'],'all_seed_event_target_balanced_delta':c['delta_rmse_target_balanced'],'shown_seed42_origin_thick_A':a,'shown_seed42_origin_thick_AC':b})
pd.DataFrame(case_metrics).to_csv(F/'case_display_metrics.csv',index=False)
(F/'CASE_SOURCE_CHECK.json').write_text(json.dumps({'state':'PASS','checks':checks,'no_synthetic_images':True,'no_prediction_upsampling':'Hunan exact archived 3x3 crop shown at native grid; Hubei saved 16x16','no_new_model_inference':True},indent=2))
(F/'FIGURE_PROVENANCE.json').write_text(json.dumps({'intended_width_mm':180,'journal':'not selected; general manuscript preview','exports':['PDF','PNG 280 dpi','SVG'],'RGB':'C03/C02/C01, fixed sqrt(clip(reflectance,0,1)); native nearest-neighbour pixels','COT':'common [0,80] scale; upper triangle for >80; missing grey','case_GHI':'seed42 lines; primary tables are equal station/3-seed scores','skills_citation':'Kassis T., Agarwal V., He Y., Patel D., Brueckner A.M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. arXiv:2609.00065, latest record v2 checked 2026-11-02. https://doi.org/10.48550/arXiv.2609.00065'},indent=2))
print('Rendered four supplement cases with all finite isolated predictions visible.')
