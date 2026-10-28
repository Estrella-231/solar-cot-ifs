"""Render saved paired GHI scores and actual COT/GHI trajectories."""
from pathlib import Path
import csv,json
from datetime import datetime,timedelta
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).parent/'attempt1';OUT=BASE/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
g=json.loads((BASE/'ghi_scores_fp64.json').read_text());c=json.loads((BASE/'cot_scores_fp64.json').read_text())
fig,ax=plt.subplots(figsize=(6.7,4.4),layout='constrained')
before=[g['comparisons'][m+'_add_COT']['before_macro_rmse_Wm2'] for m in ('simvp','dydiff')]
after=[g['comparisons'][m+'_add_COT']['after_macro_rmse_Wm2'] for m in ('simvp','dydiff')]
x=np.arange(2);width=.32
for off,vals,col,label in [(-width/2,before,'#607D8B','Without COT'),(width/2,after,'#00897B','With predicted COT')]:
 bars=ax.bar(x+off,vals,width,color=col,label=label)
 for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+1,f'{v:.2f}',ha='center',va='bottom')
ax.set_xticks(x,['Hunan SimVP','Hubei DyDiff on Hunan']);ax.set_ylim(0,155);ax.set_ylabel('GHI RMSE (W m$^{-2}$)');ax.legend(frameon=False,loc='upper right')
ax.set_title('Paired forecast replay on selected Hunan validation windows',fontsize=11)
ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.supxlabel('432 records / 15 target dates / 2 stations / 3 fixed head seeds\nSame-capacity frozen heads; no forecast-input retraining',fontsize=8)
for ext in ('png','pdf'):fig.savefig(OUT/f'paired_ghi_cot_gain.{ext}',bbox_inches='tight')
plt.close(fig)

fig,ax=plt.subplots(figsize=(6.4,4.3),layout='constrained');x=np.arange(2)
for j,(method,label,color) in enumerate([('persistence','Historical COT persistence','#9E9E9E'),('simvp','SimVP to R','#4979A7'),('dydiff','DyDiff to R','#E27D32')]):
 vals=[c['CPP_reference']['stations'][s][method]['rmse'] for s in ('sili','zhujia')]
 bars=ax.bar(x+(j-1)*.23,vals,.23,label=label,color=color)
 for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+.15,f'{v:.2f}',ha='center',va='bottom',fontsize=9)
ax.set(xticks=x,xticklabels=['Sili','Zhujia'],ylabel='Station-pixel COT RMSE vs CPP reference',ylim=(0,17.3))
ax.legend(frameon=False,fontsize=9,loc='upper right');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
ax.set_title('COT accuracy does not improve uniformly across stations',fontsize=11)
fig.supxlabel('80 common station-pixel targets per station / 14 CPP-supported dates\nSame corrected Hunan R; CPP is a satellite retrieval reference',fontsize=8)
for ext in ('png','pdf'):fig.savefig(OUT/f'cot_cpp_station_comparison.{ext}',bbox_inches='tight')
plt.close(fig)

cr=list(csv.DictReader((BASE/'cot_rows.csv').open()));gr=list(csv.DictReader((BASE/'ghi_rows.csv').open()))
with np.load(BASE/'cot_predictions.npz') as z:cot={k:z[k].copy() for k in z.files}
with np.load(BASE/'ghi_predictions.npz') as z:ghi={k:z[k].copy() for k in z.files}
for case in (1879,1146):
 fig,axes=plt.subplots(2,2,figsize=(11,6.5),layout='constrained')
 for si,station in enumerate(('sili','zhujia')):
  ids=np.array([i for i,r in enumerate(cr) if int(r['window'])==case and r['station']==station]);leads=np.array([int(cr[i]['lead_minutes']) for i in ids])
  ax=axes[0,si]
  for method,label,color,style in [('observed','Observed AGRI to R (offline)','#303030','-'),('dydiff','DyDiff to R','#D97732','-'),('simvp','SimVP to R','#377BA8','-'),('persistence','Historical COT persistence','#969696','--')]:
   ax.plot(leads,np.expm1(cot['cot_'+method][ids,8,8].astype(np.float64)),color=color,ls=style,lw=1.7,label=label)
  mask=cot['cpp_mask'][ids,8,8].astype(bool)
  ax.scatter(leads[mask],cot['cpp'][ids,8,8][mask],marker='x',s=35,color='#9A5599',label='CPP reference',zorder=4)
  ax.set_title(station.capitalize()+' : station-pixel COT');ax.set_ylabel('COT');ax.grid(alpha=.15)
  if si==0:ax.legend(frameon=False,fontsize=7)
  ax=axes[1,si];gi=np.array([i for i,r in enumerate(gr) if int(r['window'])==case and r['station']==station])
  if not len(gi):
   ax.text(.5,.5,'No matching frozen GHI validation labels',ha='center',va='center',transform=ax.transAxes)
  else:
   gl=[int(gr[i]['lead_minutes']) for i in gi]
   ax.plot(gl,ghi['truth'][gi],color='#303030',lw=2,marker='o',ms=3,label='Measured 15-min mean GHI')
   for arm,label,col in [('M00','DyDiff, without COT','#607D8B'),('M10','DyDiff, with predicted COT','#00897B')]:
    ax.plot(gl,ghi[f'{station}_seed42_{arm}_dydiff'][gi],color=col,lw=1.8,label=label)
   if si==1:ax.legend(frameon=False,fontsize=8)
  ax.set_ylabel('GHI (W m$^{-2}$)');ax.set_xlabel('Forecast lead from initialization (min)');ax.grid(alpha=.15)
  for aa in axes[:,si]:aa.set_xticks([15,60,120,180,240])
 init=datetime.fromisoformat(cr[ids[0]]['init_time_utc'])+timedelta(hours=8)
 fig.suptitle(f'COT and GHI forecast trajectories | init {init:%Y-%m-%d %H:%M} BJT',fontsize=12)
 fig.supxlabel('Fixed seasonal example; GHI curves show seed42 only. Reported scores retain all three seeds.',fontsize=8)
 for ext in ('png','pdf'):fig.savefig(OUT/f'cot_ghi_trajectory_{case:05d}.{ext}',bbox_inches='tight')
 plt.close(fig)
print('SAVED',OUT)
