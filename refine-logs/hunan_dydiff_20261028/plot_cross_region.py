"""Saved-prediction Hubei-on-Hunan diagnostics. No inference or sample filtering."""
import csv,json
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).parent/'cross_hubei_on_hunan'
out=ROOT/'figures';out.mkdir(exist_ok=True)
m=json.loads((ROOT/'metrics.json').read_text())
selection=json.loads((ROOT/'selection.json').read_text())
rows=dict(zip(selection['indices'],selection['rows']))
rmse=np.array(m['rmse_physical_lead_channel'])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
with (ROOT/'hourly_metrics.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.writer(f);w.writerow(['lead_minutes','C02_reflectance_RMSE','C12_brightness_temperature_RMSE_K','validation_windows'])
 for i in (3,7,11,15):w.writerow([(i+1)*15,rmse[i,1],rmse[i,9],m['windows']])
fig,axes=plt.subplots(1,2,figsize=(9.2,3.6),layout='constrained')
for ax,k,label in zip(axes,(1,9),('C02 reflectance RMSE','C12 brightness-temperature RMSE (K)')):
 ax.plot(m['lead_minutes'],rmse[:,k],color='#2471A3',marker='o',markersize=3,lw=1.8)
 ax.set(xlabel='Forecast lead (min)',ylabel=label,xticks=[15,60,120,180,240]);ax.grid(alpha=.2)
fig.suptitle('Frozen Hubei DyDiff on Hunan: 16 validation windows',fontsize=12)
for ext in ('png','pdf'):fig.savefig(out/f'hubei_on_hunan_lead_rmse.{ext}',bbox_inches='tight')
plt.close(fig)

# Predeclared seasonal examples, not chosen by prediction performance.
for idx in (1879,1146):
 p=ROOT/f'val_{idx:05d}_case.npz'
 with np.load(p) as z:truth=z['truth'];pred=z['pred'];mask=z['valid']
 init=datetime.fromisoformat(rows[idx]['BJT_start'])+timedelta(minutes=105)
 fig,axes=plt.subplots(4,4,figsize=(10,10.1),layout='constrained')
 for r,(array,ch,label,vmin,vmax,cmap) in enumerate([
  (truth,0,'Observed C02',0,1,'gray'),(pred,0,'Hubei DyDiff C02',0,1,'gray'),
  (truth,1,'Observed C12',200,310,'gray_r'),(pred,1,'Hubei DyDiff C12',200,310,'gray_r')]):
  for col,k in enumerate((3,7,11,15)):
   ax=axes[r,col];values=np.ma.array(array[k,ch],mask=~mask[k,ch])
   cm=plt.get_cmap(cmap).copy();cm.set_bad('#e8dbcb')
   im=ax.imshow(values,vmin=vmin,vmax=vmax,cmap=cm,interpolation='nearest')
   ax.set_xticks([]);ax.set_yticks([])
   if r==0:ax.set_title(f'+{(k+1)*15} min | {(init+timedelta(minutes=(k+1)*15)):%H:%M}')
   if col==0:ax.set_ylabel(label,fontsize=10)
  if r in (1,3):
   cb=fig.colorbar(im,ax=axes[r-1:r+1,:].ravel().tolist(),fraction=.025,pad=.015)
   cb.set_label('Reflectance' if ch==0 else 'Brightness temperature (K)')
 fig.suptitle(f'Hunan cloud forecast from frozen Hubei DyDiff\nInitialization: {init:%Y-%m-%d %H:%M} BJT',fontsize=13)
 fig.supxlabel('Same Hunan grid and reference-valid pixels; beige = excluded support. No spatial averaging.',fontsize=9)
 for ext in ('png','pdf'):fig.savefig(out/f'hubei_on_hunan_case_{idx:05d}.{ext}',bbox_inches='tight')
 plt.close(fig)
print('SAVED',out)
