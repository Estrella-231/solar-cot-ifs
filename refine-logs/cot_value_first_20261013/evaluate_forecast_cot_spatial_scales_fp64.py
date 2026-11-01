import datetime as dt
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'refine-logs/cot_value_first_20261013/results/forecast_cot_val_bank_20261014_v1'
z=np.load(OUT/'forecast_cot_val_bank.npz',allow_pickle=False)
profile=json.loads((OUT/'profile.json').read_text(encoding='utf-8'))
assert profile['split']=='val_only' and profile['test_payloads_read']==0
mask=z['cpp_valid_mask'].astype(bool); ref=z['cpp_reference_cot'].astype(np.float64)
names=('SimVP→R','Persistence→R','Offline true-future AGRI→R')
logs=('forecast_log1p_cot','persistence_log1p_cot','target_agri_r_log1p_cot')
pred=[np.expm1(z[k].astype(np.float64)) for k in logs]
regions={'station_pixel_1x1':(slice(8,9),slice(8,9)),
         'station_neighbourhood_3x3':(slice(7,10),slice(7,10)),
         'full_station_patch_16x16':(slice(0,16),slice(0,16))}
stations=('sili','zhujia'); times=z['target_time_utc'].astype(str)

def metrics(a,b,m):
    e=a[m]-b[m]
    if not e.size:return {'n':0,'rmse':None,'mae':None,'bias':None}
    return {'n':int(e.size),'rmse':float(np.sqrt(np.mean(e*e,dtype=np.float64))),
            'mae':float(np.mean(np.abs(e),dtype=np.float64)), 'bias':float(np.mean(e,dtype=np.float64))}

summary={}; lead_rows=[]
for rname,(rs,cs) in regions.items():
    mm=mask[:,:, :,rs,cs]
    yy=ref[:,:,:,rs,cs]
    pp=[x[:,:,:,rs,cs] for x in pred]
    by_station=[]
    for si,s in enumerate(stations):
        sm=mm[:,si]
        row={'station':s,'common_cpp_pixels':int(sm.sum()),'windows_with_support':int(sm.any(axis=(1,2,3)).sum())}
        for name,x in zip(names,pp):row[name]=metrics(x[:,si],yy[:,si],sm)
        by_station.append(row)
        for li in range(16):
            m=mm[:,si,li]
            for name,x in zip(names,pp):
                v=metrics(x[:,si,li],yy[:,si,li],m)
                lead_rows.append({'region':rname,'station':s,'lead_min':15*(li+1),'method':name,**v})
    summary[rname]={'station_metrics':by_station,'station_equal_rmse':{
        name:float(np.mean([x[name]['rmse'] for x in by_station])) for name in names}}

# Paired two-day BJT date-block bootstrap, same RMSE-difference estimate per station.
dates=np.empty((len(times),16),dtype=object)
for i in range(len(times)):
    for li in range(16):dates[i,li]=(dt.datetime.strptime(times[i,li],'%Y-%m-%d %H:%M:%S')+dt.timedelta(hours=8)).date().isoformat()
ud=sorted(set(dates.ravel())); ordinal={d:dt.date.fromisoformat(d).toordinal() for d in ud}
block={d:(ordinal[d]-ordinal[ud[0]])//2 for d in ud}; blocks=sorted(set(block.values()))
rng=np.random.default_rng(20261015); B=2000
for rname,(rs,cs) in regions.items():
    mm=mask[:,:,:,rs,cs]; yy=ref[:,:,:,rs,cs]; pp=[x[:,:,:,rs,cs] for x in pred]
    # Store each (date-block,station,method) squared-error sum and pixel count.
    sse=np.zeros((len(blocks),2,3),np.float64); count=np.zeros((len(blocks),2),np.int64)
    for bi,b in enumerate(blocks):
      for i in range(len(times)):
       for li in range(16):
        if block[dates[i,li]]!=b:continue
        for si in range(2):
         m=mm[i,si,li]
         if not m.any():continue
         count[bi,si]+=int(m.sum())
         for mi,x in enumerate(pp):
          e=x[i,si,li][m]-yy[i,si,li][m];sse[bi,si,mi]+=np.sum(e*e,dtype=np.float64)
    diffs=np.empty((B,2),np.float64)
    for bidx in range(B):
      chosen=rng.integers(0,len(blocks),size=len(blocks)); ns=count[chosen].sum(axis=0)
      rms=np.sqrt(sse[chosen].sum(axis=0)/ns[:,None])
      diffs[bidx,0]=np.mean(rms[:,0]-rms[:,2])
      diffs[bidx,1]=np.mean(rms[:,0]-rms[:,1])
    summary[rname]['paired_2day_block_CI']={
      'forecast_minus_offline_true_future_R':[float(x) for x in np.quantile(diffs[:,0],[.025,.975])],
      'forecast_minus_persistence':[float(x) for x in np.quantile(diffs[:,1],[.025,.975])],
      'replicates':B,'blocks':len(blocks),'unique_target_bjt_dates':len(ud)}

result={'state':'FP64_INDEPENDENT_SPATIAL_SCALE_RECOMPUTE_PASS','split':'validation_only','test_payloads_read':0,
        'mask':'same CPP-valid pixel support for all methods; all predictions finite',
        'patch_geometry':'station pixel=(8,8), centered 3x3=(7:10,7:10), full=16x16',
        'methods':list(names),'lead_minutes':[15*(i+1) for i in range(16)],'regions':summary,'lead_metrics':lead_rows,
        'note':'CPP is a retrieval reference, not independent COT truth. Offline target AGRI→R is not deployable. The 271 windows overlap and the supported dates are limited.'}
(OUT/'forecast_cot_spatial_scales_fp64.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
with (OUT/'forecast_cot_spatial_scales_lead_fp64.csv').open('w',newline='',encoding='utf-8') as f:
    import csv
    w=csv.DictWriter(f,fieldnames=list(lead_rows[0]));w.writeheader();w.writerows(lead_rows)

fig,axs=plt.subplots(1,3,figsize=(15,4.8),sharey=True)
cols={'sili':'#2563eb','zhujia':'#ea580c'}; styles={names[0]:'-',names[1]:'--',names[2]:':'}
for ax,(rname,_) in zip(axs,regions.items()):
  for s in stations:
    for name in names:
      rr=[x for x in lead_rows if x['region']==rname and x['station']==s and x['method']==name]
      ax.plot([x['lead_min'] for x in rr],[x['rmse'] for x in rr],color=cols[s],linestyle=styles[name],marker='o' if name==names[0] else None,ms=3,label=f'{s.upper()} · {name}')
  ax.set_title(rname.replace('_',' '));ax.set_xlabel('Lead (min)');ax.grid(alpha=.25)
axs[0].set_ylabel('COT RMSE vs CPP reference');axs[-1].legend(fontsize=7,loc='upper left',bbox_to_anchor=(1.02,1))
fig.suptitle('Forecast-chain COT error by station-centered spatial scale (common support)',y=1.02);fig.tight_layout()
out=OUT/'forecast_cot_rmse_spatial_scales_fp64.png';fig.savefig(out,dpi=180,bbox_inches='tight');plt.close(fig)
print(json.dumps({k:v['station_equal_rmse'] for k,v in summary.items()},indent=2))
for k,v in summary.items():print(k,'CI',v['paired_2day_block_CI'])
print(out)
