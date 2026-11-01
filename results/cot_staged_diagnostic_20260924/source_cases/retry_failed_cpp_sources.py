import csv,json,time
from pathlib import Path
import numpy as np
from netCDF4 import Dataset
B=Path('/tmp/cpp_audit_20260924')
rows=list(csv.DictReader((B/'validation_source_manifest.csv').open())); idx={}
for order,r in enumerate(rows): r['order']=order; idx.setdefault(r['timestamp_utc'],[]).append(r)
errs=[x['timestamp_utc'] for x in csv.DictReader((B/'validation_cpp_source_audit_errors.csv').open())]
d=np.load(B/'validation_labels_for_cpp_audit.npz'); target=d['target_cot']; mask=d['mask'].astype(bool); grids=np.load(B/'station_grid_coords.npz')
starts={'sili':(1368,2201),'zhujia':(1278,2203)}
res=[];t=time.time()
for ts in errs:
 stamp=ts.replace('-','').replace(':','').replace('T','').replace('Z',''); day=stamp[:8]
 path=Path('/home/Data_Pool_3/data/Global_CPP/FY/FY4B/Result/CPP/2025')/day/f'FY4B_AGRI_{stamp}.nc'
 x={'timestamp_utc':ts,'path':str(path),'success':False,'error':None,'samples':[]}
 try:
  with Dataset(path,'r') as ds:
   lat=np.asarray(ds['LAT'][:]);lon=np.asarray(ds['LON'][:]); x['qa_present']='QA' in ds.variables
   for r in idx[ts]:
    st=r['station_id']; k=int(r['order']);r0,c0=starts[st]
    cot=np.ma.filled(ds['COT'][r0:r0+16,c0:c0+16],np.nan).astype(np.float32);clp=np.ma.filled(ds['CLP'][r0:r0+16,c0:c0+16],np.nan).astype(np.float32)
    mm=mask[k];yy=target[k];zero=cot==0
    x['samples'].append({'station':st,'cot_finite':int(np.isfinite(cot).sum()),'zero':int(zero.sum()),'zero_clear':int((zero&(clp==0)).sum()),'zero_water':int((zero&(clp==1)).sum()),'zero_ice':int((zero&(clp==2)).sum()),'zero_clp_missing':int((zero&~np.isfinite(clp)).sum()),'valid_pixels':int(mm.sum()),'valid_zero':int((mm&zero).sum()),'valid_zero_clear':int((mm&zero&(clp==0)).sum()),'valid_source_mismatch':int((mm&(~np.isfinite(cot)|(np.abs(cot-yy)>1e-3))).sum()),'coord_exact':bool(np.array_equal(lat[r0:r0+16],grids[f'{st}_lat']) and np.array_equal(lon[c0:c0+16],grids[f'{st}_lon']))})
  x['success']=True
 except Exception as e:x['error']=repr(e)
 res.append(x)
(B/'validation_cpp_source_retry.json').write_text(json.dumps({'n_files':len(res),'successes':sum(x['success'] for x in res),'elapsed_seconds':time.time()-t,'results':res},indent=2),encoding='utf-8')
print(json.dumps({'n_files':len(res),'successes':sum(x['success'] for x in res),'results':res},indent=2))
