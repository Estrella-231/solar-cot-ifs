import csv, json, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from netCDF4 import Dataset
BASE=Path('/tmp/cpp_audit_20260924')
ROWS=None; TARGET=None; MASK=None; GRIDS=None
STARTS={'sili':(1368,2201),'zhujia':(1278,2203)}

def init_worker(base):
    global ROWS,TARGET,MASK,GRIDS
    base=Path(base)
    ROWS=list(csv.DictReader((base/'validation_source_manifest.csv').open()))
    d=np.load(base/'validation_labels_for_cpp_audit.npz')
    TARGET=d['target_cot']; MASK=d['mask'].astype(bool)
    GRIDS=np.load(base/'station_grid_coords.npz')

def process_time(item):
    ts, sample_rows=item
    stamp=ts.replace('-','').replace(':','').replace('T','').replace('Z','')
    day=stamp[:8]
    path=Path('/home/Data_Pool_3/data/Global_CPP/FY/FY4B/Result/CPP/2025')/day/f'FY4B_AGRI_{stamp}.nc'
    try:
        result={'timestamp_utc':ts,'path':str(path),'n_samples':len(sample_rows),'qa_present':False,'coord_exact':True,'coord_checked':0,'cot_finite':0,'clp_finite':0,'zero':0,'zero_clear':0,'zero_water':0,'zero_ice':0,'zero_clp_missing':0,'valid_pixels':0,'valid_zero':0,'valid_zero_clear':0,'valid_zero_water':0,'valid_zero_ice':0,'valid_zero_clp_missing':0,'valid_cot_missing':0,'label_mismatch':0,'per_station':{},'error':None}
        with Dataset(path,'r') as ds:
            result['qa_present']='QA' in ds.variables
            lat=np.asarray(ds.variables['LAT'][:]); lon=np.asarray(ds.variables['LON'][:])
            for r in sample_rows:
                idx=int(r['order']); station=r['station_id']; r0,c0=STARTS[station]
                cot=np.ma.filled(ds.variables['COT'][r0:r0+16,c0:c0+16],np.nan).astype(np.float32)
                clp=np.ma.filled(ds.variables['CLP'][r0:r0+16,c0:c0+16],np.nan).astype(np.float32)
                glat=GRIDS[f'{station}_lat']; glon=GRIDS[f'{station}_lon']
                la=lat[r0:r0+16] if lat.ndim==1 else lat[r0:r0+16,c0:c0+16]
                lo=lon[c0:c0+16] if lon.ndim==1 else lon[r0:r0+16,c0:c0+16]
                coordok=np.array_equal(la,glat) and np.array_equal(lo,glon)
                result['coord_checked']+=1; result['coord_exact'] &= bool(coordok)
                mm=MASK[idx]; yy=TARGET[idx]
                finitec=np.isfinite(cot); finitcl=np.isfinite(clp); z=(cot==0)
                zc=z&(clp==0); zw=z&(clp==1); zi=z&(clp==2); zm=z&~finitcl
                vz=mm&z; vzc=vz&(clp==0); vzw=vz&(clp==1); vzi=vz&(clp==2); vzm=vz&~finitcl
                mismatch=mm&(~finitec | (np.abs(cot-yy)>1e-3))
                cnt={'samples':1,'cot_finite':int(finitec.sum()),'clp_finite':int(finitcl.sum()),'zero':int(z.sum()),'zero_clear':int(zc.sum()),'zero_water':int(zw.sum()),'zero_ice':int(zi.sum()),'zero_clp_missing':int(zm.sum()),'valid_pixels':int(mm.sum()),'valid_zero':int(vz.sum()),'valid_zero_clear':int(vzc.sum()),'valid_zero_water':int(vzw.sum()),'valid_zero_ice':int(vzi.sum()),'valid_zero_clp_missing':int(vzm.sum()),'valid_cot_missing':int((mm&~finitec).sum()),'label_mismatch':int(mismatch.sum()),'coord_exact':int(coordok)}
                result['per_station'][station]=cnt
                for k in ['cot_finite','clp_finite','zero','zero_clear','zero_water','zero_ice','zero_clp_missing','valid_pixels','valid_zero','valid_zero_clear','valid_zero_water','valid_zero_ice','valid_zero_clp_missing','valid_cot_missing','label_mismatch']:
                    result[k]+=cnt[k]
        return result
    except Exception as e:
        return {'timestamp_utc':ts,'path':str(path),'n_samples':len(sample_rows),'error':repr(e)}

manifest=list(csv.DictReader((BASE/'validation_source_manifest.csv').open()))
bytime={}
for order,row in enumerate(manifest):
    row['order']=order
    bytime.setdefault(row['timestamp_utc'],[]).append(row)
items=list(bytime.items())
started=time.time(); results=[]; done=0
with ProcessPoolExecutor(max_workers=4,initializer=init_worker,initargs=(str(BASE),)) as ex:
    futures=[ex.submit(process_time,item) for item in items]
    for f in as_completed(futures):
        r=f.result(); results.append(r); done+=1
        if done%100==0:
            good=sum(x.get('error') is None for x in results)
            (BASE/'validation_cpp_source_audit.progress.json').write_text(json.dumps({'timestamps_done':done,'timestamps_total':len(items),'samples_done':sum(x.get('n_samples',0) for x in results),'successful_files':good,'elapsed_seconds':time.time()-started},indent=2))
results.sort(key=lambda x:x['timestamp_utc'])
summary={'n_requested_samples':len(manifest),'n_unique_timestamps':len(items),'n_read_samples':sum(x.get('n_samples',0) for x in results if x.get('error') is None),'n_successful_files':sum(x.get('error') is None for x in results),'n_files_with_QA_variable':sum(bool(x.get('qa_present')) for x in results if x.get('error') is None),'coordinate_checks':sum(x.get('coord_checked',0) for x in results if x.get('error') is None),'coordinate_exact':sum(x.get('coord_exact',False)*x.get('coord_checked',0) for x in results if x.get('error') is None),'elapsed_seconds':time.time()-started,'global':{},'by_station':{},'errors':[x for x in results if x.get('error')]}
keys=['cot_finite','clp_finite','zero','zero_clear','zero_water','zero_ice','zero_clp_missing','valid_pixels','valid_zero','valid_zero_clear','valid_zero_water','valid_zero_ice','valid_zero_clp_missing','valid_cot_missing','label_mismatch']
for k in keys: summary['global'][k]=sum(x.get(k,0) for x in results if x.get('error') is None)
for st in ['sili','zhujia']:
 summary['by_station'][st]={k:sum(x.get('per_station',{}).get(st,{}).get(k,0) for x in results if x.get('error') is None) for k in keys}
summary['file_results']=results
(BASE/'validation_cpp_source_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
(BASE/'validation_cpp_source_audit.progress.json').write_text(json.dumps({'timestamps_done':len(items),'timestamps_total':len(items),'samples_done':summary['n_read_samples'],'successful_files':summary['n_successful_files'],'elapsed_seconds':summary['elapsed_seconds'],'complete':True},indent=2))
print(json.dumps({k:summary[k] for k in ['n_requested_samples','n_unique_timestamps','n_read_samples','n_successful_files','n_files_with_QA_variable','coordinate_checks','coordinate_exact','elapsed_seconds','global','by_station','errors']},indent=2))
