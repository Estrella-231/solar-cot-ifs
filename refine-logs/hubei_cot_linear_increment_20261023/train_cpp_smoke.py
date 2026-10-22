"""One TRAIN-only 24-frame CPP regional read; no GHI, no model, no test."""
import json,csv,time,hashlib
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
import netCDF4 as nc
ROOT=Path('/home/Data_Pool_3/data/Global_CPP/FY/FY4B/Result/CPP');OUT=Path('/home/Data_Pool/zjnu/hubei_cpp_regional_train_smoke_20261023')
GRID=Path('/home/Data_Pool_3/chenyi/Auxiliary_data/HuBei_AGRI_Preprocessed/grid_static.npz')
M=Path(__file__).with_name('agri_paths_trainval_staged.csv');allowed={r['time_bjt']:r['split'] for r in csv.DictReader(M.open())}
OUT.mkdir(exist_ok=False);grid=np.load(GRID);lat=grid['lat'];lon=grid['lon'];assert lat.shape==lon.shape==(256,256)
# Date/time fixed by source metadata, never by COT or model error.
start=datetime(2025,8,4,8,15);times=[start+timedelta(minutes=15*j) for j in range(24)];assert all(allowed.get(t.strftime('%Y%m%d%H%M'))=='train' for t in times)
records=[];arrays=[];phases=[];beg=time.monotonic()
for t in times:
    u=t-timedelta(hours=8);f=ROOT/u.strftime('%Y')/u.strftime('%Y%m%d')/('FY4B_AGRI_'+u.strftime('%Y%m%d%H%M%S')+'.nc')
    if not f.is_file():records.append(dict(time_bjt=t.isoformat(' '),path=str(f),state='missing'));continue
    with nc.Dataset(f) as ds:
        la=np.array(ds['LAT'][:]);lo=np.array(ds['LON'][:]);ri=abs(la[:,None]-lat[:,0]).argmin(0);ci=abs(lo[:,None]-lon[0,:]).argmin(0)
        err=max(float(abs(la[ri,None]-lat).max()),float(abs(lo[ci][None,:]-lon).max()));assert err<1e-4 and len(set(ri))==len(set(ci))==256
        r0,r1=int(ri.min()),int(ri.max())+1;c0,c1=int(ci.min()),int(ci.max())+1
        co=np.ma.filled(ds['COT'][r0:r1,c0:c1],np.nan).astype(np.float32)[np.ix_(ri-r0,ci-c0)]
        cl=np.ma.filled(ds['CLP'][r0:r1,c0:c1],-999).astype(np.float32)[np.ix_(ri-r0,ci-c0)]
        valid=np.isfinite(co)&(co>=0)&(co<=100);co[~valid]=np.nan;arrays.append(co);phases.append(cl)
        rec=dict(time_bjt=t.isoformat(' '),utc=u.isoformat(' '),path=str(f),state='read',source_bytes=f.stat().st_size,max_grid_error=err,cot_valid_fraction=float(valid.mean()),clp_values={str(x):int((cl==x).sum()) for x in np.unique(cl)},clear_cot_valid=int((valid&(cl==0)).sum()),cloud_cot_missing=int((~valid&np.isin(cl,[1,2])).sum()),station_cot=[float(co[r,c]) for r,c in [(111,124),(112,126),(110,128)]])
        records.append(rec);print(t.isoformat(),rec['cot_valid_fraction'],flush=True)
np.savez_compressed(OUT/'train_cpp_24frame.npz',cot=np.array(arrays),clp=np.array(phases),lat=lat,lon=lon)
rep=dict(state='TRAIN_CPP_REGION_SMOKE_COMPLETE' if len(arrays)==24 else 'INCOMPLETE_SOURCE_SEQUENCE',requested_frames=24,read_frames=len(arrays),records=records,seconds=time.monotonic()-beg,manifest_sha256=hashlib.sha256(M.read_bytes()).hexdigest(),grid_sha256=hashlib.sha256(GRID.read_bytes()).hexdigest(),fill_missing_with_clear=False,ghi_labels_read=0,validation_payloads_read=0,test_payloads_read=0,buffer_payloads_read=0,trained=False,note='one train sequence feasibility only; no claim of seasonal coverage, R-chain consistency or forecast skill')
(OUT/'REPORT.json').write_text(json.dumps(rep,indent=2));print('TRAIN_CPP_REGION_SMOKE_DONE',len(arrays),flush=True)
