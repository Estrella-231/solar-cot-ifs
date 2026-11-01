"""FD-201 bounded train/validation raw-source reconstruction; stdout JSON only.
Prefix this script with SAMPLES=<contents of raw_source_sample_request.json>.
"""
import sys, json, hashlib, traceback
from pathlib import Path
sys.path.insert(0,'/home/Data_Pool_3/zjnu/cot_value_first_20261013/monthly_R_station16_20261013_v1/runtime_packages')
import numpy as np
import netCDF4
from datetime import datetime, timedelta
out={'samples':[], 'test_buffer_payloads_read':False}
gridpath=Path('/home/Data_Pool_3/chenyi/Auxiliary_data/HuNan_AGRI_Preprocessed/grid_static.npz')
with np.load(gridpath,allow_pickle=False) as z: lat,lon=z['lat'],z['lon']
for item in SAMPLES:
    r=item['row']; result={'station':r['station'],'utc':r['target_time_utc'],'split':r['split'],'cpp_path':r['cpp_path']}
    try:
        assert r['split'] in ('train','val')
        cr,cc={'sili':(163,148),'zhujia':(73,150)}[r['station']]
        sl=np.s_[cr-8:cr+8,cc-8:cc+8]
        raw=np.load(r['agri_path'],mmap_mode='r',allow_pickle=False)
        channels=[0,1,2,3,4,5,8,9,10,11,12,13,14]
        agr=np.asarray(raw[np.asarray(channels),sl[0],sl[1]],np.float32)
        with np.load(r['aux_path'],allow_pickle=False) as z:
            geom=np.stack([z[k][sl] for k in ['cosSOZ','cosRAA','day_mask']]).astype(np.float32)
            bits=z['agri_valid_bits'][sl]
        with netCDF4.Dataset(r['cpp_path']) as ds:
            la=ds.variables['LAT']; lo=ds.variables['LON']
            la=np.asarray(la[:,0] if la.ndim==2 else la[:]);lo=np.asarray(lo[0,:] if lo.ndim==2 else lo[:])
            rr=np.abs(la[:,None]-lat[sl][:,8][None,:]).argmin(0)
            cc0=np.abs(lo[:,None]-lon[sl][8,:][None,:]).argmin(0)
            rlo,rhi,clo,chi=rr.min(),rr.max()+1,cc0.min(),cc0.max()+1
            fields=[]
            for name in ['COT','CLP']:
                v=np.asarray(np.ma.filled(ds.variables[name][rlo:rhi,clo:chi],np.nan),np.float32)[np.ix_(rr-rlo,cc0-clo)]
                if np.diff(rr).mean()<0:v=v[::-1]
                if np.diff(cc0).mean()<0:v=v[:,::-1]
                fields.append(v)
            cot,clp=fields
            result['cpp_global_attributes']={k:str(ds.getncattr(k))[:500] for k in ds.ncattrs()}
            result['cpp_variables']=list(ds.variables)
        ok=np.stack([((bits & (1<<c))!=0) for c in channels]).all(0)&np.isfinite(agr).all(0)&np.isfinite(geom).all(0)
        mask=np.isfinite(cot)&(cot>=0)&(cot<=100)&np.isfinite(clp)&np.isin(clp,[0,1,2])&ok&(geom[2]>.5)&(geom[0]>np.cos(np.deg2rad(80)))
        x=np.concatenate([agr,geom]); x[:13,~ok]=np.nan
        delta=cot-np.asarray(item['cot'],np.float32)
        result.update(cpp_mtime_utc=datetime.utcfromtimestamp(Path(r['cpp_path']).stat().st_mtime).isoformat(),
                      raw_cot=cot.tolist(),packed_cot=item['cot'],
                      max_abs_cot_difference=float(np.nanmax(np.abs(delta))),
                      masked_rmse_cot_difference=float(np.sqrt(np.mean(delta[mask]**2))),
                      raw_x_matches_packed=bool(np.array_equal(x,np.asarray(item['x'],np.float32),equal_nan=True)),
                      raw_cot_matches_packed=bool(np.array_equal(cot,np.asarray(item['cot'],np.float32),equal_nan=True)),
                      raw_mask_matches_packed=bool(np.array_equal(mask,np.asarray(item['mask'],bool))),valid_pixels=int(mask.sum()))
    except Exception as e: result['error']=repr(e)
    out['samples'].append(result)
out['state']='PASS' if all(all(r.get(k,False) for k in ['raw_x_matches_packed','raw_cot_matches_packed','raw_mask_matches_packed']) for r in out['samples']) else 'FAIL_OR_BLOCKED'
print(json.dumps(out,indent=2))
