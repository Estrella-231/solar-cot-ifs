"""Read bounded historical CTH subsets, preserving raw units and missing QA."""
import argparse,hashlib,json
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np

def safe(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    return v

def validate_history_request(request):
    init=datetime.fromisoformat(request['init_time_BJT'])
    assert init.tzinfo is None, 'init must be explicit naive BJT, not an offset-bearing timestamp'
    rows=request['rows'];assert len(rows)==8, 'Expected exactly 8 history frames'
    expected=[(init-timedelta(minutes=off)).strftime('%Y%m%d%H%M') for off in range(105,-1,-15)]
    assert [r['time_bjt'] for r in rows]==expected, 'History must be unique, ordered, continuous -105..0 min'
    assert len({r['source'] for r in rows})==8, 'History source paths must be unique'
    for r in rows:
        assert r['split']=='train', 'This source request is TRAIN only'
        utc=(datetime.strptime(r['time_bjt'],'%Y%m%d%H%M')-timedelta(hours=8))
        assert utc.strftime('%Y%m%d%H%M')==r['UTC'], 'UTC must equal BJT minus 8h'
        assert utc.strftime('%Y%m%d%H%M%S') in Path(r['source']).name, 'Source filename time disagrees'
    return init

def main():
    import netCDF4
    ap=argparse.ArgumentParser();ap.add_argument('--request',type=Path,required=True);ap.add_argument('--grid',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    request=json.loads(a.request.read_text());validate_history_request(request)
    with np.load(a.grid,allow_pickle=False) as z:grid={k:z[k] for k in ['lat','lon']}
    a.out.mkdir(exist_ok=False);records=[]
    for row in request['rows']:
        assert row['split']=='train'
        t=datetime.strptime(row['time_bjt'],'%Y%m%d%H%M');u=t-timedelta(hours=8)
        assert u.strftime('%Y%m%d%H%M')==row['UTC'] and t<=datetime.fromisoformat(request['init_time_BJT'])
        path=Path(row['source']);assert u.strftime('%Y%m%d%H%M%S') in path.name
        before=path.stat()
        with netCDF4.Dataset(path) as ds:
            lat=np.asarray(ds['LAT'][:]);lon=np.asarray(ds['LON'][:]);assert lat.ndim==lon.ndim==1
            yy=abs(lat[:,None]-grid['lat'][:,0]).argmin(0);xx=abs(lon[:,None]-grid['lon'][0,:]).argmin(0)
            error=max(float(abs(lat[yy,None]-grid['lat']).max()),float(abs(lon[xx][None,:]-grid['lon']).max()));assert error<1e-4
            assert len(set(yy))==len(yy) and len(set(xx))==len(xx)
            y0,y1=int(yy.min()),int(yy.max())+1;x0,x1=int(xx.min()),int(xx.max())+1
            axes=(ds['LAT'].dimensions[0],ds['LON'].dimensions[0]);assert 'CTH' in ds.variables
            payload={};attributes={}
            for name in ['CTH','CLP','COT','QA']:
                if name not in ds.variables:continue
                v=ds[name];assert v.dimensions==axes,(name,v.dimensions,axes)
                raw=np.ma.filled(v[y0:y1,x0:x1].astype(np.float64),np.nan)[np.ix_(yy-y0,xx-x0)]
                payload[name+'_raw']=raw;attributes[name]={k:safe(v.getncattr(k)) for k in v.ncattrs()}
            metadata=dict(global_attributes={k:safe(ds.getncattr(k)) for k in ds.ncattrs()},variable_attributes=attributes,QA_present='QA' in ds.variables,decoded_values_in_original_units=True)
        after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        dest=a.out/(row['time_bjt']+'.npz');np.savez_compressed(dest,**payload)
        finite=payload['CTH_raw'][np.isfinite(payload['CTH_raw'])]
        records.append(dict(**row,source_size=before.st_size,source_mtime_ns=before.st_mtime_ns,max_grid_error=error,
            CTH_raw_finite_fraction=float(np.isfinite(payload['CTH_raw']).mean()),CTH_raw_quantiles=np.percentile(finite,[0,5,50,95,100]).tolist() if finite.size else [],
            subset_sha256=hashlib.sha256(np.ascontiguousarray(payload['CTH_raw']).tobytes()).hexdigest(),output_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),metadata=metadata))
    (a.out/'COMPLETE.json').write_text(json.dumps(dict(state='RAW_HISTORY_CTH_EXTRACTED_UNITS_AND_QA_REVIEW_PENDING',records=records,
        history_contract='8 ordered unique frames at -105,-90,...,0 minutes',actual_future_CTH_used=False,release_time_verified=False,deployable_claim=False,GHI_labels_read=0,test_payloads_read=0),indent=2))

if __name__=='__main__':main()
