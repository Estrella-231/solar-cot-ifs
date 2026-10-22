"""Full regional CPP extraction only on exact corrected train/val AGRI join."""
import argparse,csv,json,hashlib,os,time
from pathlib import Path
from datetime import datetime,timedelta
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import netCDF4 as nc
from solar_geometry_noaa import compute_solar_angles
GRID=None;OUT=None

def hashbytes(v):return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def safe(v):
 if isinstance(v,np.ndarray):return [safe(x) for x in v.tolist()]
 if isinstance(v,(np.floating,float)):return float(v) if np.isfinite(v) else str(v)
 if isinstance(v,np.integer):return int(v)
 return str(v) if not isinstance(v,(str,int,bool,list,dict,type(None))) else v

def init(grid,out):
 global GRID,OUT
 with np.load(grid) as z:GRID={k:z[k] for k in z.files}
 OUT=Path(out)

def work(row):
 b=datetime.strptime(row['time_bjt'],'%Y%m%d%H%M');u=b-timedelta(hours=8);p=Path(row['CPP_path']);assert row['split'] in ['train','val'] and row['AGRI_UTC']==u.strftime('%Y%m%d%H%M') and u.strftime('%Y%m%d%H%M%S') in p.name
 st=p.stat()
 with nc.Dataset(p) as ds:
  lat=np.asarray(ds['LAT'][:]);lon=np.asarray(ds['LON'][:]);yy=abs(lat[:,None]-GRID['lat'][:,0]).argmin(0);xx=abs(lon[:,None]-GRID['lon'][0,:]).argmin(0)
  err=max(float(abs(lat[yy,None]-GRID['lat']).max()),float(abs(lon[xx][None,:]-GRID['lon']).max()));assert err<1e-4 and len(set(yy))==len(set(xx))==256
  y0,y1=int(yy.min()),int(yy.max())+1;x0,x1=int(xx.min()),int(xx.max())+1
  co=np.ma.filled(ds['COT'][y0:y1,x0:x1],np.nan).astype(np.float32)[np.ix_(yy-y0,xx-x0)];cl=np.ma.filled(ds['CLP'][y0:y1,x0:x1],np.nan).astype(np.float32)[np.ix_(yy-y0,xx-x0)]
  metadata=dict(dimensions={k:len(v) for k,v in ds.dimensions.items()},variables=list(ds.variables),global_attributes={k:safe(ds.getncattr(k)) for k in ds.ncattrs()},COT_attributes={k:safe(ds['COT'].getncattr(k)) for k in ds['COT'].ncattrs()},CLP_attributes={k:safe(ds['CLP'].getncattr(k)) for k in ds['CLP'].ncattrs()})
 z,az,day=compute_solar_angles(GRID['lat'],GRID['lon'],u);solar=(day>.5)&(z<80)
 raw=np.isfinite(co)&(co>=0)&(co<=100);phase=np.isin(cl,[0,1,2]);valid=raw&phase&solar
 dest=OUT/'payload'/row['split']/(row['time_bjt']+'.npz');assert not dest.exists();np.savez_compressed(dest,cot=co,clp=cl,mask=valid)
 stations={}
 for name,(y,x) in {'Niushou':(111,124),'Wangzhai':(112,126),'Mizhuang':(110,128)}.items():stations[name]=dict(valid_center=bool(valid[y,x]),valid_3x3=int(valid[y-1:y+2,x-1:x+2].sum()),cot_center=safe(co[y,x]),clp_center=safe(cl[y,x]))
 return dict(**row,source_bytes=st.st_size,source_mtime_ns=st.st_mtime_ns,output_path=str(dest),output_sha256=sha(dest),output_bytes=dest.stat().st_size,crop_cot_sha256=hashbytes(co),crop_clp_sha256=hashbytes(cl),axes_sha256=hashbytes(np.r_[lat,lon]),max_grid_error=err,valid_pixels=int(valid.sum()),solar_pixels=int(solar.sum()),raw_cot_valid_pixels=int(raw.sum()),unknown_phase_pixels=int((~phase).sum()),clear_positive_cot_pixels=int(((cl==0)&raw&(co>0)).sum()),cloud_missing_cot_pixels=int((np.isin(cl,[1,2])&~raw).sum()),stations=stations,metadata=metadata)

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--grid',type=Path,required=True);p.add_argument('--original-grid',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
 a.out.mkdir(exist_ok=False);(a.out/'payload/train').mkdir(parents=True);(a.out/'payload/val').mkdir(parents=True)
 g=np.load(a.grid);og=np.load(a.original_grid);checks={k:float(abs(g[k].astype(np.float64)-og[k].astype(np.float64)).max()) for k in g.files};assert max(checks.values())==0
 rows=list(csv.DictReader(a.manifest.open()));assert len(rows)==3124 and len({r['time_bjt'] for r in rows})==len(rows);assert all(r['split'] in ['train','val'] for r in rows)
 started=time.monotonic();records=[];failures=[]
 (a.out/'STARTED.json').write_text(json.dumps(dict(pid=os.getpid(),frames=len(rows),workers=a.workers,manifest_sha256=sha(a.manifest),grid_sha256=sha(a.grid),grid_array_max_differences=checks)))
 with ProcessPoolExecutor(max_workers=a.workers,initializer=init,initargs=(str(a.grid),str(a.out))) as pool:
  futures={pool.submit(work,r):r for r in rows}
  for fut in as_completed(futures):
   try:records.append(fut.result())
   except Exception as e:failures.append(dict(row=futures[fut],error=repr(e)))
   if (len(records)+len(failures))%100==0:
    progress=dict(done=len(records),failed=len(failures),total=len(rows),seconds=time.monotonic()-started);(a.out/'PROGRESS.json').write_text(json.dumps(progress));print(json.dumps(progress),flush=True)
 records.sort(key=lambda r:r['time_bjt']);(a.out/'frames.json').write_text(json.dumps(records))
 summary={}
 for split in ['train','val']:
  rr=[r for r in records if r['split']==split];summary[split]=dict(frames=len(rr),dates=len(set(r['time_bjt'][:8] for r in rr)),valid_pixels=sum(r['valid_pixels'] for r in rr),frames_with_valid=sum(r['valid_pixels']>0 for r in rr),by_month={m:dict(frames=sum(r['time_bjt'].startswith(m) for r in rr),valid_pixels=sum(r['valid_pixels'] for r in rr if r['time_bjt'].startswith(m))) for m in sorted(set(r['time_bjt'][:6] for r in rr))},station_valid_centers={st:sum(r['stations'][st]['valid_center'] for r in rr) for st in ['Niushou','Wangzhai','Mizhuang']})
 rep=dict(state='REGIONAL_CPP_PAYLOAD_EXTRACTION_COMPLETE' if not failures else 'FAILED_RETAIN_PARTIAL',summary=summary,failures=failures,seconds=time.monotonic()-started,output_bytes=sum(r['output_bytes'] for r in records),coordinate_max_error=max([r['max_grid_error'] for r in records],default=None),clear_positive_cot_pixels=sum(r['clear_positive_cot_pixels'] for r in records),cloud_missing_cot_pixels=sum(r['cloud_missing_cot_pixels'] for r in records),test_payloads_read=0,buffer_payloads_read=0,ghi_labels_read=0,source_full_file_hashes_computed=False,read_subset_hashes_recorded=True,script_sha256=sha(__file__))
 (a.out/'REPORT.json').write_text(json.dumps(rep,indent=2));print(json.dumps(rep),flush=True)
 if failures:raise RuntimeError('failed source frames retained')
if __name__=='__main__':main()
