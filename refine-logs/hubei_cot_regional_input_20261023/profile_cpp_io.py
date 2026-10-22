"""Two train-file IO probes; no cache eviction, no cold-cache speedup claim."""
import csv,json,time,tempfile,hashlib
from pathlib import Path
import netCDF4 as nc
import numpy as np
P=Path(__file__).resolve().parent;rows=list(csv.DictReader((P/'selected_cpp_agri.csv').open()));chosen=[r for r in rows if r['split']=='train'];chosen=[chosen[500],chosen[2000]]
g=np.load(P/'grid_static256.npz');out=[]
def read(p):
 with nc.Dataset(p) as ds:
  la=np.array(ds['LAT'][:]);lo=np.array(ds['LON'][:]);yy=abs(la[:,None]-g['lat'][:,0]).argmin(0);xx=abs(lo[:,None]-g['lon'][0,:]).argmin(0);sl=(slice(int(yy.min()),int(yy.max())+1),slice(int(xx.min()),int(xx.max())+1))
  return np.ma.filled(np.ma.asarray(ds['COT'][sl],dtype=np.float32),np.nan),np.ma.filled(np.ma.asarray(ds['CLP'][sl],dtype=np.float32),np.nan)
for r in chosen:
 p=Path(r['CPP_path']);t=time.monotonic();cpu=time.process_time();a=read(p);direct=time.monotonic()-t;dcpu=time.process_time()-cpu;print('DIRECT',r['time_bjt'],direct,flush=True)
 with tempfile.TemporaryDirectory(prefix='cot_io_probe_',dir='/tmp') as tmp:
  q=Path(tmp)/p.name;h=hashlib.sha256();t=time.monotonic()
  with p.open('rb') as f,q.open('wb') as z:
   for block in iter(lambda:f.read(8<<20),b''):h.update(block);z.write(block)
  copy=time.monotonic()-t;t=time.monotonic();b=read(q);local=time.monotonic()-t
  assert all(np.array_equal(x,y,equal_nan=True) for x,y in zip(a,b))
 out.append(dict(time_bjt=r['time_bjt'],source=str(p),bytes=p.stat().st_size,direct_seconds=direct,direct_cpu_seconds=dcpu,copy_seconds=copy,local_read_seconds=local,source_sha256=h.hexdigest(),same_cropped_values=True));print('STAGED',copy,local,flush=True)
(Path('/home/Data_Pool/zjnu')/'hubei_cpp_io_profile_20261023.json').write_text(json.dumps(dict(state='IO_PROBE_COMPLETE',records=out,note='warm/order-dependent two-file probe; no eviction or global cold-cache speedup claim',test_payloads_read=0),indent=2))
