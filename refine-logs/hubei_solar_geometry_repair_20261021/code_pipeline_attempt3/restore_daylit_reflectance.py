"""Recover only newly-daylit C01-C06 pixels from exact UTC HDF DN/LUT."""
import argparse,csv,json,hashlib,re,tarfile
from pathlib import Path
from datetime import datetime,timedelta
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import netCDF4 as nc
STAMP=re.compile(r'_(\d{14})_\d{14}_4000M',re.I)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def task(args):
 u,request_path,sources,grid_path,out,oldstage=args;z=np.load(request_path);yy,xx=z['y'].astype(int),z['x'].astype(int);g=np.load(grid_path);rr=g['row_index'][yy,xx];cc=g['col_index'][yy,xx];rlo,rhi=int(rr.min()),int(rr.max())+1;clo,chi=int(cc.min()),int(cc.max())+1;old=Path(oldstage)/'data'/f'{u}.npy';oldraw=np.load(old,mmap_mode='r') if old.exists() else None;errors=[]
 for source in sources:
  try:
   with nc.Dataset(source) as f:
    grouped='Data' in f.groups and 'Calibration' in f.groups;d=f.groups['Data'] if grouped else f;cal=f.groups['Calibration'] if grouped else f
    vals=[]
    def image(ch):
     raw=np.asarray(d[f'NOMChannel{ch:02d}'][rlo:rhi,clo:chi],dtype=np.int64);raw=np.where((raw==65534)|(raw==65535),4096,raw);raw=np.clip(raw,0,4096);lut=np.append(np.asarray(cal[f'CALChannel{ch:02d}' if grouped else f'CalChannel{ch:02d}'][:],np.float32),np.float32(0));return lut[raw]
    for ch in range(1,7):vals.append(image(ch)[rr-rlo,cc-clo])
    val=np.asarray(vals,np.float32);assert np.isfinite(val).all()
    # Exact historical source compatibility on temperature, which was never night-zeroed.
    assert 'reference_ir13' in z.files, 'independent historical C13 source witness required'
    tempdiff=float(abs(image(13)[rr-rlo,cc-clo]-z['reference_ir13']).max());assert tempdiff<1e-4,('C13 source compatibility',tempdiff)
    if oldraw is not None:
     iy,ix=np.where((g['row_index']>=rlo)&(g['row_index']<rhi)&(g['col_index']>=clo)&(g['col_index']<chi));tp=image(13);delta=tp[g['row_index'][iy,ix]-rlo,g['col_index'][iy,ix]-clo]-oldraw[12,iy,ix];tempdiff=float(abs(delta).max());assert tempdiff<1e-4,('C13 source compatibility',tempdiff)
   path=Path(out)/f'{u}.npz';np.savez_compressed(path,y=yy.astype(np.uint16),x=xx.astype(np.uint16),reflectance=val)
   return dict(utc=u,source_hdf=str(source),source_bytes=Path(source).stat().st_size,request_sha256=sha(request_path),restored_sha256=sha(path),pixels=len(yy),C13_compatibility_max_abs=tempdiff,source_compatibility_checked=tempdiff is not None)
  except Exception as e:errors.append(dict(source=str(source),error=repr(e)))
 raise RuntimeError(json.dumps(dict(utc=u,all_candidate_failures=errors)))
def main():
 p=argparse.ArgumentParser();p.add_argument('--requests',type=Path,required=True);p.add_argument('--grid',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=6);p.add_argument('--old-stage',default='/tmp/zjnu_hubei_simvp_stage_20261016');a=p.parse_args();a.output.mkdir(exist_ok=False)
 roots=[Path('/home/Data_Pool/data/FY/FY4B/real_time_data/AGRI/4KM'),Path('/home/Data_Pool/data/FY/FY4B/AGRI/4KM')];look={};tasks=[];reqs=list(a.requests.glob('*.npz'))
 for req in sorted(reqs):
  u=req.stem;day=u[:8]
  if day not in look:
   m={}
   for root in roots:
    d=root/day[:4]/day
    if d.is_dir():
     for f in sorted(d.glob('*4000M*.HDF')):
      mat=STAMP.search(f.name)
      if mat:m.setdefault(mat.group(1)[:12],[]).append(str(f))
   look[day]=m
  src=look[day].get(u,[]);assert src,('NO_EXACT_UTC_HDF',u)
  tasks.append((u,str(req),src,str(a.grid),str(a.output),a.old_stage))
 print('EXACT_UTC_HDF_REQUESTS',len(tasks),flush=True);records=[]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  for r in pool.map(task,tasks):
   records.append(r)
   with (a.output/'completed.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
   if len(records)%25==0:print('RESTORED',len(records),len(tasks),flush=True)
 assert len(records)==len(tasks)
 report=dict(state='EXACT_UTC_HDF_RADIANCE_RESTORATION_COMPLETE',frames=len(records),pixels=sum(r['pixels'] for r in records),records=records,test_payloads_read=0,buffer_payloads_read=0);(a.output/'RESTORATION_REPORT.json').write_text(json.dumps(report,indent=2));print('RESTORATION_COMPLETE',len(records),flush=True)
 archive=a.output.parent/'restored_visible.tar.gz';partial=archive.with_suffix('.part')
 with tarfile.open(partial,'w:gz') as t:
  for f in a.output.iterdir():t.add(f,arcname=f.name)
 partial.replace(archive)
if __name__=='__main__':main()
