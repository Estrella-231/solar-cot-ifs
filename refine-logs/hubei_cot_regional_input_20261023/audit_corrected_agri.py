"""CPU-only node-local corrected AGRI data audit for the exact CPP join."""
import argparse,csv,json,hashlib,time
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
from solar_geometry_noaa import compute_solar_angles

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8<<20),b''):h.update(x)
 return h.hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--manifest',type=Path,required=True);a.add_argument('--grid',type=Path,required=True);a.add_argument('--out',type=Path,required=True);v=a.parse_args();v.out.mkdir(exist_ok=False)
 rows=list(csv.DictReader(v.manifest.open()));g=dict(np.load(v.grid));start=time.monotonic();failures=[];records=[];maxerr=0.
 for i,r in enumerate(rows):
  try:
   assert r['split'] in ['train','val'];b=datetime.strptime(r['time_bjt'],'%Y%m%d%H%M');u=b-timedelta(hours=8);assert u.strftime('%Y%m%d%H%M')==r['AGRI_UTC'];p=Path(r['AGRI_path']);ap=Path(r['aux_path'])
   assert sha(p)==r['AGRI_sha256'] and sha(ap)==r['aux_sha256'];raw=np.load(p,mmap_mode='r',allow_pickle=False);assert raw.shape==(20,256,256) and np.isfinite(raw).all();assert np.array_equal(raw[15],g['saa']) and np.array_equal(raw[16],g['saz'])
   with np.load(ap) as z:aux=np.stack([z[k] for k in ['cosSOZ','cosRAA','day_mask']])
   soz,az,day=compute_solar_angles(g['lat'],g['lon'],u);az=np.where(day>.5,az,0);expected=np.stack((np.cos(np.deg2rad(soz)),np.cos(np.deg2rad(az-g['saa'])),day)).astype(np.float32);err=float(abs(aux-expected).max());assert err<1e-6,err;maxerr=max(maxerr,err)
   records.append(dict(time_bjt=r['time_bjt'],split=r['split'],max_geometry_error=err,IR_zero_pixels=int((raw[8:15]==0).any(0).sum())))
  except Exception as e:failures.append(dict(row=r,error=repr(e)))
  if (i+1)%100==0:
   pr=dict(done=i+1,failed=len(failures),total=len(rows),seconds=time.monotonic()-start);(v.out/'PROGRESS.json').write_text(json.dumps(pr));print(json.dumps(pr),flush=True)
 (v.out/'frames.json').write_text(json.dumps(records));rep=dict(state='PASS_CORRECTED_AGRI_REGIONAL_JOIN' if not failures else 'FAIL_PRESERVE_ROWS',frames=len(records),failures=failures,max_geometry_error=maxerr,seconds=time.monotonic()-start,IR_zero_frames=sum(r['IR_zero_pixels']>0 for r in records),test_payloads_read=0,buffer_payloads_read=0)
 (v.out/'REPORT.json').write_text(json.dumps(rep,indent=2));print(json.dumps(rep),flush=True)
if __name__=='__main__':main()
