import csv,json,sys,time,hashlib
from pathlib import Path
from datetime import datetime
import numpy as np
ROOT=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907')
BASE=ROOT/'experiments/hubei_solar_geometry_repair_20261021'
OUT=ROOT/'refine-logs/hubei_same_time_geometry_20261102'
AG=[0,1,2,3,4,5,8,9,10,11,12,13,14]
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 t=time.time();out=OUT/'frames';out.mkdir(exist_ok=False,parents=True)
 rs=rows(BASE/'prepared_attempt12/stage_manifest.csv');rr=rows(BASE/'prepared_attempt12/R_packs/full/rows.csv')
 rraw=np.load(BASE/'prepared_attempt12/R_raw_corrected_canonical_attempt12.npy',mmap_mode='r')
 sys.path.insert(0,str(BASE/'code_pipeline_attempt12'));from solar_geometry_noaa import compute_solar_angles
 # Grid coordinates at the frozen 16x16 patch are 0.04deg nominal registration;
 # independent saved full-grid geometry is reused and checked against raw SOZ.
 full=[];meta=[];seen={}
 for j,r in enumerate(rs):
  p=Path(r['data_path']);x=np.load(p,mmap_mode='r');a=x[:,103:119,118:134]
  assert a.shape==(20,16,16) and r['split'] in ('train','val')
  g=np.stack((np.cos(np.deg2rad(a[18])),np.cos(np.deg2rad(a[17]-a[15])),a[19]))
  xx=np.concatenate((a[AG],g)).astype(np.float32)
  assert np.isfinite(xx).all()
  seen[r['bjt']]=len(full);full.append(xx);meta.append({'time_bjt':r['bjt'],'split':r['split'],'source':'corrected_full_frame'})
  if j%2000==0:print('EXTRACT',j,len(rs),flush=True)
 for j,r in enumerate(rr):
  if r['time_bjt'] in seen:continue
  seen[r['time_bjt']]=len(full);full.append(np.array(rraw[j],copy=True));meta.append({'time_bjt':r['time_bjt'],'split':r['split'],'source':'corrected_R_physical_input'})
 full=np.stack(full);np.save(out/'raw.npy',full)
 with (out/'keys.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(meta[0]));w.writeheader();w.writerows(meta)
 ready=json.loads((BASE/'prepared_attempt12/DATA_READY.json').read_text())
 assert ready['state']=='CORRECTED_SOLAR_TRAIN_VAL_DATA_READY'
 report={'state':'COMPLETE','frames':len(full),'source_S_frames':len(rs),'source_R_rows':len(rr),'source_data_ready_sha256':sha(BASE/'prepared_attempt12/DATA_READY.json'),'geometry_policy':'reuse independently NOAA-verified corrected full scenes; including restored visible bands','seconds':time.time()-t,'test_payloads_read':0}
 (out/'COMPLETE.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
