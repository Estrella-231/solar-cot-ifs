"""Audit all indexed train/validation geometry and queue exact daylight restoration."""
import argparse,csv,hashlib,json,tarfile,io
from pathlib import Path
from datetime import datetime,timedelta
from collections import Counter
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from solar_geometry_noaa import compute_solar_angles

def read(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,rows):
 with Path(p).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--grid',type=Path,required=True);p.add_argument('--rpack',type=Path,required=True);a=p.parse_args();out=a.root;out.mkdir(exist_ok=False);meta=out/'metadata';meta.mkdir()
 with tarfile.open(a.metadata) as t:
  assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in t.getmembers());t.extractall(meta)
 rows=read(meta/'validation_source_index.csv');assert Counter(r['split'] for r in rows)=={'train':4086,'val':407};dates={r['date_BJT'].replace('-',''):r['split'] for r in read(meta/'day_assignment.csv')}
 grid=dict(np.load(a.grid));assert all(g.shape==(256,256) for g in grid.values());lat,lon,saa=grid['lat'],grid['lon'],grid['saa'];assert abs(float(lat[111,124])-32.08)<1e-4 and int(grid['row_index'][111,124])==557
 files={};seq_counts=Counter()
 for r in rows:
  d=r['data_paths'].split('|');g=r['aux_paths'].split('|');assert len(d)==len(g)==24
  ts=[]
  for dp,gp in zip(d,g):
   u=Path(dp).stem[:12];b=Path(gp).stem[:12];utc=datetime.strptime(u,'%Y%m%d%H%M');bjt=datetime.strptime(b,'%Y%m%d%H%M');assert bjt-utc==timedelta(hours=8) and dates[b[:8]]==r['split'];ts.append(bjt)
   val=dict(utc=u,bjt=b,split=r['split'],data_name=Path(dp).name,aux_name=Path(gp).name)
   if u in files:assert files[u]==val
   files[u]=val
  assert all(ts[j+1]-ts[j]==timedelta(minutes=15) for j in range(23));seq_counts[r['split']]+=1
 (out/'aux').mkdir();(out/'restore_requests').mkdir();requests={};frame_results=[];ref=(out/'reference_cases.jsonl').open('w');points=[(111,124),(112,126),(110,128),(0,0),(255,255)]
 def geometry(stamp,sl=None):
  utc=datetime.strptime(stamp,'%Y%m%d%H%M');z,az,day=compute_solar_angles(lat,lon,utc);az=np.where(day>.5,az,0.);co=np.cos(np.deg2rad(z));ca=np.cos(np.deg2rad(az-saa));return np.stack((co,ca,day)).astype(np.float32),z,az
 def compute_frame(item):
  u,f=item
  dp=a.stage/'data'/f['data_name'];gp=a.stage/'aux'/f['aux_name'];raw=np.load(dp,mmap_mode='r');assert raw.shape==(20,256,256) and raw.dtype==np.float32;old=np.load(gp);oldgeom=np.stack([old[k] for k in ('cosSOZ','cosRAA','day_mask')]);new,z,az=geometry(u)
  assert np.max(abs(raw[15]-saa))<1e-4 and np.max(abs(raw[16]-grid['saz']))<1e-4
  old_aux_z_error=float(np.max(abs(oldgeom[0]-np.cos(np.deg2rad(raw[18])))));old_aux_raa_error=float(np.max(abs(oldgeom[1]-np.cos(np.deg2rad(raw[17]-raw[15])))));old_aux_day_mismatch=int(np.count_nonzero(oldgeom[2]!=raw[19]))
  if old_aux_z_error>=1e-4 or old_aux_raa_error>=1e-4 or old_aux_day_mismatch:print('OLD_AUX_MISMATCH',u,old_aux_z_error,old_aux_raa_error,old_aux_day_mismatch,flush=True)
  assert np.isfinite(new).all() and abs(new[:2]).max()<=1.000001 and np.isin(new[2],[0,1]).all()
  restore=(new[2]>.5)&(raw[19]<.5);removed=(new[2]<.5)&(raw[19]>.5)
  if restore.any():requests[u]=restore
  np.savez_compressed(out/'aux'/f['aux_name'],cosSOZ=new[0],cosRAA=new[1],day_mask=new[2])
  result=dict(**f,old_aux_cosSOZ_error=old_aux_z_error,old_aux_cosRAA_error=old_aux_raa_error,old_aux_day_mismatch=old_aux_day_mismatch,data_path=str(dp),aux_path=str(gp),restore_pixels=int(restore.sum()),zero_pixels=int(removed.sum()),max_delta_cosSOZ=float(abs(new[0]-oldgeom[0]).max()),max_delta_cosRAA=float(abs(new[1]-oldgeom[1]).max()))
  refrows=[]
  for yy,xx in points:refrows.append(dict(utc=u,lat=float(lat[yy,xx]),lon=float(lon[yy,xx]),zenith=float(z[yy,xx]),azimuth=float(compute_solar_angles(float(lat[yy,xx]),float(lon[yy,xx]),datetime.strptime(u,'%Y%m%d%H%M'))[1]),split=f['split']))
  return result,refrows
 with ThreadPoolExecutor(max_workers=4) as pool:
  for j,(result,refrows) in enumerate(pool.map(compute_frame,sorted(files.items()))):
   frame_results.append(result)
   for q in refrows:ref.write(json.dumps(q)+'\n')
   if j%256==0:print('GEOMETRY_FRAMES',j,len(files),flush=True)
 ref.close();write(out/'frames_manifest.csv',frame_results)
 # R contains additional train/val times; validate and recompute all, too.
 rn=json.loads((a.rpack/'norm.json').read_text());rx=np.load(a.rpack/'x_norm.npy',mmap_mode='r');rr=read(a.rpack/'rows.csv');mu=np.array(rn['mean'],np.float32)[None,:,None,None];sd=np.array(rn['std'],np.float32)[None,:,None,None];rraw=np.asarray(rx)*sd+mu;gresults=[]
 with (out/'reference_cases.jsonl').open('a') as ref:
  for i,r in enumerate(rr):
   b=r['time_bjt'];assert r['split'] in ('train','val') and dates[b[:8]]==r['split'];u=(datetime.strptime(b,'%Y%m%d%H%M')-timedelta(hours=8)).strftime('%Y%m%d%H%M');new,z,az=geometry(u);sl=(slice(103,119),slice(118,134));ng=new[:,sl[0],sl[1]];og=rraw[i,13:16].copy();need=(ng[2]>.5)&(og[2]<.5)
   if need.any():
    if u not in requests:requests[u]=np.zeros((256,256),bool)
    requests[u][sl]|=need
   rraw[i,13:16]=ng;rraw[i,:6,ng[2]<.5]=0
   gresults.append(dict(time_bjt=b,utc=u,split=r['split'],restore_pixels=int(need.sum()),zero_pixels=int(((ng[2]<.5)&(og[2]>.5)).sum()),max_delta_cosSOZ=float(abs(ng[0]-og[0]).max()),max_delta_cosRAA=float(abs(ng[1]-og[1]).max())))
   for yy,xx in points[:3]:
    zz,aa,dd=compute_solar_angles(float(lat[yy,xx]),float(lon[yy,xx]),datetime.strptime(u,'%Y%m%d%H%M'));ref.write(json.dumps(dict(utc=u,lat=float(lat[yy,xx]),lon=float(lon[yy,xx]),zenith=float(zz),azimuth=float(aa),split=r['split']))+'\n')
 write(out/'R_geometry_manifest.csv',gresults);np.save(out/'R_raw_pending_restoration.npy',rraw,allow_pickle=False)
 requestrows=[]
 for u,m in sorted(requests.items()):
  y,x=np.where(m);np.savez_compressed(out/'restore_requests'/f'{u}.npz',y=y.astype(np.uint16),x=x.astype(np.uint16));requestrows.append(dict(utc=u,bjt=(datetime.strptime(u,'%Y%m%d%H%M')+timedelta(hours=8)).strftime('%Y%m%d%H%M'),pixels=len(y)))
 if requestrows:write(out/'restore_requests.csv',requestrows)
 report=dict(old_aux_inconsistent_frames=sum(r['old_aux_cosSOZ_error']>=1e-4 or r['old_aux_cosRAA_error']>=1e-4 or r['old_aux_day_mismatch']>0 for r in frame_results),restoration_day_source='raw channel19 controls actual zeroed visible radiances; old aux inconsistency recorded and all aux rebuilt',state='GEOMETRY_COMPUTED_REQUIRES_INDEPENDENT_REFERENCE_AND_RADIANCE_RESTORATION',S_sequences=dict(seq_counts),S_unique_frames=len(files),S_frames_by_split=dict(Counter(f['split'] for f in files.values())),R_rows=len(rr),R_rows_by_split=dict(Counter(r['split'] for r in rr)),restoration_frames=len(requests),restoration_pixels=sum(int(m.sum()) for m in requests.values()),source_grid_sha256=sha(a.grid),source_S_index_sha256=sha(meta/'validation_source_index.csv'),source_R_pack_sha256=sha(a.rpack/'pack_status.json'),solar_source_sha256=sha(Path(__file__).parent/'solar_geometry_noaa.py'),test_payloads_read=0,buffer_payloads_read=0)
 (out/'GEOMETRY_CALCULATION_REPORT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
