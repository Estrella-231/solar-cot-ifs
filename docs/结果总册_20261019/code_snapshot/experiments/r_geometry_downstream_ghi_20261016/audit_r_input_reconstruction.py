#!/usr/bin/env python3
"""Verify live AGRI/aux reconstruction against the exact paired-R training inputs."""
import argparse,csv,hashlib,json,random
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
from solar_geometry_corrected import compute_solar_angles

CHANNELS=(0,1,2,3,4,5,8,9,10,11,12,13,14)
COORD={'sili':(163,148),'zhujia':(73,150)}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<22),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--paired-r',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--grid-static',type=Path,required=True);p.add_argument('--monthly-norm',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--n',type=int,default=64);a=p.parse_args()
 pack=a.paired_r/'data/paired_common_v1';rows=list(csv.DictReader((pack/'rows_supervised.csv').open(newline='',encoding='utf-8')))
 if {r['split'] for r in rows}!={'train','val'}:raise ValueError('unexpected split in paired R source')
 xold=np.load(pack/'x_old_geometry.npy',mmap_mode='r',allow_pickle=False);xnew=np.load(pack/'x_corrected_geometry.npy',mmap_mode='r',allow_pickle=False)
 if xold.shape!=(len(rows),16,16,16) or xnew.shape!=xold.shape:raise ValueError('paired R input shape mismatch')
 rng=random.Random(20261016);ids=sorted(rng.sample(range(len(rows)),min(a.n,len(rows))))
 norm=json.loads(a.monthly_norm.read_text());fill=np.asarray(norm['mean'],np.float32)
 with np.load(a.grid_static,allow_pickle=False) as z:lat=np.asarray(z['lat'],np.float32);lon=np.asarray(z['lon'],np.float32);saa=np.asarray(z['saa'],np.float32)
 checks=[]
 for i in ids:
  r=rows[i];t=datetime.fromisoformat(r['target_time_utc']);bjt=t+timedelta(hours=8);rel=f'{t:%Y/%Y%m/%Y%m%d}/{t:%Y%m%d%H%M}.npy';raw=np.load(a.data_root/'data'/rel,mmap_mode='r',allow_pickle=False)
  auxfiles=sorted((a.data_root/'aux_256'/bjt.strftime('%Y%m')).glob(bjt.strftime('%Y%m%d%H%M')+'_*_aux.npz'))
  if len(auxfiles)!=1:raise ValueError(f'aux sidecar count not 1: {auxfiles}')
  rr,cc=COORD[r['station']];sl=np.s_[rr-8:rr+8,cc-8:cc+8]
  physical=np.asarray(raw[np.asarray(CHANNELS),sl[0],sl[1]],np.float32)
  with np.load(auxfiles[0],allow_pickle=False) as z:
   bits=np.asarray(z['agri_valid_bits']);valid=np.stack([((bits[sl]&(1<<k))!=0) for k in CHANNELS])&np.isfinite(physical)
   physical=np.where(valid,physical,fill[:,None,None])
   oldgeo=np.stack([np.asarray(z[k][sl],np.float32) for k in ('cosSOZ','cosRAA','day_mask')])
  soz,soa=compute_solar_angles(lat[sl],lon[sl],t);soa=np.where(soz>=90.0,0.0,soa)
  newgeo=np.stack([np.cos(np.deg2rad(soz)),np.cos(np.deg2rad(soa-saa[sl])),(soz<85.0).astype(np.float32)]).astype(np.float32)
  expected_old=np.concatenate([physical,oldgeo]);expected_new=np.concatenate([physical,newgeo])
  err0=np.abs(np.asarray(xold[i])-expected_old);err1=np.abs(np.asarray(xnew[i])-expected_new)
  if not np.isfinite(err0).all() or not np.isfinite(err1).all():raise ValueError('nonfinite input reconstruction')
  checks.append(dict(array_index=i,split=r['split'],station=r['station'],target_time_utc=r['target_time_utc'],old_max_abs=float(err0.max()),old_agri_max_abs=float(err0[:13].max()),old_geometry_max_abs=float(err0[13:].max()),corrected_max_abs=float(err1.max()),corrected_agri_max_abs=float(err1[:13].max()),corrected_geometry_max_abs=float(err1[13:].max()),source_agri=rel,source_aux=str(auxfiles[0])))
  if err0.max()>2e-5 or err1.max()>2e-5:raise AssertionError(f'live input reconstruction differs: {checks[-1]}')
 out=dict(state='PASS_PAIRED_R_INPUT_RECONSTRUCTION',sampled_rows=len(checks),sample_seed=20261016,channels=CHANNELS,station_coordinates=COORD,all_differences_le_2e_5=True,test_payloads_read=0,source_hashes={str(pack/'rows_supervised.csv'):sha(pack/'rows_supervised.csv'),str(pack/'x_old_geometry.npy'):sha(pack/'x_old_geometry.npy'),str(pack/'x_corrected_geometry.npy'):sha(pack/'x_corrected_geometry.npy'),str(a.monthly_norm):sha(a.monthly_norm)},rows=checks)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
