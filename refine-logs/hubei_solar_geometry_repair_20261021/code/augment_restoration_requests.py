"""Add independent historical C13 witnesses to exact restoration requests."""
import csv,json,hashlib,argparse
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
p=argparse.ArgumentParser();p.add_argument('preparation',type=Path);p.add_argument('old_stage',type=Path);a=p.parse_args();o=a.preparation/'restore_requests_with_witness';o.mkdir(exist_ok=False);rr=list(csv.DictReader((a.preparation/'metadata/validation_source_index.csv').open()));rpack=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_cloud_probability_20261020/R_pack_correctutc_attempt2');rs=list(csv.DictReader((rpack/'rows.csv').open()));rx=np.load(a.preparation/'R_raw_pending_restoration.npy',mmap_mode='r');lut={(datetime.strptime(r['time_bjt'],'%Y%m%d%H%M')-timedelta(hours=8)).strftime('%Y%m%d%H%M'):i for i,r in enumerate(rs)};out=[]
for f in sorted((a.preparation/'restore_requests').glob('*.npz')):
 z=np.load(f);y,x=z['y'].astype(int),z['x'].astype(int);src=a.old_stage/'data'/f'{f.stem}.npy'
 if src.is_file():ir=np.load(src,mmap_mode='r')[12,y,x];source=str(src)
 else:
  assert f.stem in lut and ((y>=103)&(y<119)&(x>=118)&(x<134)).all();ir=rx[lut[f.stem],10,y-103,x-118];source='old R pack C13 decoded with source train-only norm'
 assert np.isfinite(ir).all();np.savez_compressed(o/f.name,y=y.astype(np.uint16),x=x.astype(np.uint16),reference_ir13=ir.astype(np.float32));out.append(dict(utc=f.stem,witness_source=source,pixels=len(y),augmented_request_sha256=hashlib.sha256((o/f.name).read_bytes()).hexdigest()))
(a.preparation/'RESTORATION_REQUEST_WITNESS.json').write_text(json.dumps(dict(state='ALL_RESTORATION_REQUESTS_HAVE_C13_WITNESS',frames=len(out),records=out),indent=2));print('RESTORATION_WITNESS_REQUESTS',len(out))
