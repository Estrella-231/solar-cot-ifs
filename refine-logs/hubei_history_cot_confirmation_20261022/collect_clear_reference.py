"""Small train/validation geometry reference only; no GHI or test values."""
import csv,json
from datetime import datetime
from pathlib import Path
import numpy as np
ROOT=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments')
pack=ROOT/'hubei_solar_geometry_repair_20261021/GHI_pack_attempt12'
rows=list(csv.DictReader((pack/'rows.csv').open()));clear=np.load(pack/'clear_sky_wm2.npy',mmap_mode='r')
assert len(rows)==len(clear);out=[];seen=set()
for i,r in enumerate(rows):
    assert r['split'] in ['train','val']
    t=datetime.strptime(r['target_time_bjt'],'%Y%m%d%H%M');key=(r['station'],t.strftime('%Y%m'))
    if key in seen:continue
    seen.add(key);out.append(dict(station=r['station'],target_time_BJT=t.isoformat(' '),clear_sky_wm2=float(clear[i])))
dest=ROOT/'hubei_history_cot_confirmation_20261022/CLEAR_SKY_REFERENCE.json'
dest.write_text(json.dumps(out,indent=2)+'\n');print(len(out))
