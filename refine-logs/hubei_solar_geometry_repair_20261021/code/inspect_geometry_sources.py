import sys,json,csv
from pathlib import Path
import numpy as np
mode=sys.argv[1]
if mode=='fd':
 import netCDF4 as nc
 p=Path('/home/Data_Pool/data/FY/FY4B/AGRI/FY4B_AGRI_SAZ_SOZ_SAA_SOA_202403180000_4km.nc')
 with nc.Dataset(p) as z:g={k:np.asarray(z[v][1112:1368,2077:2333]) for k,v in [('lat','lat'),('lon','lon'),('saa','SAA'),('saz','SAZ')]}
 for k,n in [('row_index','LAT_4DEG_INDEX_105.npy'),('col_index','LON_4DEG_INDEX_105.npy')]:g[k]=np.load(p.parent/n,mmap_mode='r')[1112:1368,2077:2333].copy()
 o=Path(sys.argv[2]);np.savez_compressed(o,**g);print('GRID_EXPORT',o,{k:(v.shape,str(v.dtype)) for k,v in g.items()},flush=True)
 for name,(y,x) in {'Niushou':(111,124),'Wangzhai':(112,126),'Mizhuang':(110,128)}.items():print(name,{k:float(v[y,x]) for k,v in g.items()})
else:
 p=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/hubei_independent_trainval_stage_20261017');d=next((p/'data').glob('*.npy'));a=np.load(d,mmap_mode='r');f=next((p/'aux').glob('*.npz'));z=np.load(f);print('SOURCE',d,a.shape,a.dtype,f,{k:(z[k].shape,str(z[k].dtype)) for k in z.files})
 rp=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_cloud_probability_20261020/R_pack_correctutc_attempt2');s=json.loads((rp/'pack_status.json').read_text());print('RSTATUS',json.dumps(s));print('RROW',next(csv.DictReader((rp/'rows.csv').open())))
 env=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/hubei_cot_independent_20261017/.venv_simvp/bin/python');print('SIMVP_ENV_EXISTS',env.exists())
