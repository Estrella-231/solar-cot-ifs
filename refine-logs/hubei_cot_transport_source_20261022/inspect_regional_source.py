from pathlib import Path
import numpy as np
import json

root=Path('/home/Data_Pool_3/chenyi/Auxiliary_data/HuBei_AGRI_Preprocessed')
for p in [Path('/home/Data_Pool/zjnu/solar_cot_hubei_20261020/input/grid_static_hubei_witness.npz'),root/'grid_static.npz']:
    z=np.load(p)
    print(str(p), [(k,z[k].shape) for k in z.files],flush=True)
    if 'lat' in z.files:
        print('latlon',float(np.nanmin(z['lat'])),float(np.nanmax(z['lat'])),float(np.nanmin(z['lon'])),float(np.nanmax(z['lon'])),flush=True)
p=root/'data/2025/202508/20250822'
files=list(p.iterdir())
print('day_files', [str(x) for x in files[:3]],flush=True)
allowed=[x for x in files if '202508221200' in x.name]
for p in allowed[:1]:
    z=np.load(p)
    if hasattr(z,'files'): print('sample',str(p),[(k,z[k].shape) for k in z.files],flush=True)
    else:
        print('sample',str(p),z.shape,z.dtype,flush=True)
        g=np.load(root/'grid_static.npz')
        for ch,key in [(15,'saa'),(16,'saz')]:
            print('angle_match',key,float(np.nanmax(np.abs(z[ch]-g[key]))),flush=True)
        for name,lat,lon in [('Niushou',32.08,112.04),('Wangzhai',32.04,112.12),('Mizhuang',32.12,112.20)]:
            ij=np.unravel_index(np.argmin((g['lat']-lat)**2+(g['lon']-lon)**2),(256,256))
            print(name,list(map(int,ij)),float(g['lat'][ij]),float(g['lon'][ij]),int(g['row_index'][ij]),int(g['col_index'][ij]),flush=True)
