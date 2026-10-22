import csv,json,numpy as np
from pathlib import Path
B=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments');P=B/'hubei_cot_regional_input_20261023';R=B/'hubei_solar_geometry_repair_20261021/prepared_attempt12/R_packs/full';rows=list(csv.DictReader((R/'rows.csv').open()));assert all(r['split'] in ['train','val'] for r in rows)
t=np.load(R/'target.npy',mmap_mode='r');m=np.load(R/'mask.npy',mmap_mode='r');np.savez_compressed(P/'old_CPP_reference.npz',time_bjt=np.array([r['time_bjt'] for r in rows]),split=np.array([r['split'] for r in rows]),cot=np.asarray(t)*100.,mask=np.asarray(m,dtype=bool))
print(len(rows),t.shape)
