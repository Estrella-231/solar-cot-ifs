import json,csv,numpy as np
from pathlib import Path
H=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_forecast_cot_increment_20261022');E=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_history_cot_confirmation_20261022/diagnosis_20261023')
out={}
for st in ['Niushou','Wangzhai','Mizhuang']:
 p=H/'prepared_attempt1'/st; roles=np.load(p/'roles.npy');co=np.load(p/'cot.npy',mmap_mode='r');hist=np.load(p/'history.npy',mmap_mode='r'); valid=hist[:,:,13:14]>np.cos(np.deg2rad(80.)); validmean=valid.mean(axis=(1,2,3,4)); rows=list(csv.DictReader((p/'rows.csv').open()));hours=np.array([int(r['init_time_bjt'][8:10])+int(r['init_time_bjt'][10:12])/60 for r in rows]);out[st]={}
 for role in ['F_fit','F_select','val']:
  m=roles==role;out[st][role]=dict(rows=int(m.sum()),all_invalid=int((validmean[m]==0).sum()),partial_valid=int(((validmean[m]>0)&(validmean[m]<1)).sum()),all_valid=int((validmean[m]==1).sum()),init_hour_range=[float(hours[m].min()),float(hours[m].max())],before08=int((hours[m]<8).sum()),cot_absmax=float(abs(co[m]).max()))
(E/'OLD_MASK_DISTRIBUTION_DETAIL.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
