import json,numpy as np
from pathlib import Path
H=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_forecast_cot_increment_20261022')
E=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_history_cot_confirmation_20261022/diagnosis_20261023')
out={}
for st in ['Niushou','Wangzhai','Mizhuang']:
    p=H/'prepared_attempt1'/st;roles=np.load(p/'roles.npy');co=np.load(p/'cot.npy',mmap_mode='r');zero=np.all(co==0,axis=(1,2,3,4));out[st]={}
    for role in ['F_fit','F_select','val']:
        m=roles==role;out[st][role]=dict(rows=int(m.sum()),all_standardized_zero_cot_rows=int(zero[m].sum()),fraction=float(zero[m].mean()))
(E/'OLD_MASK_DISTRIBUTION.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
