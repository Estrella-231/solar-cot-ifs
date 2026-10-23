import csv,json
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent/'hubei_matched_R_dynamics_20261023';B=P.parent/'hubei_solar_geometry_repair_20261021';H=P.parent/'hubei_forecast_cot_increment_20261022';scope=json.loads((P/'CASE_SCOPE.json').read_text());init=scope['init']
rows=list(csv.DictReader((H/'prepared_attempt1/Niushou/rows.csv').open()));matched=[r for r in rows if r['init_time_bjt']==init];assert matched and len({(r['source_role'],r['source_sequence_index']) for r in matched})==1;r=matched[0];assert r['split']=='train';role=r['source_role'];seq=int(r['source_sequence_index'])
bank=B/'S_banks_attempt12'/role/'val';keys=list(csv.DictReader((bank/'keys.csv').open()));assert keys[seq]['init_time_bjt']==init
t=datetime.strptime(init,'%Y%m%d%H%M');times=[(t+timedelta(minutes=k)).strftime('%Y%m%d%H%M') for k in range(0,241,15)];assert keys[seq]['target_times_bjt'].split('|')==times[1:]
frames=list(csv.DictReader((R/'raw_attempt1/frames.csv').open()));lookup={r['bjt']:r for r in frames};raw=np.load(R/'raw_attempt1/raw16.npy',mmap_mode='r');observed=np.array(raw[[int(lookup[v]['frame_index']) for v in times]])
with np.load(R/'reference_attempt1'/role/'reference.npz') as z:ref={k:z[k] for k in z.files}
ii={str(t):i for i,t in enumerate(ref['bjt'])};actual_cot=ref['cot'][[ii[v] for v in times]];actual_valid=ref['valid'][[ii[v] for v in times]]
forecast=np.array(np.load(bank/'pred_agri_patch.npy',mmap_mode='r')[seq]);predcot=np.array(np.load(B/'COT_banks_attempt12'/f'{role}.npy',mmap_mode='r')[seq])
np.savez_compressed(P/'CASE_MATERIAL.npz',init=init,times=times,source_role=role,observed_AGRI=observed,forecast_AGRI=forecast,actual_R_COT=actual_cot,actual_valid=actual_valid,forecast_R_COT=predcot)
print(init,role,seq,'case material exported')
