import csv,json
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
import torch
S=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907')
E=S/'experiments/regional_station_cot_20260925'
B=E/'sili_cot_dynamics_bank_train512_val512_20260927_v1'
D=S/'data/head_pack_trainval_20260908_v1'
O=S/'experiments/history_cot_forecast_ghi_20261015_v1'
days={r['date_BJT'] for r in csv.DictReader((O/'day_assignment.csv').open()) if r['split'] in ('train','val')}
allow={s:set() for s in ('train','val')}
rows=list(csv.DictReader((D/'rows.csv').open()))
for r in rows:
    if r['station']!='sili':continue
    t=datetime.fromisoformat(r['init_time_bjt'])
    if {str((t+timedelta(minutes=k*15)).date()) for k in range(-7,17)}<=days:allow[{'validation':'val'}.get(r['split'],r['split'])].add(int(r['cache_index']))
summary={}
for s,ib in [('train','regional_ghi_pilot_bank_20260927_v1'),('val','regional_ghi_pilot_bank_val512_20260927_v1')]:
    ids=np.load(B/s/'sequence_indices.npy');selected=ids[np.isin(ids,list(allow[s]))]
    ii=np.load(E/ib/s/'sequence_indices.npy');assert set(selected)<=set(ii)
    rid=np.load(B/s/'head_row_ids.npy');seq=np.load(D/'sequence.npy',mmap_mode='r')[rid];st=np.load(D/'station.npy',mmap_mode='r')[rid]
    rr=rid[np.isin(seq,selected)&(st==0)];assert len(rr)>0
    allseq=np.load(D/'sequence.npy',mmap_mode='r')
    for x in rr:
        r=rows[int(x)];assert int(r['cache_index'])==int(allseq[x])
    summary[s]=dict(source_sequences=len(ids),allowed_sequences=len(selected),ghi_rows=len(rr))
c=torch.load(E/'sili_full_vs_repeat_last_seed42_20260928_v5/full_history/best_learned.pt',map_location='cpu',weights_only=False)
assert c['update']==96
summary['state']='PASS_KEYS_NO_NEW_TEST_BUFFER_WINDOWS';summary['test_payload_read']=False
(O/'preflight.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
