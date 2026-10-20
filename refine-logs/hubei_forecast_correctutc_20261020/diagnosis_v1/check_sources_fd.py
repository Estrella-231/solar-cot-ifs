from pathlib import Path
import numpy as np,json
from datetime import datetime,timedelta
import csv
r=Path('/home/Data_Pool_3/chenyi/Auxiliary_data/HuBei_AGRI_Preprocessed');o=Path('/home/Data_Pool/zjnu/hubei_cot_independent_20261016/source_pair_probe_20261020');o.mkdir(exist_ok=False)
records=[]
bank=Path('/home/Data_Pool/zjnu/hubei_cot_independent_20261016/forecast_bank/strict24_same_day_v3_best_epoch10_fulltrainval_20261016/val')
keys=list(csv.DictReader((bank/'keys.csv').open()));bt=np.load(bank/'true_agri_patch.npy',mmap_mode='r');channel=[0,1,2,3,4,5,8,9,10,11,12,13,14]
for bjt in ('202511060900','202508230900'):
 p=list((r/'data16_label'/bjt[:6]).glob(bjt+'_*.npz'));assert len(p)==1
 with np.load(p[0],allow_pickle=False) as z:a=z['patch'][:15].copy()
 for offset in (-8,0,8):
  dt=datetime.strptime(bjt,'%Y%m%d%H%M')+timedelta(hours=offset);t=dt.strftime('%Y%m%d%H%M');full=r/'data'/t[:4]/t[:6]/t[:8]/(t+'.npy')
  if not full.exists():continue
  raw=np.load(full,allow_pickle=False,mmap_mode='r');v=raw[:15,103:119,118:134];err=abs(v-a)
  seq,li=next((i,key['target_times_bjt'].split('|').index(bjt)) for i,key in enumerate(keys) if bjt in key['target_times_bjt'].split('|'))
  bankerr=abs(bt[seq,li]-v[channel]);rec=dict(bjt=bjt,offset_hours=offset,source=str(full),shape=list(raw.shape),same_crop_max_abs=float(err.max()),bank_truth_vs_full_max_abs=float(bankerr.max()),bank_sequence=seq,bank_lead_index=li,max_by_channel=err.max((1,2)).tolist());records.append(rec)
  np.savez_compressed(o/f'{bjt}_offset{offset}.npz',patch16=a,full_crop=v,full_C13=np.asarray(raw[12]),saved_bank_truth=bt[seq,li])
  if offset==-8:
   # Single physical frame; search crop origin independently using channel C13.
   vfull=np.asarray(raw[12]);best=(1e20,None,None)
   for y in range(241):
    for x in range(241):
     d=float(np.mean((vfull[y:y+16,x:x+16]-a[12])**2))
     if d<best[0]:best=(d,y,x)
   rec['best_C13_crop_mse_row_col']=best
(o/'report.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
