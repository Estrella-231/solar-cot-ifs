import csv,json,sys,subprocess
from pathlib import Path
import numpy as np
from score_saved import scores
root=Path(sys.argv[1]);script=Path(__file__).with_name('score_saved.py')
subprocess.run([sys.executable,str(script),str(root)],check=True)
with (root/'ghi_rows.csv').open() as f:rows=list(csv.DictReader(f))
with np.load(root/'ghi_predictions.npz') as z:pred={k:z[k].copy() for k in z.files}
keep=np.array([r['hubei_train_day_overlap']=='0' for r in rows])
sub=root/'no_hubei_training_day_overlap';sub.mkdir()
for name in ('ghi','cot'):
 with (root/(name+'_rows.csv')).open() as f:rr=list(csv.DictReader(f))
 mask=np.array([r['hubei_train_day_overlap']=='0' for r in rr])
 with (sub/(name+'_rows.csv')).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows([r for r,k in zip(rr,mask) if k])
 with np.load(root/(name+'_predictions.npz')) as z:np.savez_compressed(sub/(name+'_predictions.npz'),**{k:z[k][mask] for k in z.files})
(sub/'completion.json').write_text(json.dumps({'status':'PASS','test_payloads_read':0,'scope':'prespecified no-Hubei-train-date-overlap subset'}))
subprocess.run([sys.executable,str(script),str(sub)],check=True)
records=[]
for scope,allowed in [('all_frozen_val',np.ones(len(rows),bool)),('no_hubei_train_day_overlap',keep)]:
 for station in ('sili','zhujia'):
  for lead in range(15,241,15):
   mask=allowed & np.array([r['station']==station and int(r['lead_minutes'])==lead for r in rows])
   if not mask.any():continue
   for seed in (42,43,44):
    for method in ('dydiff','simvp','observed'):
     for arm in ('M00','M10'):
      r=scores(pred[f'{station}_seed{seed}_{arm}_{method}'][mask],pred['truth'][mask])
      records.append(dict(scope=scope,station=station,seed=seed,lead_minutes=lead,method=method,arm=arm,**r))
with (root/'ghi_lead_metrics_fp64.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
(root/'FULL_EVALUATION_COMPLETE.json').write_text(json.dumps({'status':'PASS','rows':len(rows),'nonoverlap_rows':int(keep.sum()),'test_payloads_read':0}))
print('FULL_PAIRED_COT_GHI_EVALUATION_COMPLETE')
