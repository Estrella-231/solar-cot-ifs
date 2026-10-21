from pathlib import Path
import csv,hashlib,json
import numpy as np
root=Path(__file__).parent
out=root/'evidence'/'remote_215541'/'oof_train'
ev=out/'evidence'/'independent_train_only_v3'
s=json.loads((ev/'summary.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
rows=[]; common_n=0; mae_num=0.0; mse_num=0.0; full_sum=0.0; oof_sum=0.0
for f in range(5):
 d=out/f'fold_{f}'; rep=json.loads((d/'fold_report.json').read_text())
 with (d/'rows.csv').open(encoding='utf-8-sig',newline='') as fp: rr=list(csv.DictReader(fp))
 c=np.load(d/'cot_oof.npy',mmap_mode='r',allow_pickle=False); day=np.load(d/'day_mask.npy',mmap_mode='r',allow_pickle=False)
 assert rep['fold']==f and rep['rows']==len(rr)==c.shape[0]==day.shape[0]
 assert rep['state']=='COMPLETE_TRAIN_DATE_HELDOUT_HISTORICAL_COT_INFERENCE'
 assert all(rep[k]==0 for k in ('test_payloads_read','buffer_payloads_read','cpp_target_or_quality_payloads_read','ghi_labels_read'))
 assert sha(d/'rows.csv')==s['oof_rows_sha256'][str(f)]==rep['prediction_rows_sha256'] if 'prediction_rows_sha256' in rep else sha(d/'rows.csv')==s['oof_rows_sha256'][str(f)]
 assert np.isfinite(c).all() and c.shape==(len(rr),16,16) and day.shape==c.shape
 assert all(x['split']=='train' and int(x['heldout_fold'])==f for x in rr)
 rows.extend((x['time_bjt'],f) for x in rr)
 # OOF per-day bounds from the actual train-only shard.
 m=day.astype(bool)
 assert int(((c>100)&m).sum())==rep['out_of_range_day_pixels_gt100']
with (ev/'per_frame.csv').open(encoding='utf-8-sig',newline='') as fp: pf=list(csv.DictReader(fp))
assert len(pf)==20206 and len(rows)==20206
assert len({x[0] for x in rows})==20206
assert {x[0] for x in rows}=={x['time_bjt'] for x in pf}
for x in pf:
 n=int(x['common_inrange_pixels'])
 if n:
  mae_num+=n*float(x['common_inrange_MAE'])
  mse_num+=n*float(x['common_inrange_RMSE'])**2
  full_sum+=n*float(x['full_r_common_inrange_mean'])
  oof_sum+=n*float(x['oof_r_common_inrange_mean'])
  common_n+=n
mae=mae_num/common_n; rmse=(mse_num/common_n)**0.5
assert common_n==s['common_daytime_inrange_pixels']
assert abs(mae-s['OOF_minus_full_R_common_inrange']['MAE'])<1e-10
assert abs(rmse-s['OOF_minus_full_R_common_inrange']['RMSE'])<1e-10
assert abs(full_sum/common_n-s['common_inrange_full_r_distribution']['mean'])<1e-10
assert abs(oof_sum/common_n-s['common_inrange_oof_r_distribution']['mean'])<1e-10
print(json.dumps({'state':'PASS_LOCAL_ARTIFACT_HASH_KEY_AND_FP64_REAGGREGATION_AUDIT','train_rows':len(rows),'fold_counts':[sum(1 for _,ff in rows if ff==f) for f in range(5)],'common_inrange_pixels':common_n,'recomputed_MAE':mae,'recomputed_RMSE':rmse,'recomputed_full_mean':full_sum/common_n,'recomputed_oof_mean':oof_sum/common_n,'test_payloads_read':s['test_payloads_read'],'buffer_payloads_read':s['buffer_payloads_read']},indent=2))
