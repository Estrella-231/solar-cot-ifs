import csv,json,numpy as np
from pathlib import Path
root=Path(__file__).resolve().parents[2]; exp=root/'experiments/monthly_R_station16_20261013_v1'; pack=exp/'results/ghi_head_pack_new_r_v1'; cdir=root/'refine-logs/cot_value_first_20261013'
rows=list(csv.DictReader(open(pack/'rows.csv',encoding='utf-8')));ix={(r['station'],r['target_time_utc']):i for i,r in enumerate(rows) if r['split']=='val'}
p=np.load(cdir/'validation_predictions_monthly_R_seed42.npz');q=np.load(cdir/'clp_reference_monthly_R_seed42.npz');ri=[];gi=[]
for j,(s,t) in enumerate(zip(q['station'].astype(str),q['target_time_utc'].astype(str))):
 if (s,t) in ix:ri.append(j);gi.append(ix[(s,t)])
meta=json.load(open(pack/'pack_manifest.json'));raw=np.asarray(np.load(pack/'cot_log1p.npy',mmap_mode='r')[gi,0],float)*meta['COT_norm_std']+meta['COT_norm_mean'];pred=np.asarray(p['pred_log1p_cot'][ri,0],float)
print('common',len(ri),'missing',len(q['station'])-len(ri),'pack-vs-R mean abs/max',np.mean(np.abs(raw-pred[:,7:10,7:10])),np.max(np.abs(raw-pred[:,7:10,7:10])),'corr',np.corrcoef(raw.ravel(),pred[:,7:10,7:10].ravel())[0,1])
for h in range(6,11):
 for w in range(6,11):
  z=pred[:,h:h+3,w:w+3];print(h,w,round(float(np.mean(np.abs(raw-z))),3),round(float(np.corrcoef(raw.ravel(),z.ravel())[0,1]),3))
print('example',q['station'][ri[0]],q['target_time_utc'][ri[0]],'bank',raw[0].round(2).tolist(),'R center',pred[0,7:10,7:10].round(2).tolist(),'R16',pred[0].round(1).tolist())
