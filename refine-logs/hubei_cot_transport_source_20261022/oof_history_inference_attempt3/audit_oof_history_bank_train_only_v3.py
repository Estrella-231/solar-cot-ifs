#!/usr/bin/env python3
"""Join and summarize train-only held-out-R COT maps versus existing full-R bank."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

STAGED_SHA='8b1f95c2a2cf1d66cabd11539b975fe3dccd486146bf3bec17fb18e537f72183'
FOLD_SHA='c86bd6713d5afea1d6864cc4ef6048a93b596051ea4358e7954f39c32b054122'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def stats(x):
 x=np.asarray(x,dtype=np.float64); x=x[np.isfinite(x)]
 if not x.size:return {'n':0}
 return {'n':int(x.size),'mean':float(x.mean()),'std':float(x.std()),'p50':float(np.quantile(x,.5)),'p90':float(np.quantile(x,.9)),'p99':float(np.quantile(x,.99)),'max':float(x.max())}
def main():
 p=argparse.ArgumentParser();p.add_argument('--staged-rows',type=Path,required=True);p.add_argument('--fold-map',type=Path,required=True);p.add_argument('--full-bank',type=Path,required=True);p.add_argument('--oof-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 if sha(a.staged_rows)!=STAGED_SHA or sha(a.fold_map)!=FOLD_SHA:raise ValueError('frozen manifest hash mismatch')
 staged=read(a.staged_rows); bank=read(a.full_bank/'rows.csv')
 train=[r for r in staged if r['split']=='train']
 bankmap={(r['time_bjt'],r['split']):int(r['index']) for r in bank}
 if len(bankmap)!=len(bank) or set(bankmap)!={(r['time_bjt'],r['split']) for r in staged}:raise ValueError('full-R bank key support does not exactly match staged rows')
 br=json.loads((a.full_bank/'BANK_REPORT.json').read_text())
 if br.get('test_payloads_read')!=0 or br.get('train_predictions_are_oof') is not False:raise ValueError('unexpected full-R bank provenance')
 full=np.load(a.full_bank/'cot_pred_trainval_full.npy',mmap_mode='r',allow_pickle=False)
 # Keep this audit strictly train-only: the full bank also contains validation predictions.
 # A global finite-value scan would touch those validation outputs, so validate only
 # the train rows when they are selected below.
 if full.shape not in ((len(bank),1,16,16),(len(bank),16,16)):raise ValueError(f'full bank shape invalid {full.shape}')
 if full.ndim==4:full=full[:,0]
 oof={}; reports=[]
 for f in range(5):
  d=a.oof_root/f'fold_{f}'; rr=read(d/'rows.csv'); c=np.load(d/'cot_oof.npy',mmap_mode='r',allow_pickle=False); day=np.load(d/'day_mask.npy',mmap_mode='r',allow_pickle=False); rep=json.loads((d/'fold_report.json').read_text())
  if rep.get('state')!='COMPLETE_TRAIN_DATE_HELDOUT_HISTORICAL_COT_INFERENCE' or any(rep.get(k)!=0 for k in ('test_payloads_read','buffer_payloads_read','cpp_target_or_quality_payloads_read','ghi_labels_read')):raise ValueError(f'fold {f} receipt invalid')
  if c.shape!=(len(rr),16,16) or day.shape!=c.shape or not np.isfinite(c).all():raise ValueError(f'fold {f} arrays mismatch')
  for i,r in enumerate(rr):
   if r['split']!='train' or int(r['heldout_fold'])!=f or r['time_bjt'] in oof:raise ValueError(f'fold {f} row duplication/split mismatch')
   oof[r['time_bjt']]=(c[i],day[i],f)
  reports.append(rep)
 if len(oof)!=20206 or set(oof)!={r['time_bjt'] for r in train}:raise ValueError('OOF train keys incomplete or extra')
 by_date=[]; all_common=0; outfull=0; outoof=0; belowfull=0; belowoof=0; daytotal=0;delta_parts=[];a_parts=[];b_parts=[]; full_day_parts=[];oof_day_parts=[]; train_bank_hasher=hashlib.sha256()
 for r in train:
  t=r['time_bjt']; fi=bankmap[(t,'train')]; a0=np.asarray(full[fi],dtype=np.float64); train_bank_hasher.update(a0.tobytes(order='C')); b0,day,f=oof[t]; b0=np.asarray(b0,dtype=np.float64); day=np.asarray(day,dtype=bool)
  if day.shape!=(16,16) or not np.isfinite(a0).all():raise ValueError(f'bad day/full map {t}')
  outfull+=int(((a0>100)&day).sum());outoof+=int(((b0>100)&day).sum());belowfull+=int(((a0<0)&day).sum());belowoof+=int(((b0<0)&day).sum());daytotal+=int(day.sum());full_day_parts.append(a0[day]);oof_day_parts.append(b0[day])
  m=day & (a0>=0)&(a0<=100)&(b0>=0)&(b0<=100)&np.isfinite(a0)&np.isfinite(b0)
  av=a0[m];bv=b0[m]
  if av.size:
   d=bv-av; all_common+=int(av.size);delta_parts.append(d);a_parts.append(av);b_parts.append(bv)
  by_date.append({'time_bjt':t,'date_bjt':t[:8],'fold':f,'day_pixels':int(day.sum()),'common_inrange_pixels':int(m.sum()),
    'full_r_day_mean':float(a0[day].mean()) if day.any() else None,'oof_r_day_mean':float(b0[day].mean()) if day.any() else None,'full_r_common_inrange_mean':float(av.mean()) if av.size else None,'oof_r_common_inrange_mean':float(bv.mean()) if bv.size else None,'full_r_day_min':float(a0[day].min()) if day.any() else None,'full_r_day_max':float(a0[day].max()) if day.any() else None,'oof_r_day_min':float(b0[day].min()) if day.any() else None,'oof_r_day_max':float(b0[day].max()) if day.any() else None,
    'common_inrange_MAE':float(np.abs(bv-av).mean()) if av.size else None,'common_inrange_RMSE':float(np.sqrt(np.mean((bv-av)**2))) if av.size else None,
    'full_r_day_gt100':int(((a0>100)&day).sum()),'oof_r_day_gt100':int(((b0>100)&day).sum())})
 da=np.concatenate(delta_parts) if delta_parts else np.empty(0,dtype=np.float64)
 av=np.concatenate(a_parts) if a_parts else np.empty(0,dtype=np.float64)
 bv=np.concatenate(b_parts) if b_parts else np.empty(0,dtype=np.float64)
 abs_d=np.abs(da)
 corr=float(np.corrcoef(av,bv)[0,1]) if len(av)>1 and av.std()>0 and bv.std()>0 else None
 flat={}
 for k,v in __import__('itertools').groupby(sorted(by_date,key=lambda x:x['date_bjt']),key=lambda x:x['date_bjt']):flat[k]=list(v)
 date_summary=[]
 for date,v in flat.items():
  vals=[x for x in v if x['common_inrange_MAE'] is not None]
  date_summary.append({'date_bjt':date,'frames':len(v),'day_pixels':sum(x['day_pixels'] for x in v),'common_inrange_pixels':sum(x['common_inrange_pixels'] for x in v), 'full_r_common_inrange_mean':float(np.average([x['full_r_common_inrange_mean'] for x in v if x['common_inrange_pixels']],weights=[x['common_inrange_pixels'] for x in v if x['common_inrange_pixels']])) if any(x['common_inrange_pixels'] for x in v) else None,'oof_r_common_inrange_mean':float(np.average([x['oof_r_common_inrange_mean'] for x in v if x['common_inrange_pixels']],weights=[x['common_inrange_pixels'] for x in v if x['common_inrange_pixels']])) if any(x['common_inrange_pixels'] for x in v) else None,
    'mean_frame_MAE':float(np.mean([x['common_inrange_MAE'] for x in vals])) if vals else None,
    'mean_frame_RMSE':float(np.mean([x['common_inrange_RMSE'] for x in vals])) if vals else None})
 result={'state':'COMPLETE_OOF_HISTORY_INPUT_DISTRIBUTION_DIAGNOSTIC','interpretation':'exploratory input-source consistency only; not COT skill, forecast gain, or GHI result',
  'train_rows':len(train),'oof_rows':len(oof),'validation_rows_inferred':0,'test_payloads_read':0,'buffer_payloads_read':0,'cpp_target_or_quality_payloads_read':0,'ghi_labels_read':0,
  'staged_rows_sha256':sha(a.staged_rows),'fold_map_sha256':sha(a.fold_map),'full_r_bank_train_rows_prediction_sha256_fp64':train_bank_hasher.hexdigest(),
  'full_r_bank_rows_sha256':sha(a.full_bank/'rows.csv'),'full_r_bank_train_is_oof':False,'oof_rows_sha256':{f:sha(a.oof_root/f'fold_{f}'/'rows.csv') for f in range(5)},
  'common_daytime_inrange_pixels':all_common,'daytime_pixels':daytotal,'full_r_gt100_day_pixel_fraction':outfull/daytotal if daytotal else None,
  'oof_r_gt100_day_pixel_fraction':outoof/daytotal if daytotal else None,'full_r_below0_day_pixel_fraction':belowfull/daytotal if daytotal else None,'oof_r_below0_day_pixel_fraction':belowoof/daytotal if daytotal else None,'full_r_day_raw_distribution':stats(np.concatenate(full_day_parts)),'oof_r_day_raw_distribution':stats(np.concatenate(oof_day_parts)),
  'common_inrange_full_r_distribution':stats(av),'common_inrange_oof_r_distribution':stats(bv),
  'OOF_minus_full_R_common_inrange':{'MAE':float(abs_d.mean()) if len(abs_d) else None,'RMSE':float(np.sqrt(np.mean(da**2))) if len(da) else None,'median_abs':float(np.quantile(abs_d,.5)) if len(abs_d) else None,'p95_abs':float(np.quantile(abs_d,.95)) if len(abs_d) else None,'pearson_r':corr},
  'fold_runtime_receipts':reports,'notes':['Common support is daytime and both predictions in [0,100].','Out-of-range values were counted, not clipped.','No CPP reference targets or GHI labels were used.']}
 a.out.parent.mkdir(parents=True,exist_ok=False);a.out.write_text(json.dumps(result,indent=2)+'\n')
 with (a.out.parent/'per_frame.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(by_date[0]));w.writeheader();w.writerows(by_date)
 with (a.out.parent/'per_date.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(date_summary[0]));w.writeheader();w.writerows(date_summary)
 print(json.dumps({'state':result['state'],'train_rows':len(train),'common_daytime_inrange_pixels':all_common,'MAE':result['OOF_minus_full_R_common_inrange']['MAE'],'RMSE':result['OOF_minus_full_R_common_inrange']['RMSE'],'test_payloads_read':0,'buffer_payloads_read':0},indent=2))
if __name__=='__main__':main()
