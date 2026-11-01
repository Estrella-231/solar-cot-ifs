#!/usr/bin/env python3
"""Cross-fit a station/lead temporal-increment COT calibration on train OOF data."""
from __future__ import annotations
import argparse,csv,hashlib,json
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np

STATIONS=("sili","zhujia");LEADS=tuple(range(15,241,15));NEIGHBOR=np.ones((3,3),bool);NEIGHBOR[1,1]=False
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def timekey(s):return datetime.strptime(s,'%Y-%m-%d %H:%M:%S')
def fmt(t):return t.strftime('%Y-%m-%d %H:%M:%S')
def main():
 p=argparse.ArgumentParser();p.add_argument('--forecast-pack',type=Path,required=True);p.add_argument('--actual-r-pack',type=Path,required=True);p.add_argument('--bank-index',type=Path,required=True);p.add_argument('--fold-root',type=Path,required=True);p.add_argument('--center-calibration',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=2000);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 fm=json.loads((a.forecast_pack/'pack_manifest.json').read_text());rm=json.loads((a.actual_r_pack/'pack_manifest.json').read_text());cc=json.loads(a.center_calibration.read_text())
 if fm.get('state')!='COMPLETE_CAUSAL_FUTURE_GHI_HEAD_PACK' or fm.get('train_forecast_oof') is not True or fm.get('forecast_bank_oof_fold_count')!=5:raise ValueError('five-fold OOF forecast pack required')
 if fm.get('test_payload_opened') is not False or fm.get('buffer_payload_opened') is not False or fm.get('future_observed_agri_used_as_input') is not False or fm.get('cpp_used_as_input') is not False:raise ValueError('forecast pack causal/test closure failed')
 if rm.get('state')!='COMPLETE' or rm.get('split_id')!=fm.get('split_id') or rm.get('test_payload_opened') is not False:raise ValueError('actual-R pack source/split mismatch')
 if cc.get('state')!='COMPLETE_CROSSFIT_TRAIN_ONLY_OOF_FUTURE_COT_RELIABILITY_CALIBRATION' or cc.get('forecast_pack_manifest_sha256')!=sha(a.forecast_pack/'pack_manifest.json'):raise ValueError('center reliability calibration mismatch')
 seqidx={}
 with a.bank_index.open(newline='') as f:
  for r in csv.DictReader(f):
   if r['split']=='train':
    ix=int(r['sequence_index_within_split']);
    if ix in seqidx:raise ValueError('duplicate train bank index')
    seqidx[ix]=r['seq_id']
 seqfold={};foldhash={}
 for k in range(5):
  q=a.fold_root/f'fold_{k}'/'fold_oof_ids.txt';foldhash[str(k)]=sha(q)
  for sid in q.read_text().splitlines():
   if sid in seqfold:raise ValueError('sequence assigned to multiple folds')
   seqfold[sid]=k
 if set(seqfold)!=set(seqidx.values()):raise ValueError('fold manifests do not exactly cover training OOF bank')
 actual_rows=list(csv.DictReader((a.actual_r_pack/'rows.csv').open(newline='')));act={}
 for i,r in enumerate(actual_rows):
  if r['split']=='train':
   k=(r['station'],r['target_time_utc'])
   if k in act:raise ValueError('duplicate actual-R train timestamp')
   act[k]=i
 keys=list(csv.DictReader((a.forecast_pack/'train'/'keys.csv').open(newline='')))
 idx=[];currref=[];prevref=[];folds=[];days=[];stlist=[];leads=[];fold_by_row=[];seen=set()
 for i,r in enumerate(keys):
  if r['split']!='train':raise ValueError('nontrain key in train rows')
  st=r['station'];lead=int(r['lead_minutes']);kt=(st,r['init_time_utc'],r['target_time_utc'],lead,int(r['ghi_row_index']))
  if kt in seen:raise ValueError('duplicate GHI train key')
  seen.add(kt)
  cr=act.get((st,r['target_time_utc']));pr=act.get((st,fmt(timekey(r['target_time_utc'])-timedelta(minutes=15))))
  if cr is None:raise ValueError('target train COT reference absent')
  si=int(r['sequence_index_within_split']);f=seqfold.get(seqidx.get(si,''))
  if f is None:raise ValueError('training key lacks OOF fold')
  idx.append(i);currref.append(cr);prevref.append(-1 if pr is None else pr);folds.append(f);days.append((timekey(r['target_time_utc'])+timedelta(hours=8)).strftime('%Y-%m-%d'));stlist.append(st);leads.append(lead)
 if len(keys)!=fm.get('rows',{}).get('train') or len(keys)!=714914:raise ValueError('unexpected GHI train key count')
 pred=np.load(a.forecast_pack/'train'/'cot_log1p.npy',mmap_mode='r',allow_pickle=False);truth=np.load(a.actual_r_pack/'cot_log1p.npy',mmap_mode='r',allow_pickle=False)
 if pred.shape!=(len(keys),3,1,3,3) or pred.dtype!=np.float32:raise ValueError('forecast COT array shape/dtype mismatch')
 if truth.shape!=(len(actual_rows),1,3,3) or truth.dtype!=np.float32:raise ValueError('actual-R COT shape/dtype mismatch')
 if sha(a.forecast_pack/'train'/'cot_log1p.npy')!=fm['array_sha256']['train']['cot_log1p.npy'] or sha(a.actual_r_pack/'cot_log1p.npy')!=rm['array_sha256']['cot_log1p.npy'] or sha(a.actual_r_pack/'rows.csv')!=rm['array_sha256']['rows.csv']:raise ValueError('COT source hash mismatch')
 groups={}
 prepared={}
 for st in STATIONS:
  for ld in LEADS:
   for f in range(5):groups[(st,ld,f)]=[]
 for i,(st,ld,f,cr,pr) in enumerate(zip(stlist,leads,folds,currref,prevref)):
  if pr>=0:groups[(st,ld,f)].append(i)
 sums={};prepared={};full_sums={}
 for (st,ld,f),vals in groups.items():
  ix=np.asarray(vals,np.int64)
  if not len(ix):raise ValueError(f'no exact adjacent-frame training pairs: {st}/{ld}/fold{f}')
  ri=np.asarray(currref,np.int64)[ix];pi=np.asarray(prevref,np.int64)[ix]
  yc=np.asarray(truth[ri,0],np.float64);yp=np.asarray(truth[pi,0],np.float64)
  rawc=np.asarray(pred[ix,2,0],np.float64);rawp=np.asarray(pred[ix,1,0],np.float64)
  pc=cc['parameters_excluding_each_fold'][str(f)][st][str(ld)];pt=pc['intercept']+pc['slope']*rawc
  pld=ld-15
  if pld>0:
   ppcoef=cc['parameters_excluding_each_fold'][str(f)][st][str(pld)];pp=ppcoef['intercept']+ppcoef['slope']*rawp
  else:pp=rawp
  dc=pt[:,1,1]-pp[:,1,1];dy=yc[:,1,1]-yp[:,1,1]
  sx=float(np.sum(dc*dy,dtype=np.float64));ss=float(np.sum(dc*dc,dtype=np.float64));sums[(st,ld,f)]=(sx,ss)
  prepared[(st,ld,f)]=(ix,pt,pp,yc,yp)
  # Validation uses full-train affine calibration on both adjacent future frames.
  fc=cc['parameters_full_train'][st][str(ld)];fpt=fc['intercept']+fc['slope']*rawc
  if pld>0:
   fpcoef=cc['parameters_full_train'][st][str(pld)];fpp=fpcoef['intercept']+fpcoef['slope']*rawp
  else:fpp=rawp
  fdc=fpt[:,1,1]-fpp[:,1,1];fsx=float(np.sum(fdc*dy,dtype=np.float64));fss=float(np.sum(fdc*fdc,dtype=np.float64));old=full_sums.get((st,ld),(0.,0.));full_sums[(st,ld)]=(old[0]+fsx,old[1]+fss)
 alpha_full={};alpha_ex={str(k):{} for k in range(5)};metric=[];base_mse=np.zeros(len(keys));new_mse=np.zeros(len(keys));valid_row=np.zeros(len(keys),bool)
 for st in STATIONS:
  alpha_full[st]={}
  for ld in LEADS:
   sx,ss=full_sums[(st,ld)];raw=sx/max(ss,1e-12);alpha_full[st][str(ld)]={'alpha':float(np.clip(raw,0,2)),'unclipped_alpha':float(raw)}
   for ex in range(5):
    sx_e=sum(sums[(st,ld,k)][0] for k in range(5) if k!=ex);ss_e=sum(sums[(st,ld,k)][1] for k in range(5) if k!=ex);r=sx_e/max(ss_e,1e-12);alpha_ex[str(ex)].setdefault(st,{})[str(ld)]={'alpha':float(np.clip(r,0,2)),'unclipped_alpha':float(r)}
   for f in range(5):
    ix,pc,pp,yc,yp=prepared[(st,ld,f)];al=alpha_ex[str(f)][st][str(ld)]['alpha'];cand=pp+al*(pc-pp)
    base_mse[ix]=(pc[:,1,1]-yc[:,1,1])**2;new_mse[ix]=(cand[:,1,1]-yc[:,1,1])**2;valid_row[ix]=True
    metric.append({'station':st,'lead_minutes':ld,'fold':f,'n':len(ix),'alpha_excluding_fold':al,'center_rmse_acrel_log1p':float(np.sqrt(np.mean((pc[:,1,1]-yc[:,1,1])**2))),'center_rmse_candidate_log1p':float(np.sqrt(np.mean((cand[:,1,1]-yc[:,1,1])**2))),'patch_rmse_acrel_log1p':float(np.sqrt(np.mean((pc-yc)**2))),'patch_rmse_candidate_log1p':float(np.sqrt(np.mean((cand-yc)**2))),'pred_delta_std_center':float(np.std(pc[:,1,1]-pp[:,1,1])),'ref_delta_std_center':float(np.std(yc[:,1,1]-yp[:,1,1]))})
 # Date-block bootstrap for center RMSE, equal station and equal lead.
 rowidx=np.flatnonzero(valid_row);dunique=np.asarray(sorted(set(np.asarray(days)[rowidx])));dm={x:i for i,x in enumerate(dunique)};di=np.asarray([dm[days[i]] for i in rowidx]);sts=np.asarray(stlist);lds=np.asarray(leads);draw=np.empty(a.bootstrap);rng=np.random.default_rng(20261020);by={}
 for st in STATIONS:
  for ld in LEADS:
   sub=(sts[rowidx]==st)&(lds[rowidx]==ld);s0=np.bincount(di[sub],weights=base_mse[rowidx[sub]],minlength=len(dunique));s1=np.bincount(di[sub],weights=new_mse[rowidx[sub]],minlength=len(dunique));cnt=np.bincount(di[sub],minlength=len(dunique)).astype(float);by[(st,ld)]=(s0,s1,cnt)
 for b in range(a.bootstrap):
  sm=rng.integers(0,len(dunique),size=len(dunique));vv=[]
  for st in STATIONS:
   for ld in LEADS:
    s0,s1,cnt=by[(st,ld)];den=max(float(np.sum(cnt[sm])),1.);vv.append(np.sqrt(float(np.sum(s1[sm]))/den)-np.sqrt(float(np.sum(s0[sm]))/den))
  draw[b]=np.mean(vv)
 def macro(err):return float(np.mean([np.sqrt(np.mean(err[valid_row&(sts==s)&(lds==l)])) for s in STATIONS for l in LEADS]))
 center={}
 for st in STATIONS:
  m=valid_row&(sts==st);center[st]={'ACREL_center_rmse_log1p':float(np.sqrt(np.mean((base_mse[m])))),'ACRELTEMP_center_rmse_log1p':float(np.sqrt(np.mean(new_mse[m])))}
 out={'state':'COMPLETE_CROSSFIT_TRAIN_ONLY_COT_TEMPORAL_INCREMENT_AUDIT','transform':'target future log1p-COT = previous ACREL-calibrated context frame + alpha[station,lead]*(current ACREL target - previous ACREL frame)','fit':'through-origin least squares of exact 15-minute target and previous-frame log1p-COT increments','alpha_clip':[0.,2.],'train_rows_total':len(keys),'train_rows_with_exact_adjacent_actual_R':int(valid_row.sum()),'unmatched_previous_reference_rows':int(len(keys)-len(rowidx)),'target_bjt_date_blocks':len(dunique),'equal_station_equal_lead_center_rmse_log1p':{'ACREL':macro(base_mse),'ACRELTEMP':macro(new_mse),'delta_candidate_minus_ACREL':macro(new_mse)-macro(base_mse)},'date_block_bootstrap_equal_station_equal_lead_center_delta_ci95':np.quantile(draw,[.025,.975]).tolist(),'station_center_rmse_log1p':center,'parameters_full_train':alpha_full,'parameters_excluding_each_fold':alpha_ex,'fold_by_sequence_index':{str(ix):int(seqfold[sid]) for ix,sid in seqidx.items()},'metrics_by_station_lead_fold':metric,'forecast_pack_manifest_sha256':sha(a.forecast_pack/'pack_manifest.json'),'actual_r_pack_manifest_sha256':sha(a.actual_r_pack/'pack_manifest.json'),'center_calibration_sha256':sha(a.center_calibration),'bank_index_sha256':sha(a.bank_index),'fold_oof_ids_sha256':foldhash,'cpp_used':False,'validation_labels_used_for_fit':False,'test_payload_opened':False,'buffer_payload_opened':False,'future_observed_agri_used_as_inference_input':False}
 a.output.mkdir(parents=True);(a.output/'temporal_increment_calibration.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');
 with (a.output/'metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(metric[0]));w.writeheader();w.writerows(metric)
 print(json.dumps({'state':out['state'],'rows':out['train_rows_with_exact_adjacent_actual_R'],'rmse':out['equal_station_equal_lead_center_rmse_log1p'],'ci':out['date_block_bootstrap_equal_station_equal_lead_center_delta_ci95'],'station':center},indent=2))
if __name__=='__main__':main()
