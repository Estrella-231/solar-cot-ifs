#!/usr/bin/env python3
"""Nested train-date-fold screen for a fixed +/-15-minute forecast-COT phase shift."""
from __future__ import annotations
import argparse,csv,hashlib,json
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
from train_future_ghi_heads_cot_reliability import apply_reliability_calibration

STATIONS=('sili','zhujia');LEADS=tuple(range(15,226,15));SHIFTS=(-15,0,15)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bjt_day(s):return (datetime.strptime(s,'%Y-%m-%d %H:%M:%S')+timedelta(hours=8)).strftime('%Y-%m-%d')
def macro_rmse(err,st,ld,mask):
 if err.ndim==3:err=err[:,1,1]
 vals=[]
 for ss in STATIONS:
  for ll in LEADS:
   q=mask&(st==ss)&(ld==ll)
   if q.any():vals.append(float(np.sqrt(np.mean(err[q]))))
 return float(np.mean(vals))
def main():
 p=argparse.ArgumentParser();p.add_argument('--forecast-pack',type=Path,required=True);p.add_argument('--oracle-pack',type=Path,required=True);p.add_argument('--bank-index',type=Path,required=True);p.add_argument('--forecast-bank',type=Path,required=True);p.add_argument('--fold-root',type=Path,required=True);p.add_argument('--center-calibration',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=2000);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 fm=json.loads((a.forecast_pack/'pack_manifest.json').read_text());om=json.loads((a.oracle_pack/'pack_manifest.json').read_text());cal=json.loads(a.center_calibration.read_text())
 if fm.get('state')!='COMPLETE_CAUSAL_FUTURE_GHI_HEAD_PACK' or fm.get('train_forecast_oof') is not True or fm.get('forecast_bank_oof_fold_count')!=5:raise ValueError('five-fold OOF forecast pack required')
 if any(fm.get(k) is not False for k in ('test_payload_opened','buffer_payload_opened','future_observed_agri_used_as_input','cpp_used_as_input')):raise ValueError('forecast causal/test closure failed')
 if om.get('state')!='COMPLETE_ORACLE_FUTURE_COT_GHI_PACK' or om.get('split_id')!=fm.get('split_id') or om.get('deployable') is not False or om.get('test_payloads_read')!=0 or om.get('buffer_payload_opened') is not False:raise ValueError('oracle training-reference source/closure failed')
 if cal.get('state')!='COMPLETE_CROSSFIT_TRAIN_ONLY_OOF_FUTURE_COT_RELIABILITY_CALIBRATION' or cal.get('forecast_pack_manifest_sha256')!=sha(a.forecast_pack/'pack_manifest.json'):raise ValueError('reliability calibration mismatch')
 bank_manifest=json.loads((a.forecast_bank/'bank_manifest.json').read_text())
 if bank_manifest.get('state')!='COMPLETE_FORECAST_AGRI_R_TRAIN_VAL_BANK' or bank_manifest.get('oof_fold_count')!=5:raise ValueError('five-fold forecast bank required')
 if any(bank_manifest.get(k) is not False for k in ('test_payload_opened','buffer_payload_opened','future_observed_agri_read','cpp_read')):raise ValueError('forecast bank test/buffer/future-observation closure failed')
 bankcot_path=a.forecast_bank/'forecast_cot_log1p.npy'
 if sha(bankcot_path)!=bank_manifest['array_sha256']['forecast_cot_log1p.npy']:raise ValueError('forecast COT bank hash mismatch')
 bankcot=np.load(bankcot_path,mmap_mode='r',allow_pickle=False)
 expected_layout=bank_manifest['layouts']['forecast_cot_log1p.npy']
 if list(bankcot.shape)!=expected_layout or bankcot.dtype!=np.float32 or len(expected_layout)!=6 or expected_layout[1:]==[]:raise ValueError('forecast COT bank shape/dtype mismatch')
 seqidx={};bankrow={}
 for r in csv.DictReader((a.bank_index).open(newline='')):
  if r['split']=='train':
   si=int(r['sequence_index_within_split']);sid=r['seq_id']
   if si in seqidx or sid in bankrow:raise ValueError('duplicate train sequence in bank index')
   seqidx[si]=sid;bankrow[sid]=int(r['bank_index'])
 seqfold={}
 for f in range(5):
  for sid in (a.fold_root/f'fold_{f}'/'fold_oof_ids.txt').read_text().splitlines():
   if sid in seqfold:raise ValueError('sequence duplicated across date folds')
   seqfold[sid]=f
 if set(seqfold)!=set(seqidx.values()):raise ValueError('fold IDs do not exactly cover OOF bank')
 keys=list(csv.DictReader((a.forecast_pack/'train'/'keys.csv').open(newline='')));n=len(keys)
 if n!=fm.get('rows',{}).get('train') or n!=714914:raise ValueError('unexpected train key count')
 pred=np.load(a.forecast_pack/'train'/'cot_log1p.npy',mmap_mode='r',allow_pickle=False)
 if pred.shape!=(n,3,1,3,3) or pred.dtype!=np.float32:raise ValueError('forecast COT shape/dtype mismatch')
 if sha(a.forecast_pack/'train'/'cot_log1p.npy')!=fm['array_sha256']['train']['cot_log1p.npy']:raise ValueError('forecast array hash mismatch')
 cal_pred=apply_reliability_calibration(np.asarray(pred),keys,'train',cal)
 oracle_keys=list(csv.DictReader((a.oracle_pack/'train'/'keys.csv').open(newline='')))
 if len(oracle_keys)!=n:raise ValueError('oracle training row count differs from forecast pack')
 def keytuple(r):return (r['station'],r['init_time_utc'],r['target_time_utc'],str(int(r['lead_minutes'])),str(int(r['ghi_row_index'])))
 oracle_map={keytuple(r):i for i,r in enumerate(oracle_keys)}
 if len(oracle_map)!=len(oracle_keys):raise ValueError('duplicate oracle GHI row key')
 oracle_rows=[oracle_map.get(keytuple(r)) for r in keys]
 if any(i is None for i in oracle_rows):raise ValueError('forecast/oracle exact training key mismatch')
 oricot=np.load(a.oracle_pack/'train'/'cot_log1p.npy',mmap_mode='r',allow_pickle=False)
 if oricot.shape!=(n,3,1,3,3) or oricot.dtype!=np.float32:raise ValueError('oracle training COT shape/dtype mismatch')
 if sha(a.oracle_pack/'train'/'cot_log1p.npy')!=om['split_reports']['train']['cot_log1p_sha256']:raise ValueError('oracle COT hash mismatch')
 rowmap={};seq=[];station=[];lead=[];fold=[];day=[];refs=[];valid=[];bank_rows=[]
 for i,r in enumerate(keys):
  if r['split']!='train':raise ValueError('non-train row in train array')
  si=int(r['sequence_index_within_split']);st=r['station'];ld=int(r['lead_minutes']);sid=seqidx.get(si)
  if sid is None:raise ValueError('missing sequence ID')
  key=(sid,st,ld)
  if key in rowmap:raise ValueError('duplicate sequence/station/lead row')
  rowmap[key]=i
  if ld in LEADS:
   ri=oracle_rows[i]
   if sid not in bankrow:raise ValueError('train forecast sequence missing from OOF bank index')
   seq.append((sid,st,ld));station.append(st);lead.append(ld);fold.append(seqfold[sid]);day.append(bjt_day(r['target_time_utc']));refs.append(ri);valid.append(i);bank_rows.append(bankrow[sid])
 ix=np.asarray(valid,np.int64);st=np.asarray(station);ld=np.asarray(lead);fo=np.asarray(fold);days=np.asarray(day);ri=np.asarray(refs,np.int64);bi=np.asarray(bank_rows,np.int64)
 shifts={-15:cal_pred[ix,1,0],0:cal_pred[ix,2,0]};plus=[]
 station_pos={s:i for i,s in enumerate(bank_manifest['stations'])}
 max_pack_diff=0.0;pack_compare_count=0
 for k,(sid,ss,ll) in enumerate(seq):
  source_lead=ll+15
  if source_lead>240:raise ValueError(f'+15 shifted source outside 8-to-16 forecast bank for lead {ll}')
  params=cal['parameters_excluding_each_fold'][str(int(fo[k]))][ss][str(source_lead)]
  raw=bankcot[bi[k],station_pos[ss],source_lead//15-1,0]
  calibrated=np.float32(params['intercept'])+np.float32(params['slope'])*np.asarray(raw,np.float32)
  plus.append(calibrated)
  # Cross-check the direct OOF-bank frame against a packed row whenever that
  # adjacent GHI target happens to exist; COT source availability must not
  # depend on whether the adjacent station label survived GHI masking.
  j=rowmap.get((sid,ss,source_lead))
  if j is not None:
   max_pack_diff=max(max_pack_diff,float(np.max(np.abs(calibrated-cal_pred[j,2,0]))));pack_compare_count+=1
 if pack_compare_count==0 or max_pack_diff>1e-6:raise ValueError(f'direct bank +15 COT source disagrees with packed rows: n={pack_compare_count}, max_abs={max_pack_diff}')
 shifts[15]=np.asarray(plus,np.float32)
 y=np.asarray(oricot[ri,2,0],np.float32);base=shifts[0];base_e=(base-y)**2
 # Nested selection: choose one global shift on four folds, score only the untouched fold.
 selected={};cand=np.empty_like(base)
 for f in range(5):
  train=fo!=f;test=fo==f
  scores={s:macro_rmse((v-y)**2,st,ld,train) for s,v in shifts.items()}
  pick=min(SHIFTS,key=lambda s:(scores[s],abs(s),s));selected[str(f)]={'selected_shift_minutes':pick,'train_macro_center_rmse_by_shift':{str(s):scores[s] for s in SHIFTS}}
  cand[test]=shifts[pick][test]
 cand_e=(cand-y)**2; patch_base=np.mean((base-y)**2,axis=(1,2));patch_cand=np.mean((cand-y)**2,axis=(1,2));
 # Paired target-date block bootstrap, equal weight by station and lead.
 ud=np.asarray(sorted(set(days)));dm={d:i for i,d in enumerate(ud)};di=np.asarray([dm[d] for d in days]);by={}
 for ss in STATIONS:
  for ll in LEADS:
   q=(st==ss)&(ld==ll);c0=np.bincount(di[q],weights=base_e[q,1,1],minlength=len(ud));c1=np.bincount(di[q],weights=cand_e[q,1,1],minlength=len(ud));cp0=np.bincount(di[q],weights=patch_base[q],minlength=len(ud));cp1=np.bincount(di[q],weights=patch_cand[q],minlength=len(ud));cnt=np.bincount(di[q],minlength=len(ud)).astype(float);by[(ss,ll)]=(c0,c1,cp0,cp1,cnt)
 rng=np.random.default_rng(20261020);draw=np.empty(a.bootstrap);drawp=np.empty(a.bootstrap)
 for b in range(a.bootstrap):
  sm=rng.integers(0,len(ud),size=len(ud));dc=[];dp=[]
  for ss in STATIONS:
   for ll in LEADS:
    c0,c1,p0,p1,cnt=by[(ss,ll)];den=max(float(cnt[sm].sum()),1.);dc.append(np.sqrt(c1[sm].sum()/den)-np.sqrt(c0[sm].sum()/den));dp.append(np.sqrt(p1[sm].sum()/den)-np.sqrt(p0[sm].sum()/den))
  draw[b]=np.mean(dc);drawp[b]=np.mean(dp)
 station_metrics={}
 for ss in STATIONS:
  q=st==ss;station_metrics[ss]={'baseline_center_rmse_log1p':float(np.sqrt(base_e[q,1,1].mean())),'nested_shift_center_rmse_log1p':float(np.sqrt(cand_e[q,1,1].mean())),'delta_center_rmse_log1p':float(np.sqrt(cand_e[q,1,1].mean())-np.sqrt(base_e[q,1,1].mean())),'baseline_patch_rmse_log1p':float(np.sqrt(patch_base[q].mean())),'nested_shift_patch_rmse_log1p':float(np.sqrt(patch_cand[q].mean()))}
 full_scores={s:macro_rmse((v-y)**2,st,ld,np.ones(len(ix),bool)) for s,v in shifts.items()};fullpick=min(SHIFTS,key=lambda s:(full_scores[s],abs(s),s));freq={str(s):sum(v['selected_shift_minutes']==s for v in selected.values()) for s in SHIFTS}
 gate=(float(np.quantile(draw,.975))<0 and all(v['delta_center_rmse_log1p']<=0 for v in station_metrics.values()) and max(freq.values())>=4)
 out={'state':'COMPLETE_NESTED_TRAIN_ONLY_FUTURE_COT_LEAD_PHASE_AUDIT','candidate_shifts_minutes':list(SHIFTS),'selected_shift_full_train_minutes':fullpick,'full_train_macro_center_rmse_by_shift':{str(s):full_scores[s] for s in SHIFTS},'outer_fold_selected_shifts':selected,'outer_fold_selection_frequency':freq,'nested_equal_station_equal_lead_center_delta_candidate_minus_current_log1p':float(macro_rmse(cand_e,st,ld,np.ones(len(ix),bool))-macro_rmse(base_e,st,ld,np.ones(len(ix),bool))),'nested_center_delta_date_block_ci95_log1p':np.quantile(draw,[.025,.975]).tolist(),'nested_patch_delta_equal_station_equal_lead_log1p':float(np.mean([np.sqrt(np.mean(patch_cand[(st==ss)&(ld==ll)]))-np.sqrt(np.mean(patch_base[(st==ss)&(ld==ll)])) for ss in STATIONS for ll in LEADS])),'nested_patch_delta_date_block_ci95_log1p':np.quantile(drawp,[.025,.975]).tolist(),'stations':station_metrics,'rows':int(len(ix)),'target_bjt_date_blocks':int(len(ud)),'leads_scored_minutes':list(LEADS),'plus15_source':'direct same-sequence OOF forecast-COT bank frame at target lead + 15 min','plus15_bank_vs_pack_crosscheck_rows':pack_compare_count,'plus15_bank_vs_pack_crosscheck_max_abs_log1p':max_pack_diff,'forecast_bank_manifest_sha256':sha(a.forecast_bank/'bank_manifest.json'),'forecast_cot_bank_sha256':sha(bankcot_path),'gate':'PASS' if gate else 'NO_GO','gate_rule':'center date-block 95% CI wholly below zero; neither station center RMSE worsens; same shift selected in >=4/5 outer folds','forecast_pack_manifest_sha256':sha(a.forecast_pack/'pack_manifest.json'),'forecast_cot_sha256':sha(a.forecast_pack/'train'/'cot_log1p.npy'),'oracle_pack_manifest_sha256':sha(a.oracle_pack/'pack_manifest.json'),'oracle_train_cot_sha256':sha(a.oracle_pack/'train'/'cot_log1p.npy'),'calibration_sha256':sha(a.center_calibration),'bank_index_sha256':sha(a.bank_index),'test_payload_opened':False,'buffer_payload_opened':False,'future_observed_agri_used_as_inference_input':False,'cpp_used':False,'validation_labels_used_for_fit':False,'oracle_future_agri_used_only_as_train_reference':True}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps({k:out[k] for k in ('state','selected_shift_full_train_minutes','outer_fold_selection_frequency','nested_equal_station_equal_lead_center_delta_candidate_minus_current_log1p','nested_center_delta_date_block_ci95_log1p','stations','gate')},indent=2))
if __name__=='__main__':main()
