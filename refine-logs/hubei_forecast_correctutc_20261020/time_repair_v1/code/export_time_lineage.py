"""Export only train/val metadata and existing GHI15 labels after UTC/BJT audit."""
import csv,json,hashlib,tarfile,shutil
from pathlib import Path
from datetime import datetime,timedelta
ROOT=Path('/home/Data_Pool/zjnu/hubei_cot_independent_20261016')
OUT=ROOT/'time_repair_export_20261020_v2'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def stamp(p):return Path(p).stem[:12]
def times(row):
 ds=[datetime.strptime(stamp(p),'%Y%m%d%H%M') for p in row['data_paths'].split('|')];gs=[datetime.strptime(stamp(p),'%Y%m%d%H%M') for p in row['aux_paths'].split('|')]
 assert len(ds)==len(gs)==24 and all(g-d==timedelta(hours=8) for d,g in zip(ds,gs))
 assert all(gs[i+1]-gs[i]==timedelta(minutes=15) for i in range(23))
 assert len({g.date() for g in gs})==1
 return gs
def main():
 OUT.mkdir(exist_ok=False);man=ROOT/'manifests/oof_fold_manifests_20261016_v5';dayspath=man/'input/hubei_day_assignment_20261016.csv';foldspath=man/'input/hubei_oof_train_date_fold_20261016.csv'
 days={r['date_BJT'].replace('-',''):r['split'] for r in read(dayspath)};folds={r['date_BJT'].replace('-',''):int(r['heldout_fold']) for r in read(foldspath)}
 names=['S_oof_fold0_oof_bank_fd201_20261017_raw_v5_attempt2_gpu0','S_oof_fold1_oof_bank_fd201_20261017_raw_v5_attempt2_gpu1','S_oof_fold2_oof_bank_fd201_20261017_raw_v5_attempt2_gpu2','S_oof_fold3_oof_bank_fd201_20261017_raw_v5_tc6000_import_gpu0','S_oof_fold4_oof_bank_fd201_20261017_raw_v5_tc6000_import_gpu0'];audit=[]
 for f in range(6):
  role=f'fold_{f}' if f<5 else 'validation';b=ROOT/'oof_training'/names[f] if f<5 else ROOT/'forecast_bank/strict24_same_day_v3_best_epoch10_fulltrainval_20261016';idx=man/f'folds/fold_{f}/simvp_oof_index.csv' if f<5 else ROOT/'manifests/strict24frame_same_day_20261016_v2/index.csv'
  ir=read(idx);assert all(r['split'] in ('train','val') for r in ir);report=json.loads((b/'BANK_REPORT.json').read_text());assert report['index_sha256']==sha(idx)
  lookup={r['data_paths']:r for r in ir if r['split']=='val'};assert len(lookup)==sum(r['split']=='val' for r in ir)
  train_dates=set();heldout_dates=set()
  for r in ir:
   gt=times(r);d=gt[7].strftime('%Y%m%d');expected='train' if f<5 or r['split']=='train' else 'val';assert days[d]==expected
   if r['split']=='train':
    train_dates.add(d)
    if f<5:assert folds[d]!=f
   else:
    heldout_dates.add(d)
    if f<5:assert folds[d]==f
  training_index_sha=None
  if f<5:
   ti=man/f'folds/fold_{f}/simvp_trainval_index.csv';train_dates=set()
   for tr in read(ti):
    assert tr['split'] in ('train','val');gt=times(tr);d=gt[7].strftime('%Y%m%d')
    if tr['split']=='train':assert days[d]=='train' and folds[d]!=f;train_dates.add(d)
    else:assert days[d]=='val'
   training_index_sha=sha(ti);shutil.copyfile(ti,OUT/f'{role}_training_index.csv')
  assert train_dates and not train_dates&heldout_dates
  keys=read(b/'val/keys.csv');fixed=[]
  for i,k in enumerate(keys):
   r=lookup[k['data_paths']];g=times(r);assert int(k['sample_index'])==i
   actual=g[7];assert actual-datetime.strptime(k['init_time_bjt'],'%Y%m%d%H%M')==timedelta(hours=8)
   assert all(g[j+8]-datetime.strptime(t,'%Y%m%d%H%M')==timedelta(hours=8) for j,t in enumerate(k['target_times_bjt'].split('|')))
   fixed.append(dict(sample_index=i,split='train' if f<5 else 'val',role=role,init_time_bjt=actual.strftime('%Y%m%d%H%M'),target_times_bjt='|'.join(t.strftime('%Y%m%d%H%M') for t in g[8:]),source_utc_times='|'.join(stamp(p) for p in r['data_paths'].split('|')),source_aux_bjt_times='|'.join(t.strftime('%Y%m%d%H%M') for t in g),data_paths=r['data_paths'],aux_paths=r['aux_paths']))
  p=OUT/f'{role}_keys.csv'
  with p.open('w',newline='') as stream:w=csv.DictWriter(stream,fieldnames=list(fixed[0]));w.writeheader();w.writerows(fixed)
  shutil.copyfile(idx,OUT/f'{role}_source_index.csv');shutil.copyfile(b/'BANK_REPORT.json',OUT/f'{role}_BANK_REPORT.json')
  audit.append(dict(role=role,sequences=len(keys),source_index_sha256=sha(idx),training_index_sha256=training_index_sha,old_keys_sha256=sha(b/'val/keys.csv'),canonical_keys_sha256=sha(p),train_date_count=len(train_dates),heldout_date_count=len(heldout_dates),train_heldout_date_overlap=0,all_24_UTC_plus8_equals_aux_BJT=True,all_true_BJT_dates_match_original_frozen_split=True,fold_exclusion_checked=f<5))
 for p,name in [(dayspath,'day_assignment.csv'),(foldspath,'train_date_folds.csv'),(ROOT/'labels/ghi15_train_val_daylight_clear.csv','ghi15_train_val_daylight_clear.csv')]:shutil.copyfile(p,OUT/name)
 assert all(r['split'] in ('train','val') for r in read(OUT/'ghi15_train_val_daylight_clear.csv'))
 result=dict(state='ALL_TRAIN_VAL_TIME_LINEAGE_AUDIT_PASS',roles=audit,test_payloads_read=0,buffer_payloads_read=0,source_prediction_arrays_unchanged=True,simvp_training_and_fold_dates_reusable=True)
 result['files']={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in OUT.iterdir() if p.is_file()}
 (OUT/'audit.json').write_text(json.dumps(result,indent=2))
 with tarfile.open(OUT/'time_lineage.tar.gz','w:gz') as tar:
  for p in OUT.iterdir():
   if p.name!='time_lineage.tar.gz':tar.add(p,arcname=p.name,recursive=False)
 (OUT/'archive_sha256.txt').write_text(sha(OUT/'time_lineage.tar.gz')+'  time_lineage.tar.gz\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
