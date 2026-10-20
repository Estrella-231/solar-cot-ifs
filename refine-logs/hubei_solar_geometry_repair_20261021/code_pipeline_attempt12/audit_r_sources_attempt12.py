#!/usr/bin/env python3
"""Hash/shape audit for all R rows absent from the corrected S frame cache; train/val only."""
import argparse,csv,hashlib,json
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np

def rows(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--preparation',type=Path,required=True);a=p.parse_args()
 base=a.base;rpack=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_cloud_probability_20261020/R_pack_correctutc_attempt2');old=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/hubei_independent_trainval_stage_20261017/data');new=base/'r_only_data_attempt6'
 rr=rows(rpack/'rows.csv');s=rows(base/'prepared_attempt4'/'stage_manifest.csv');splits={r['date_BJT'].replace('-',''):r['split'] for r in rows(a.preparation/'metadata'/'day_assignment.csv')}
 assert len(rr)==3479 and len({r['time_bjt'] for r in rr})==3479 and len(s)==10473 and len({r['bjt'] for r in s})==10473
 s_bjt={r['bjt'] for r in s};missing=[r for r in rr if r['time_bjt'] not in s_bjt];assert len(missing)==560
 recovered=json.loads((base/'MISSING_R_SOURCE_AUDIT_attempt6.json').read_text());transfer=json.loads((base/'TRANSFER_AUDIT_attempt6.json').read_text())
 assert recovered['state']=='PASS_SOURCE_R_ONLY_TRAIN_VAL' and recovered['count']==202
 assert transfer['state']=='PASS_TRANSFERRED_R_ONLY_TRAIN_VAL_SHA256' and transfer['count']==202 and transfer['all_hashes_match'] and transfer['source_audit_sha256']==sha(base/'MISSING_R_SOURCE_AUDIT_attempt6.json')
 recov_lut={r['bjt']:r for r in recovered['records']};assert len(recov_lut)==202
 records=[];counts={'old_stage_flat':0,'recovered_fd201':0}
 for i,r in enumerate(missing):
  bjt=r['time_bjt'];utc=(datetime.strptime(bjt,'%Y%m%d%H%M')-timedelta(hours=8)).strftime('%Y%m%d%H%M');split=r['split'];assert split in ('train','val') and splits[bjt[:8]]==split
  rel=Path(utc[:4])/utc[:6]/utc[:8]/f'{utc}.npy'
  oldp=old/f'{utc}.npy';newp=new/rel
  if oldp.is_file():kind='old_stage_flat';path=oldp
  else:
   assert bjt in recov_lut,('missing from both canonical source inventories',bjt,utc)
   src=recov_lut[bjt];assert src['utc']==utc and src['split']==split and src['relative']==rel.as_posix()
   path=newp;kind='recovered_fd201'
   assert path.is_file() and sha(path)==src['sha256'],('recovered file/hash mismatch',bjt,str(path))
  arr=np.load(path,mmap_mode='r',allow_pickle=False)
  if arr.shape!=(20,256,256) or arr.dtype!=np.float32 or not np.isfinite(arr).all():raise ValueError((bjt,arr.shape,arr.dtype,'nonfinite'))
  h=sha(path)
  if kind=='recovered_fd201':assert h==recov_lut[bjt]['sha256']
  records.append({'index':i,'bjt':bjt,'utc':utc,'split':split,'relative':rel.as_posix(),'source_kind':kind,'source_path':str(path),'shape':list(arr.shape),'dtype':str(arr.dtype),'bytes':path.stat().st_size,'sha256':h})
  counts[kind]+=1
  if i%64==0:print('AUDITED_R_NON_S_SOURCE',i+1,len(missing),flush=True)
 assert counts=={'old_stage_flat':358,'recovered_fd201':202},counts
 report={'state':'PASS_R_NON_S_CACHE_SOURCES_TRAIN_VAL_SHA256','count':len(records),'split_counts':{x:sum(r['split']==x for r in records) for x in ('train','val')},'source_counts':counts,'R_rows':len(rr),'S_frames':len(s),'missing_from_S_cache':len(missing),'R_rows_cover_missing_keys_exactly':{r['bjt'] for r in records}=={r['time_bjt'] for r in missing},'R_only_transfer_audit_sha256':sha(base/'TRANSFER_AUDIT_attempt6.json'),'restoration_witness_audit_sha256':sha(base/'RESTORATION_WITNESS_ARCHIVE_AUDIT_attempt7.json'),'records':records,'test_payloads_read':0,'buffer_payloads_read':0}
 assert report['R_rows_cover_missing_keys_exactly'] and report['split_counts']['train']+report['split_counts']['val']==560
 (base/'R_NON_S_CACHE_SOURCE_AUDIT_attempt12.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2),flush=True)
if __name__=='__main__':main()
