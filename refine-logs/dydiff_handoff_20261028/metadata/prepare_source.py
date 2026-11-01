"""Freeze epoch150 and select complete shared validation windows before scoring."""
from pathlib import Path
from datetime import datetime,timedelta
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv,json,math,os,shutil
import numpy as np
import torch
from paths import *

def read(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
def cache_frame(r):
 src=RAW/r['relative'];st=src.stat()
 assert (st.st_size,st.st_mtime_ns)==(r['source_bytes'],r['source_mtime_ns'])
 dst=Path(os.environ['GHI_RAW_CACHE'])/src.name;shutil.copy2(src,dst)
 assert sha(dst)==r['source_sha256']
 return r['relative']

def main():
 protocol=json.loads((OUT/'protocol.json').read_text());assert protocol['dydiff_epoch']==150
 completed=json.loads((BASE/'run_seed42_150ep_v1/training_complete.json').read_text())
 assert completed['status']=='PASS' and completed['epochs']==150
 checkpoint=OUT/'checkpoint_epoch150.pt'
 assert not checkpoint.exists(),'Do not silently overwrite frozen epoch150'
 shutil.copy2(BASE/'run_seed42_150ep_v1/last.pt',checkpoint)
 ckpt=torch.load(checkpoint,map_location='cpu');assert ckpt['epoch']==150 and ckpt['global_step']==34950
 meta=json.loads((BASE/'run_seed42_150ep_v1/run_metadata.json').read_text())
 assert ckpt['signature']==meta['signature'];del ckpt
 for name,h in meta['source_sha256'].items():assert sha(BASE/'train_code_v1'/name)==h
 ready=json.loads((PREP/'DATA_READY.json').read_text())
 assert ready['status']=='PASS' and sha(PREP/'norm.json')==ready['norm_sha256']
 assert sha(PREP/'trainval_manifest.csv')==ready['manifest_sha256']
 oldmeta=json.loads((OLD/'S_runs_attempt12/full/run_metadata.json').read_text())
 assert sha(OLD/'prepared_attempt12/indexes/full.csv')==oldmeta['index_sha256']
 newall=read(PREP/'trainval_manifest.csv');newval=[r for r in newall if r['split']=='val']
 assert len(newval)==3171
 newbyinit={(datetime.strptime(r['BJT_start'],'%Y-%m-%d %H:%M:%S')+timedelta(minutes=105)).strftime('%Y%m%d%H%M'):(i,r) for i,r in enumerate(newval)}
 assert len(newbyinit)==len(newval)
 trainframes={Path(p).stem for r in newall if r['split']=='train' for p in r['data_relpaths'].split('|')}
 oldindex=read(OLD/'prepared_attempt12/indexes/full.csv')
 oldtrainframes={Path(p).stem for r in oldindex if r['split']=='train' for p in r['data_paths'].split('|')}
 oldkeys=read(BANK/'keys.csv');assert len(oldkeys)==407
 packreport=json.loads((PACK/'pack_report.json').read_text())
 assert sha(PACK/'rows.csv')==json.loads((HEADS/'observed_train_v1/manifest.json').read_text())['rows_sha256']
 rows=read(PACK/'rows.csv');oldvalrows=[r for r in rows if r['split']=='val'];assert len(oldvalrows)==15275
 ghi_train_dates={r['target_time_bjt'][:8] for r in rows if r['split']=='train'}
 exclusion=Counter();selected=[]
 for oldk in oldkeys:
  init=oldk['init_time_bjt']
  if init not in newbyinit:exclusion['not_a_complete_qualified_DyDiff_validation_window']+=1;continue
  vi,r=newbyinit[init];paths=r['data_relpaths'].split('|');stems=[Path(p).stem for p in paths]
  assert stems==[Path(p).stem for p in oldk['data_paths'].split('|')]
  dates=[(datetime.strptime(s,'%Y%m%d%H%M')+timedelta(hours=8)).strftime('%Y%m%d') for s in stems]
  assert not set(stems)&trainframes and not set(stems)&oldtrainframes
  assert not set(dates)&ghi_train_dates
  targets=[(datetime.strptime(s,'%Y%m%d%H%M')+timedelta(hours=8)).strftime('%Y%m%d%H%M') for s in stems[8:]]
  assert targets==oldk['target_times_bjt'].split('|')
  selected.append(dict(index=len(selected),old_sequence_index=int(oldk['sample_index']),
    original_val_index=vi,init_time_bjt=init,target_times_bjt=targets,row=r))
 assert selected and len({r['init_time_bjt'] for r in selected})==len(selected)
 selectedbyold={r['old_sequence_index']:r for r in selected}
 # Load only previously audited, deterministic solar geometry/validity sidecars.
 auxcache={}
 def aux(rel):
  stem=Path(rel).stem
  if stem not in auxcache:
   with np.load(PREP/'aux'/f'{stem}.npz') as z:
    auxcache[stem]={k:z[k][103:119,118:134].copy() for k in ['cosSOZ','cosRAA','day_mask','valid_bits']}
  return auxcache[stem]
 geometry=np.empty((len(selected),16,3,16,16),np.float32)
 valid=np.empty((len(selected),16,16,16),bool)
 for r in selected:
  for t,rel in enumerate(r['row']['data_relpaths'].split('|')[8:]):
   a=aux(rel);geometry[r['index'],t]=np.stack([a[k] for k in ['cosSOZ','cosRAA','day_mask']])
   valid[r['index'],t]=(a['valid_bits']==8191)&(a['day_mask']>.5)&(a['cosSOZ']>math.cos(math.radians(80)))
 oldgeom=np.load(BANK/'future_geometry_patch.npy',mmap_mode='r')
 geom_delta=float(np.max(abs(geometry-oldgeom[[r['old_sequence_index'] for r in selected]])))
 assert geom_delta<1e-6,('known geometry disagreement',geom_delta)
 selectedrows=[];rowexclude=Counter()
 for r in oldvalrows:
  oi=int(r['source_sequence_index'])
  if oi not in selectedbyold:rowexclude['sequence_not_in_shared_qualified_validation']+=1;continue
  s=selectedbyold[oi];t=int(r['lead_minutes'])//15-1;yy,xx=CENTERS[r['station']]
  assert r['init_time_bjt']==s['init_time_bjt'] and r['target_time_bjt']==s['target_times_bjt'][t]
  if not valid[s['index'],t,yy-1:yy+2,xx-1:xx+2].all():
   rowexclude['station_3x3_not_fully_supported_or_sza_ge80']+=1;continue
  selectedrows.append(dict(pack_index=int(r['index']),sequence_index=s['index'],old_sequence_index=oi,
    station=r['station'],init_time_bjt=r['init_time_bjt'],target_time_bjt=r['target_time_bjt'],lead_minutes=int(r['lead_minutes'])))
 assert {r['station'] for r in selectedrows}==set(STATIONS)
 assert len({(r['station'],r['init_time_bjt'],r['target_time_bjt'],r['lead_minutes']) for r in selectedrows})==len(selectedrows)
 dates=sorted({r['target_time_bjt'][:8] for r in selectedrows});assert len(dates)>=10
 dump(OUT/'selection.json',dict(sequences=selected,rows=selectedrows))
 np.save(OUT/'future_geometry.npy',geometry)
 # Verify all frozen head identities before the first prediction is scored.
 inventory=[]
 for st in STATIONS:
  for seed in SEEDS:
   root=HEADS/'training_v1'/f'{st}_seed{seed}';m=json.loads((root/'run_metadata.json').read_text())
   assert m['state']=='COMPLETE' and m['epochs']==30 and m['parameters']==24489
   for arm in ARMS:
    path=root/f'{arm}_final.pt';digest=sha(path)
    assert digest==m['results'][arm]['final']['checkpoint_sha256']
    pp=root/f'{arm}_final_predictions.npz';assert sha(pp)==m['results'][arm]['final']['prediction_sha256']
    inventory.append(dict(station=st,seed=seed,arm=arm,path=str(path),sha256=digest))
 rmeta=json.loads((OLD/'R_runs_attempt12/full/run_metadata.json').read_text())
 assert sha(OLD/'code_pipeline_attempt12/train_cot_repaired.py')==rmeta['model_source_sha256']
 for name in ['agri_raw.npy','geometry.npy','cot.npy','ghi_wm2.npy','clear_sky_wm2.npy']:
  assert sha(PACK/name)==packreport['arrays_sha256'][name]
 needed={p for r in selected for p in r['row']['data_relpaths'].split('|')[:8]}
 audit={}
 for line in (PREP/'frame_audit.jsonl').open():
  a=json.loads(line)
  if a['relative'] in needed:assert not a.get('rejected');audit[a['relative']]=a
 assert set(audit)==needed
 dump(OUT/'history_audit.json',audit)
 with ProcessPoolExecutor(max_workers=6) as pool:
  for i,_ in enumerate(pool.map(cache_frame,audit.values(),chunksize=4),1):
   if i%200==0:print('HISTORY_CACHE_VERIFIED',i,len(needed),flush=True)
 dump(OUT/'PREPARE_COMPLETE.json',dict(status='PASS',dydiff_epoch=150,dydiff_checkpoint_sha256=sha(checkpoint),
  common_sequences=len(selected),common_GHI_rows=len(selectedrows),target_dates=dates,
  rows_by_station=dict(Counter(r['station'] for r in selectedrows)),sequence_exclusions=dict(exclusion),row_exclusions=dict(rowexclude),
  observed_future_AGRI_payloads_read=0,test_payloads_read=0,buffer_payloads_read=0,
  history_frames_cached=len(needed),geometry_max_abs_delta=geom_delta,head_inventory=inventory,
  frozen_R_checkpoint_sha256=sha(OLD/'R_runs_attempt12/full/best.pt'),frozen_R_norm_sha256=sha(OLD/'prepared_attempt12/R_packs/full/norm.json'),
  protocol_sha256=sha(OUT/'protocol.json'),selection_sha256=sha(OUT/'selection.json'),norm_sha256=sha(PREP/'norm.json')))
 print('PREPARE_COMPLETE',len(selected),'sequences',len(selectedrows),'GHI rows',dates,flush=True)
if __name__=='__main__':main()
