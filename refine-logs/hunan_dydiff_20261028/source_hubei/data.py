"""Frozen FULL observation Hubei contract; independent of COT/GHI subsets."""
import csv,hashlib,json
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
from torch.utils.data import Dataset
CHANNELS=(0,1,2,3,4,5,8,9,10,11,12,13,14)
BASE=Path('/public/home/fyzhang/ttzhou/swc/irradiance_forecast/AGRI_DyDiff_HubeiFull_20261023')
PREPARED=BASE/'prepared_v2'
DATA_ROOT=Path('/public/home/slfu/ttzhou/Auxiliary_data/HuBei_AGRI_Preprocessed/data')
SOURCE_MANIFEST_SHA256='d7e6a810cb99997dd8480d58037ba63bfc172a99ad1d66b42a8da5b25ff88d76'
class HubeiFullDataset(Dataset):
 def __init__(self,split='train'):
  if split not in ('train','val'):raise ValueError('test payloads closed')
  ready=json.loads((PREPARED/'DATA_READY.json').read_text())
  assert ready['status']=='PASS' and ready['region']=='Hubei'
  assert ready['source_manifest_sha256']==SOURCE_MANIFEST_SHA256
  assert ready['source_inventory_frame_count']==73371 and ready['frame_overlap_between_splits']==0
  assert ready['audited_trainval_frames']==55628 and ready['test_payloads_read']==ready['buffer_payloads_read']==0
  manifest=PREPARED/'trainval_manifest.csv';norm=PREPARED/'norm.json'
  self.manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()
  self.normalization_sha256=hashlib.sha256(norm.read_bytes()).hexdigest()
  assert self.manifest_sha256==ready['manifest_sha256'] and self.normalization_sha256==ready['norm_sha256']
  days={r['date_BJT']:r['split'] for r in csv.DictReader((BASE/'split_v1/day_assignment.csv').open())}
  with manifest.open() as f:all_rows=list(csv.DictReader(f))
  seen={'train':set(),'val':set()}
  for row in all_rows:
   assert row['split'] in seen
   paths=row['data_relpaths'].split('|');ts=[datetime.strptime(Path(p).stem,'%Y%m%d%H%M') for p in paths]
   assert len(paths)==24 and all(b-a==timedelta(minutes=15) for a,b in zip(ts,ts[1:]))
   assert str(ts[0]+timedelta(hours=8))==row['BJT_start']
   assert all(days[str((t+timedelta(hours=8)).date())]==row['split'] for t in ts)
   for p in paths:assert not Path(p).is_absolute() and '..' not in Path(p).parts
   seen[row['split']].update(paths)
  assert not seen['train']&seen['val']
  self.rows=[r for r in all_rows if r['split']==split];assert len(self.rows)==ready['sequence_counts'][split]
  assert len(self.rows)>({'train':30000,'val':2500}[split])
  n=json.loads(norm.read_text());assert tuple(n['channels'])==CHANNELS and n['test_used'] is False
  self.mean=np.asarray(n['mean'],np.float32)[:,None,None];self.std=np.asarray(n['std'],np.float32)[:,None,None]
  assert self.mean.shape==self.std.shape==(13,1,1) and np.isfinite(self.mean).all() and np.isfinite(self.std).all() and (self.std>0).all()
  self.records={}
  with (PREPARED/'frame_audit.jsonl').open() as f:
   for line in f:
    r=json.loads(line)
    if r['relative'] in seen[split]:
     assert not r.get('rejected');self.records[r['relative']]=(r['source_bytes'],r['source_mtime_ns'])
  assert len(self.records)==len(seen[split])
  self.root,self.split,self.ready=DATA_ROOT,split,ready
 def __len__(self):return len(self.rows)
