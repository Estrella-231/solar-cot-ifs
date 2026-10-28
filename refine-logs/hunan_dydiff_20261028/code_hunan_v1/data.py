"""Hunan frozen split and existing train-only normalization; no payload rescan."""
import csv, hashlib, json, os
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
CHANNELS=(0,1,2,3,4,5,8,9,10,11,12,13,14)
BASE=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/HunanDyDiff_20261028')
PREPARED=BASE/'prepared_v1'
DATA_ROOT=Path('/public/home/slfu/ttzhou/Auxiliary_data/HuNan_AGRI_Preprocessed')
SOURCE_MANIFEST_SHA256='f6703ed5e42f1ae00aac6538b108dc4d9361197ad57ad86a0f407767aa2e4cb2'
class HunanDataset(Dataset):
 def __init__(self,split='train'):
  if split not in ('train','val'):raise ValueError('test payloads closed')
  manifest=PREPARED/'hunan_8to16.csv';norm=PREPARED/'hunan_train_norm.json'
  self.manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()
  assert self.manifest_sha256==SOURCE_MANIFEST_SHA256
  self.normalization_sha256=hashlib.sha256(norm.read_bytes()).hexdigest()
  n=json.loads(norm.read_text());assert n['manifest_sha256']==self.manifest_sha256
  assert tuple(n['source_indices'])==CHANNELS and n['validation_test_used'] is False
  assert n['train_unique_frames']==51236
  self.mean=np.asarray(n['mean'],np.float32)[:,None,None];self.std=np.asarray(n['std'],np.float32)[:,None,None]
  assert np.isfinite(self.mean).all() and np.isfinite(self.std).all() and (self.std>0).all()
  with manifest.open() as f:self.rows=[r for r in csv.DictReader(f) if r['split']==split]
  assert len(self.rows)=={'train':35320,'val':3140}[split]
  self.split=split;self.ready={'source_manifest_sha256':self.manifest_sha256}
 def __len__(self):return len(self.rows)
