"""Hunan QA-preserving on-demand local cache, corrected UTC solar geometry."""
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from datetime import datetime,timedelta
import fcntl,os,shutil
import numpy as np
from data import CHANNELS,BASE,DATA_ROOT,HunanDataset
from solar_geometry_noaa import compute_solar_angles

@lru_cache(maxsize=64)
def aux_index(directory):
 out={}
 for p in Path(directory).glob('*_aux.npz'):
  parts=p.stem.split('_')
  if len(parts)==3 and parts[1] in ('0','1'):out.setdefault(parts[0],[]).append(p)
 return out

@lru_cache(maxsize=1)
def grid():
 with np.load(BASE/'grid_static_hunan.npz') as z:return {k:z[k] for k in ('lat','lon','saa')}

class CachedDataset(HunanDataset):
 def __init__(self,split,cache_frames=16):
  super().__init__(split);self.cache_frames=cache_frames;self.frame_cache=OrderedDict()
  self.cache=Path(os.environ['HUNAN_DY_CACHE']);self.cache.mkdir(parents=True,exist_ok=True)
 def _source(self,rel):
  p=Path(rel);assert not p.is_absolute() and '..' not in p.parts
  utc=datetime.strptime(p.stem,'%Y%m%d%H%M');bjt=utc+timedelta(hours=8)
  matches=aux_index(str(DATA_ROOT/'aux_256'/bjt.strftime('%Y%m'))).get(bjt.strftime('%Y%m%d%H%M'),[])
  if len(matches)!=1:raise ValueError(('aux UTC/BJT match',rel,matches))
  raw=np.load(DATA_ROOT/'data'/rel,mmap_mode='r',allow_pickle=False)
  assert raw.shape==(20,256,256) and raw.dtype==np.float32
  x=np.asarray(raw[list(CHANNELS)],np.float32)
  with np.load(matches[0],allow_pickle=False) as z:
   valid=np.stack([(z['agri_valid_bits']&(1<<k))!=0 for k in CHANNELS]) & np.isfinite(x)
  static=grid();soz,soa,day=compute_solar_angles(static['lat'],static['lon'],utc)
  geom=np.stack([np.cos(np.deg2rad(soz)),np.cos(np.deg2rad(soa-static['saa'])),day]).astype(np.float32)
  bits=np.zeros((256,256),np.uint16)
  for k in range(13):bits|=valid[k].astype(np.uint16)<<k
  assert np.isfinite(geom).all()
  return x,bits,geom
 def _frame(self,rel):
  if rel in self.frame_cache:
   self.frame_cache.move_to_end(rel);return self.frame_cache[rel]
  dst=self.cache/(rel+'.npz');dst.parent.mkdir(parents=True,exist_ok=True)
  if not dst.exists():
   with dst.with_suffix('.lock').open('w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    if not dst.exists():
     if shutil.disk_usage(self.cache).free<25*2**30:raise RuntimeError('node-local cache disk reserve below25GiB')
     x,bits,g=self._source(rel);tmp=dst.with_name(dst.name+f'.{os.getpid()}.tmp')
     with tmp.open('wb') as f:np.savez(f,x=x,bits=bits,g=g)
     tmp.replace(dst)
  with np.load(dst,allow_pickle=False) as z:x=z['x'];bits=z['bits'];geom=z['g']
  valid=np.stack([(bits&(1<<k))!=0 for k in range(13)])
  normalized=np.zeros_like(x);np.subtract(x,self.mean,out=normalized,where=valid);np.divide(normalized,self.std,out=normalized,where=valid)
  result=(normalized,valid,geom)
  if self.cache_frames:
   self.frame_cache[rel]=result
   while len(self.frame_cache)>self.cache_frames:self.frame_cache.popitem(last=False)
  return result
 def __getitem__(self,index):
  frames=[self._frame(p) for p in self.rows[index]['data_relpaths'].split('|')]
  assert len(frames)==24
  return {'history':np.stack([x[0] for x in frames[:8]]),'future':np.stack([x[0] for x in frames[8:]]),
   'geometry':np.stack([x[2] for x in frames]),'history_valid':np.stack([x[1] for x in frames[:8]]),'future_valid':np.stack([x[1] for x in frames[8:]])}
class IndexedDataset:
 def __init__(self,dataset):self.dataset=dataset
 def __len__(self):return len(self.dataset)
 def __getitem__(self,index):return {**self.dataset[index],'sample_index':index}
