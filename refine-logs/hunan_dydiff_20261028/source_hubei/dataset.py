"""Full Hubei frames with corrected geometry and explicit conservative support.

Zero-filled daytime reflective values are missing observations, not labels.
Only historical masks enter the model; future support is used for loss/metrics.
"""
from collections import OrderedDict
from pathlib import Path
import numpy as np
from data import CHANNELS,PREPARED,HubeiFullDataset
class CachedDataset(HubeiFullDataset):
 def __init__(self,split,cache_frames=64):
  super().__init__(split);self.cache_frames=cache_frames;self.frame_cache=OrderedDict()
 def _frame(self,rel):
  if rel in self.frame_cache:
   self.frame_cache.move_to_end(rel);return self.frame_cache[rel]
  assert rel in self.records
  p=self.root/rel;st=p.stat();assert (st.st_size,st.st_mtime_ns)==self.records[rel],('source changed after audit',rel)
  raw=np.load(p,mmap_mode='r',allow_pickle=False);assert raw.shape==(20,256,256) and raw.dtype==np.float32
  x=np.asarray(raw[list(CHANNELS)],np.float32)
  with np.load(PREPARED/'aux'/f'{p.stem}.npz',allow_pickle=False) as z:
   valid=np.stack([(z['valid_bits']&(1<<k))!=0 for k in range(13)])
   geom=np.stack([z[k] for k in ('cosSOZ','cosRAA','day_mask')]).astype(np.float32)
  assert geom.shape==(3,256,256) and np.isfinite(geom).all() and np.isfinite(x[valid]).all()
  normalized=np.zeros_like(x);np.subtract(x,self.mean,out=normalized,where=valid);np.divide(normalized,self.std,out=normalized,where=valid)
  result=(normalized,valid,geom)
  if self.cache_frames:
   self.frame_cache[rel]=result
   while len(self.frame_cache)>self.cache_frames:self.frame_cache.popitem(last=False)
  return result
 def __getitem__(self,index):
  frames=[self._frame(p) for p in self.rows[index]['data_relpaths'].split('|')]
  return {'history':np.stack([x[0] for x in frames[:8]]),'future':np.stack([x[0] for x in frames[8:]]),
          'geometry':np.stack([x[2] for x in frames]),'history_valid':np.stack([x[1] for x in frames[:8]]),'future_valid':np.stack([x[1] for x in frames[8:]])}
class IndexedDataset:
 def __init__(self,dataset):self.dataset=dataset
 def __len__(self):return len(self.dataset)
 def __getitem__(self,index):return {**self.dataset[index],'sample_index':index}
