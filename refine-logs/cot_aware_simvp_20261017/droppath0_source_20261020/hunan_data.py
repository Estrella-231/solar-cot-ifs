"""Hunan-only full frame data. AGRI UTC; auxiliary solar/QA files BJT."""
import csv
import json
from datetime import datetime,timedelta
from pathlib import Path
from functools import lru_cache
import numpy as np
from torch.utils.data import Dataset

CHANNELS=(0,1,2,3,4,5,8,9,10,11,12,13,14)


@lru_cache(maxsize=64)
def aux_month_index(directory):
    index={}
    for path in Path(directory).glob('*_aux.npz'):
        parts=path.stem.split('_')
        if len(parts)==3 and parts[1] in ('0','1'):
            index.setdefault(parts[0],[]).append(path)
    return index


def resolve_aux(root,t):
    directory=Path(root)/'aux_256'/t.strftime('%Y%m')
    matches=aux_month_index(str(directory)).get(t.strftime('%Y%m%d%H%M'),[])
    if len(matches)!=1:
        raise ValueError(f'Expected one daytime/nighttime auxiliary file for {t}: {matches}')
    return matches[0]


def load_frame(root,rel):
    root=Path(root)
    t=datetime.strptime(Path(rel).stem,'%Y%m%d%H%M')+timedelta(hours=8)
    aux=resolve_aux(root,t)
    raw=np.load(root/'data'/rel,mmap_mode='r',allow_pickle=False)
    if raw.shape!=(20,256,256) or raw.dtype!=np.float32: raise ValueError(str(rel))
    agri=np.asarray(raw[list(CHANNELS)],dtype=np.float32)
    with np.load(aux,allow_pickle=False) as z:
        bits=z['agri_valid_bits']
        valid=np.stack([(bits&(1<<k))!=0 for k in CHANNELS]) & np.isfinite(agri)
        geometry=np.stack([z[k] for k in ('cosSOZ','cosRAA','day_mask')])
    if not np.isfinite(geometry).all(): raise ValueError('invalid solar geometry: '+str(aux))
    return agri,valid,geometry


class HunanDataset(Dataset):
    def __init__(self,root,manifest,norm,split):
        self.root=root
        with open(manifest,newline='') as f:
            self.rows=[r for r in csv.DictReader(f) if r['split']==split]
        self.mean=np.asarray(norm['mean'],np.float32)[:,None,None]
        self.std=np.asarray(norm['std'],np.float32)[:,None,None]
        if not self.rows: raise ValueError('empty '+split)
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i];images=[];masks=[];geo=[]
        for rel in row['data_relpaths'].split('|'):
            a,v,g=load_frame(self.root,rel)
            images.append(np.where(v,(a-self.mean)/self.std,0).astype(np.float32))
            masks.append(v);geo.append(g)
        return np.stack(images[:8]),np.stack(geo[:16]),np.stack(images[8:]),np.stack(masks[8:])
