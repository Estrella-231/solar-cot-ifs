import json
from pathlib import Path
import numpy as np

root = Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907')
base = root / 'data/head_pack_trainval_20260908_v1'
specs = json.loads((base / 'source_shards.json').read_text())
split = np.load(base / 'split.npy', mmap_mode='r')
sequence = np.load(base / 'sequence.npy', mmap_mode='r')
station = np.load(base / 'station.npy', mmap_mode='r')
lead = np.load(base / 'lead.npy', mmap_mode='r')
print('pack', len(split), 'train', int((split == 0).sum()), 'val', int((split == 1).sum()))
for flag, name in ((0, 'train'), (1, 'val')):
    ids = np.where(split == flag)[0]
    print(name, 'sequence range', int(sequence[ids].min()), int(sequence[ids].max()),
          'station', np.unique(station[ids]).tolist(), 'lead', np.unique(lead[ids]).tolist())
for spec in specs:
    path = root / 'data/frozen_forecast_trainval_20260907_v1' / spec['split'] / spec['name']
    with np.load(path, allow_pickle=False) as z:
        ids = z['indices']
        print('shard', spec['split'], spec['name'], 'indices', ids.shape, int(ids.min()), int(ids.max()),
              'agri', z['predicted_agri_physical'].shape, 'geometry', z['geometry'].shape)
        flag = 0 if spec['split'] == 'train' else 1
        table = np.full((int(sequence[split == flag].max()) + 1, 2, 16), -1, dtype=np.int64)
        row_ids = np.where(split == flag)[0]
        table[sequence[row_ids], station[row_ids], lead[row_ids]] = row_ids
        target = table[ids]
        print('  target missing', int((target < 0).sum()), 'target shape', target.shape,
              'target sample', target.reshape(-1)[:8].tolist())
