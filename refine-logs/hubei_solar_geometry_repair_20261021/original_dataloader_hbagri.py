"""SimVP dataloader: 从 split CSV 读取 data/ + aux_256/ 对齐序列。"""

import os, csv
import numpy as np
import torch
from torch.utils.data import Dataset

from openstl.datasets.utils import create_loader

AGRI_CH_INDICES = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]
AUX_NAMES = ['cosSOZ', 'cosRAA', 'day_mask']
N_AGRI_CH = len(AGRI_CH_INDICES)
N_AUX_CH = len(AUX_NAMES)

CH_INDICES = AGRI_CH_INDICES
N_CH = N_AGRI_CH
CH_NAMES = ['C01 0.47um', 'C02 0.65um', 'C03 0.83um',
            'C04 1.38um', 'C05 1.61um', 'C06 2.22um',
            'C09 7.10um', 'C10 7.80um', 'C11 8.60um',
            'C12 10.3um', 'C13 11.8um', 'C14 12.8um', 'C15 13.5um']

DEFAULT_CSV = '/home/Data_Pool_3/chenyi/Auxiliary_data/HuBei_AGRI_Preprocessed/split_simvp_2h4h_15min.csv'
_CSV_SPLITS = {}


def _load_csv(csv_path):
    """一次性加载 CSV 全量，缓存"""
    csv_path = os.path.realpath(csv_path)
    if csv_path in _CSV_SPLITS:
        return _CSV_SPLITS[csv_path]
    splits = {'train': [], 'val': [], 'test': []}
    with open(csv_path, 'r') as f:
        for row in csv.DictReader(f):
            s = row['split']
            if s not in splits: continue
            splits[s].append({
                'data': row['data_paths'].split('|'),
                'aux': row['aux_paths'].split('|'),
            })
    _CSV_SPLITS[csv_path] = splits
    for k in splits:
        print(f"  HBAGRI {k}: {len(splits[k])} sequences (from CSV)")
    return splits


class HBAGRIDataset(Dataset):
    """FY-4B AGRI Hubei 256x256, AGRI + aux, 从 CSV 读取索引。"""

    def __init__(self, data_root, pre_seq_length=8, aft_seq_length=16,
                 temporal_stride=1, split='train', data_name='hbagri',
                 data_dir=None, aux_dir=None, **kwargs):
        super().__init__()
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.total_length = pre_seq_length + aft_seq_length
        self.split = split
        self.data_name = data_name

        csv_path = kwargs.get('index_csv') or DEFAULT_CSV
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        all_splits = _load_csv(csv_path)
        self.samples = all_splits[split]
        invalid_lengths = [
            (len(sample['data']), len(sample['aux']))
            for sample in self.samples
            if len(sample['data']) != self.total_length
            or len(sample['aux']) != self.total_length
        ]
        if invalid_lengths:
            raise ValueError(
                f"{split} contains sequences that do not match total_length="
                f"{self.total_length}; first bad data/aux lengths={invalid_lengths[0]}"
            )

        if split == 'train':
            self._compute_stats()
        else:
            self.mean = None; self.std = None
            self.input_mean = None; self.input_std = None

    def _compute_stats(self):
        max_files = 200
        data_files, aux_files = [], []
        for sample in self.samples:
            data_files.extend(sample['data'][:self.pre_seq_length])
            aux_files.extend(sample['aux'][:self.total_length])
            if len(data_files) >= max_files: break
        data_files = data_files[::max(1, len(data_files) // max_files)][:max_files]
        aux_files = aux_files[::max(1, len(aux_files) // max_files)][:max_files]

        agri_sum, agri_sq, agri_n = np.zeros(N_AGRI_CH, np.float64), np.zeros(N_AGRI_CH, np.float64), np.zeros(N_AGRI_CH, np.float64)
        for fp in data_files:
            raw = np.load(fp)[AGRI_CH_INDICES].astype(np.float64)
            for i, sc in enumerate(AGRI_CH_INDICES):
                ch = raw[i]; valid = np.isfinite(ch)
                if sc < 6: valid &= ch != 0
                if valid.any():
                    agri_sum[i] += ch[valid].sum(); agri_sq[i] += (ch[valid]**2).sum(); agri_n[i] += valid.sum()
        agri_n = np.maximum(agri_n, 1)
        self.mean = (agri_sum / agri_n).astype(np.float32)
        self.std = np.sqrt(np.maximum(agri_sq / agri_n - self.mean**2, 1e-8)).astype(np.float32)
        self.std = np.maximum(self.std, 1e-6)

        aux_sum, aux_sq = np.zeros(N_AUX_CH, np.float64), np.zeros(N_AUX_CH, np.float64)
        aux_n = 0
        for fp in aux_files:
            aux = self._load_aux(fp).astype(np.float64)
            aux_sum += aux.reshape(N_AUX_CH, -1).sum(axis=1)
            aux_sq += (aux.reshape(N_AUX_CH, -1)**2).sum(axis=1)
            aux_n += aux.shape[1] * aux.shape[2]
        aux_n = max(aux_n, 1)
        aux_mean = (aux_sum / aux_n).astype(np.float32)
        aux_std = np.sqrt(np.maximum(aux_sq / aux_n - aux_mean**2, 1e-8)).astype(np.float32)
        aux_std = np.maximum(aux_std, 1e-6)

        self.input_mean = np.concatenate([self.mean, aux_mean]).astype(np.float32)
        self.input_std = np.concatenate([self.std, aux_std]).astype(np.float32)
        print(f"  AGRI mean: {[f'{m:.3f}' for m in self.mean]}")
        print(f"  AGRI std:  {[f'{s:.3f}' for s in self.std]}")

    @staticmethod
    def _load_aux(path):
        aux = np.load(path)
        return np.stack([aux[n].astype(np.float32) for n in AUX_NAMES], axis=0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]
        agri = torch.from_numpy(np.stack([np.load(fp)[AGRI_CH_INDICES].astype(np.float32) for fp in s['data']], axis=0))
        aux = torch.from_numpy(np.stack([self._load_aux(fp) for fp in s['aux']], axis=0))

        inp = torch.cat([agri[:self.pre_seq_length], aux[:self.pre_seq_length]], dim=1)
        tgt = agri[self.pre_seq_length:]

        if self.input_mean is not None:
            im = torch.from_numpy(self.input_mean).view(1,-1,1,1); ist = torch.from_numpy(self.input_std).view(1,-1,1,1)
            inp = (inp - im) / ist
            gm = torch.from_numpy(self.mean).view(1,-1,1,1); gs = torch.from_numpy(self.std).view(1,-1,1,1)
            tgt = (tgt - gm) / gs
            am = torch.from_numpy(self.input_mean[N_AGRI_CH:]).view(1,-1,1,1); astd = torch.from_numpy(self.input_std[N_AGRI_CH:]).view(1,-1,1,1)
            aux = (aux - am) / astd

        return inp.float(), tgt.float(), aux.float()


def load_data(batch_size, val_batch_size, data_root, num_workers=4,
              pre_seq_length=8, aft_seq_length=16, in_shape=None,
              distributed=False, use_augment=False, use_prefetcher=False,
              drop_last=False, temporal_stride=1, **kwargs):
    common = dict(data_root=data_root, pre_seq_length=pre_seq_length,
                  aft_seq_length=aft_seq_length, temporal_stride=temporal_stride,
                  data_name=kwargs.get('data_name', 'hbagri'),
                  index_csv=kwargs.get('index_csv'))
    train_set = HBAGRIDataset(split='train', **common)
    val_set = HBAGRIDataset(split='val', **common)
    test_set = HBAGRIDataset(split='test', **common)
    val_set.mean = train_set.mean; val_set.std = train_set.std
    val_set.input_mean = train_set.input_mean; val_set.input_std = train_set.input_std
    test_set.mean = train_set.mean; test_set.std = train_set.std
    test_set.input_mean = train_set.input_mean; test_set.input_std = train_set.input_std
    dl_train = create_loader(train_set, batch_size=batch_size, shuffle=True, is_training=True, pin_memory=True, drop_last=drop_last, num_workers=num_workers, distributed=distributed, use_prefetcher=use_prefetcher)
    dl_val = create_loader(val_set, batch_size=val_batch_size, shuffle=False, is_training=False, pin_memory=True, drop_last=drop_last, num_workers=num_workers, distributed=distributed, use_prefetcher=use_prefetcher)
    dl_test = create_loader(test_set, batch_size=val_batch_size, shuffle=False, is_training=False, pin_memory=True, drop_last=drop_last, num_workers=num_workers, distributed=distributed, use_prefetcher=use_prefetcher)
    return dl_train, dl_val, dl_test
