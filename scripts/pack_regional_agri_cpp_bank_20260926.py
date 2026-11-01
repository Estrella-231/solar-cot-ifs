"""Stream accepted regional CPP into an isolated real-AGRI R training cache."""
import argparse
import csv
import hashlib
import importlib.util
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    for key in ('cpp-bank', 'data-root', 'manifest', 'hunan-data-code', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    acceptance = json.loads((a.cpp_bank / 'acceptance_20260926.json').read_text())
    bank = json.loads((a.cpp_bank / 'index.json').read_text())
    assert acceptance['state'] == 'PASS_EXPLORATORY_CPP_BANK_INTEGRITY'
    assert bank['state'] == 'COMPLETE_EXPLORATORY_CPP_BANK' and bank['test_used'] is False
    assert acceptance['identity'] == bank['identity']
    assert sha(a.manifest) == bank['identity']['manifest_sha256']
    identity = {'cpp_acceptance_sha256': sha(a.cpp_bank / 'acceptance_20260926.json'),
                'manifest_sha256': sha(a.manifest), 'data_code_sha256': sha(a.hunan_data_code),
                'packer_sha256': sha(__file__), 'data_root': str(a.data_root),
                'mask': 'finite CPP COT 0..100 AND all13 AGRI valid AND day_mask AND cosSOZ>cos80'}
    by_stamp = {}
    with a.manifest.open(newline='') as f:
        for row in csv.DictReader(f):
            if row['split'] in ('train', 'val'):
                for rel in row['data_relpaths'].split('|'):
                    key = (row['split'], Path(rel).stem)
                    if key in by_stamp:
                        assert by_stamp[key] == rel
                    by_stamp[key] = rel
    spec = importlib.util.spec_from_file_location('regional_data', a.hunan_data_code)
    data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data)
    a.output.mkdir(parents=True, exist_ok=True)
    index = a.output / 'index.json'
    if index.exists():
        state = json.loads(index.read_text())
        assert state['identity'] == identity
    else:
        state = {'state': 'BUILDING_REAL_AGRI_CPP_CACHE', 'test_used': False,
                 'identity': identity, 'shards': [], 'source_limit': 'exploratory historic CPP; producer and physical QA unresolved'}
        atomic(index, state)
    for old in state['shards']:
        for name, digest in old['hashes'].items():
            assert sha(a.output / old['directory'] / name) == digest
    for item in bank['shards'][len(state['shards']):]:
        source = a.cpp_bank / item['file']
        assert sha(source) == item['sha256']
        directory = a.output / item['file'].replace('cpp_', 'pair_').replace('.npz', '')
        directory.mkdir(exist_ok=True)
        records, inputs, refs, masks = [], [], [], []
        total, second, count = [np.zeros(16, np.float64) for _ in range(3)]
        with np.load(source, allow_pickle=False) as z:
            cot, cpp_valid = z['cot'], z['valid']
            for i, (stamp, split) in enumerate(zip(z['utc'], z['split'])):
                stamp, split = stamp.decode(), split.decode()
                rel = by_stamp[(split, stamp)]
                agri, valid, geom = data.load_frame(a.data_root, rel)
                assert agri.shape == valid.shape == (13, 256, 256) and geom.shape == (3, 256, 256)
                supervision = cpp_valid[i] & valid.all(0) & (geom[2] > .5) & (geom[0] > np.cos(np.deg2rad(80)))
                rec = {'utc': stamp, 'split': split, 'agri_relpath': rel,
                       'supervised_fraction': float(supervision.mean()),
                       'included_for_R': bool(supervision.any())}
                if supervision.any():
                    raw = np.concatenate((agri, geom)).astype(np.float32)
                    channel_valid = np.concatenate((valid, np.isfinite(geom))) & np.isfinite(raw)
                    raw[~channel_valid] = np.nan
                    rec['array_row'] = len(inputs)
                    inputs.append(raw)
                    refs.append(cot[i])
                    masks.append(supervision)
                    values = cot[i][supervision]
                    rec.update(mean_cot=float(values.mean()), zero_fraction=float((values == 0).mean()),
                               thick_ge30_fraction=float((values >= 30).mean()))
                    rec['stations'] = {}
                    for name, (r, c) in {'sili': (163, 148), 'zhujia': (73, 150)}.items():
                        region = np.zeros((256, 256), bool)
                        region[r-2:r+3, c-2:c+3] = True
                        selected = supervision & region
                        rec['stations'][name] = {'valid_center5_pixels': int(selected.sum()),
                                                 'mean_cot_center5': float(cot[i][selected].mean()) if selected.any() else None}
                    if split == 'train':
                        nv = channel_valid & supervision[None]
                        values = np.where(nv, raw, 0)
                        total += values.sum((1, 2), dtype=np.float64)
                        second += np.square(values, dtype=np.float64).sum((1, 2))
                        count += nv.sum((1, 2))
                records.append(rec)
        arrays = {'x_raw.npy': np.stack(inputs) if inputs else np.empty((0, 16, 256, 256), np.float32),
                  'cot.npy': np.stack(refs) if refs else np.empty((0, 256, 256), np.float32),
                  'mask.npy': np.stack(masks) if masks else np.empty((0, 256, 256), bool)}
        for name, array in arrays.items():
            temp = directory / (name + '.tmp')
            with temp.open('wb') as f:
                np.save(f, array, allow_pickle=False)
            temp.replace(directory / name)
        atomic(directory / 'records.json', records)
        state['shards'].append({'directory': directory.name, 'source_cpp_shard_sha256': item['sha256'],
                                'records': len(records), 'usable_R_frames': len(inputs),
                                'train_sum': total.tolist(), 'train_second': second.tolist(), 'train_count': count.tolist(),
                                'hashes': {name: sha(directory / name) for name in (*arrays, 'records.json')}})
        atomic(index, state)
        print(json.dumps({'shards': len(state['shards']), 'usable_R_frames': sum(s['usable_R_frames'] for s in state['shards'])}), flush=True)
    count = np.sum([s['train_count'] for s in state['shards']], axis=0)
    assert np.all(count > 0)
    mean = np.sum([s['train_sum'] for s in state['shards']], axis=0) / count
    second = np.sum([s['train_second'] for s in state['shards']], axis=0) / count
    std = np.sqrt(np.maximum(second - mean ** 2, 1e-8))
    mean[15], std[15] = 0, 1
    atomic(a.output / 'norm.json', {'mean': mean.tolist(), 'std': std.tolist(),
                                   'fit_scope': 'train-only paired daytime supervised pixels; no val/test'})
    monthly = {}
    unique = {'train': set(), 'val': set()}
    for shard in state['shards']:
        records = json.loads((a.output / shard['directory'] / 'records.json').read_text())
        for rec in records:
            split, stamp = rec['split'], rec['utc']
            assert stamp not in unique[split]
            unique[split].add(stamp)
            bjt = datetime.strptime(stamp, '%Y%m%d%H%M') + timedelta(hours=8)
            if split == 'train':
                assert datetime(2024, 4, 2) <= bjt < datetime(2025, 7, 1)
            else:
                assert datetime(2025, 7, 1) <= bjt < datetime(2025, 10, 1)
            key = split + '/' + bjt.strftime('%Y%m')
            entry = monthly.setdefault(key, {'available_CPP_frames': 0, 'usable_R_frames': 0,
                                            'supervised_pixels': 0, 'cot_sum': 0.})
            entry['available_CPP_frames'] += 1
            entry['usable_R_frames'] += int(rec['included_for_R'])
            n = round(rec['supervised_fraction'] * 256 * 256)
            entry['supervised_pixels'] += n
            entry['cot_sum'] += n * rec.get('mean_cot', 0.)
    assert not unique['train'] & unique['val']
    for rec in monthly.values():
        rec['mean_cot_supervised'] = rec['cot_sum'] / rec['supervised_pixels'] if rec['supervised_pixels'] else None
    atomic(a.output / 'distribution_audit.json', {'state': 'COMPLETE_PAIRED_CACHE_DISTRIBUTION',
                                                'test_used': False, 'month_time_system': 'BJT (UTC+8)',
                                                'monthly': dict(sorted(monthly.items())),
                                                'limits': 'descriptive CPP reference and training representativeness; no physical QA or predictive skill claim'})
    state['state'] = 'COMPLETE_EXPLORATORY_REAL_AGRI_CPP_CACHE'
    state['normalization_sha256'] = sha(a.output / 'norm.json')
    atomic(index, state)


if __name__ == '__main__':
    main()
