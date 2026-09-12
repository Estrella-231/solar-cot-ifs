"""Accept every frozen train/val shard and pack identical A/AC labeled rows.

CPU only. Source assets are read-only; a completion marker is written last.
Statistics use labeled TRAIN rows only. No test payload or target clipping.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


EXPECTED = {'train': 538053, 'validation': 99849}
SEQUENCES = {'train': 27675, 'val': 4258}
LABEL_SHA = '01587fda1acc500845e3b2fc5d1e0cc2f8028330f9b543b27b6e9d01fb8ddcd0'
STATIONS = ['sili', 'zhujia']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    temp.replace(path)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def moment_stats(total, square, count):
    mean = total / count
    var = np.maximum(square / count - mean**2, 0)
    scale = np.sqrt(var)
    require(np.isfinite(mean).all() and np.isfinite(scale).all(), 'nonfinite train statistics')
    require((scale > 1e-12).all(), 'constant training feature needs explicit new contract')
    return {'mean': mean.tolist(), 'std': scale.tolist(), 'count': int(count)}


def self_test():
    train = np.array([[1., 2.], [3., 6.]])
    stats = moment_stats(train.sum(0), (train**2).sum(0), len(train))
    require(np.allclose(stats['mean'], [2, 4]), 'train mean test')
    require(np.allclose(stats['std'], [1, 2]), 'train std test')
    untouched_val = np.array([[999., -999.]])
    require(np.array_equal(untouched_val, [[999., -999.]]), 'val immutability test')
    groups = np.array([0, 0, 1])
    errors = np.array([2., 6., 12.])
    counts = np.bincount(groups, minlength=2)
    weights = len(groups) / (2 * counts[groups])
    require(np.isclose((weights * errors).mean(), (4 + 12) / 2), 'group unbiased loss test')
    require((np.array([5.5, 9000.], dtype=np.float64) > 1).all(), 'no target clipping test')
    print('SELF_TEST_PASS', flush=True)


def main(args):
    project = args.project.resolve()
    cache = project / 'data/frozen_forecast_trainval_20260907_v1'
    labels_dir = project / 'data/forecast_label_join_20260907_v1'
    out = args.output.resolve()
    require(out.parent == project / 'data', 'pack output must be directly under project/data')
    require(not out.exists(), 'refuse to overwrite existing pack, including incomplete packs')
    source_complete = json.loads((cache / 'complete.json').read_text())
    require(source_complete['state'] == 'COMPLETE_FEATURE_CACHE' and source_complete['test_used'] is False,
            'feature cache is not complete or test-free')
    contract = json.loads((cache / 'contract.json').read_text())
    sc = json.loads((project / 'configs/s_frozen_hunan_seed42.json').read_text())
    rc = json.loads((project / 'configs/r_frozen_repaired_seed42.json').read_text())
    require(contract['S'] == sc and contract['R'] == rc and contract['test_used'] is False, 'frozen contract drift')
    for key in ('checkpoint', 'manifest', 'normalization'):
        require(sha(sc[key]) == sc[key + '_sha256'], 'S asset hash: ' + key)
    require(sha(rc['checkpoint']) == rc['checkpoint_sha256'], 'R checkpoint hash')
    require(list(sc['station_patches']) == STATIONS, 'station order mismatch')
    labels = labels_dir / 'labels.csv'
    label_audit = json.loads((labels_dir / 'audit.json').read_text())
    require(label_audit['state'] == 'PASS_LABEL_TO_SEQUENCE_JOIN', 'unapproved labels')
    require(label_audit['counts'] == EXPECTED and label_audit['exclusions_added'] == 0, 'label count/exclusion drift')
    require(sha(labels) == label_audit['labels_sha256'] == LABEL_SHA, 'label hash drift')
    require(label_audit['manifest_sha256'] == sc['manifest_sha256'], 'manifest label mismatch')
    validation = json.loads((cache / 'validation_reload.json').read_text())
    require(validation['status'] == 'PASS', 'missing frozen validation reload')

    indices = {}
    splits = {s: [] for s in SEQUENCES}
    frame_sets = {s: set() for s in SEQUENCES}
    with Path(sc['manifest']).open(newline='') as f:
        for row in csv.DictReader(f):
            if row['split'] not in splits:
                continue  # test manifest metadata are not used or read as payloads
            s = row['split']
            rels = row['data_relpaths'].split('|')
            ts = [datetime.strptime(Path(v).stem, '%Y%m%d%H%M') for v in rels]
            require(len(ts) == 24 and all(t == ts[0] + timedelta(minutes=i * 15) for i, t in enumerate(ts)),
                    'noncanonical manifest sequence')
            require(datetime.fromisoformat(row['BJT_start']) == ts[0] + timedelta(hours=8), 'manifest BJT/UTC')
            splits[s].append((row['seq_id'], ts[7]))
            frame_sets[s].update(rels)
    require(not (frame_sets['train'] & frame_sets['val']), 'train/val share source frames')
    completion = {}
    for s, count in SEQUENCES.items():
        folder = cache / s
        report = json.loads((folder / 'complete.json').read_text())
        require(report['state'] == 'COMPLETE' and report['samples'] == count and report['test_used'] is False,
                'bad split completion ' + s)
        require(report['stations'] == STATIONS and report['center_pixel'] == [8, 8], 'station/center drift')
        require(report['future_AGRI_input'] is False and report['future_CPP_input'] is False, 'causal cache drift')
        require(sha(folder / 'index.csv') == report['index_sha256'], 'index hash mismatch')
        require(len(splits[s]) == count, 'manifest sequence count mismatch')
        with (folder / 'index.csv').open(newline='') as f:
            for i, row in enumerate(csv.DictReader(f)):
                seq, init = splits[s][i]
                parsed = datetime.fromisoformat(row['init_time_utc'])
                require(int(row['index']) == i and row['seq_id'] == seq and row['split'] == s,
                        'cache index identity mismatch')
                require(parsed.utcoffset() == timedelta(0) and parsed.replace(tzinfo=None) == init,
                        'cache initialization mismatch')
            require(i + 1 == count, 'cache index row count mismatch')
        names = [r['name'] for r in report['shards']]
        require(names == [f'shard_{i:05d}.npz' for i in range(len(names))], 'shard names not sequential')
        require(set(names) == {p.name for p in folder.glob('shard_*.npz')}, 'missing/extra source shards')
        completion[s] = report
        indices[s] = np.full((count, 2, 16), -1, dtype=np.int64)

    out.mkdir()
    started = time.monotonic()
    n = sum(EXPECTED.values())
    shapes = {'x': ('float32', (n, 13, 16, 16)), 'c': ('float32', (n, 4)), 'g': ('float32', (n, 3)),
              'kt': ('float64', (n,)), 'ghi': ('float64', (n,)), 'clear': ('float64', (n,)),
              'station': ('int64', (n,)), 'lead': ('int64', (n,)), 'split': ('int8', (n,)),
              'sequence': ('int64', (n,)), 'weight': ('float32', (n,))}
    arrays = {k: np.lib.format.open_memmap(out / (k + '.npy'), mode='w+', dtype=dt, shape=shape)
              for k, (dt, shape) in shapes.items()}
    counts = Counter()
    group_counts = np.zeros((2, 16), dtype=np.int64)
    with labels.open(newline='') as f:
        for i, row in enumerate(csv.DictReader(f)):
            require(i < n and row['split'] in EXPECTED, 'extra or test label')
            s = 'val' if row['split'] == 'validation' else 'train'
            split_id = int(s == 'val')
            index, station, lead = [int(row[k]) for k in ('cache_index', 'station_index', 'lead_index')]
            require(0 <= index < SEQUENCES[s] and station in (0, 1) and 0 <= lead < 16, 'label feature index range')
            require(indices[s][index, station, lead] == -1, 'duplicate station/init/lead row')
            seq, init_naive = splits[s][index]
            init = datetime.fromisoformat(row['init_time_utc'])
            target = datetime.fromisoformat(row['target_time_utc'])
            init_bjt = datetime.fromisoformat(row['init_time_bjt'])
            target_bjt = datetime.fromisoformat(row['target_time_bjt'])
            require(init.utcoffset() == target.utcoffset() == timedelta(0), 'label UTC offset')
            require(init_bjt.utcoffset() == target_bjt.utcoffset() == timedelta(hours=8), 'label BJT offset')
            require(init == init_bjt and target == target_bjt and init.replace(tzinfo=None) == init_naive, 'label time equivalence')
            require(target - init == timedelta(minutes=(lead + 1) * 15) and int(row['lead_minutes']) == (lead + 1) * 15,
                    'canonical forecast lead mismatch')
            record_times = [datetime.fromisoformat(v) for v in row['label_record_times_bjt'].split('|')]
            require(record_times == [target_bjt - timedelta(minutes=v) for v in (10, 5, 0)], 'label end interval mismatch')
            require(row['simvp_seq_id'] == seq and row['station'] == STATIONS[station], 'station/sequence label mismatch')
            require(row['day_mask_sza_lt_90'] == 'True', 'unexpected non-daytime label')
            ghi, clear, kt = [float(row[k]) for k in ('observed_ghi_15min_wm2', 'clear_sky_ghi_15min_wm2', 'kt')]
            require(all(math.isfinite(v) for v in (ghi, clear, kt)) and ghi >= 0 and clear > 0 and kt >= 0, 'invalid label')
            require(abs(ghi - clear * kt) < 1e-8, 'kt/GHI reconstruction mismatch')
            indices[s][index, station, lead] = i
            for key, value in {'kt': kt, 'ghi': ghi, 'clear': clear, 'station': station, 'lead': lead,
                               'split': split_id, 'sequence': index}.items():
                arrays[key][i] = value
            counts[row['split']] += 1
            if s == 'train':
                group_counts[station, lead] += 1
    require(i + 1 == n and dict(counts) == EXPECTED and (group_counts > 0).all(), 'incomplete label cohort')
    shutil.copyfile(labels, out / 'rows.csv')
    require(sha(out / 'rows.csv') == LABEL_SHA, 'output label identity drift')
    totals = {k: np.zeros(d, dtype=np.float64) for k, d in (('x', 13), ('c', 4), ('g', 2))}
    squares = {k: np.zeros_like(v) for k, v in totals.items()}
    filled = np.zeros(n, dtype=bool)
    checked_shards = []
    for s in ('train', 'val'):
        offset = 0
        for shard in completion[s]['shards']:
            path = cache / s / shard['name']
            receipt = json.loads(path.with_suffix('.json').read_text())
            require(receipt == shard and sha(path) == receipt['sha256'], 'shard receipt/hash mismatch ' + str(path))
            b = int(shard['samples'])
            with np.load(path, allow_pickle=False) as z:
                require(set(z.files) == {'indices', 'predicted_agri_physical', 'geometry', 'cot_log1p', 'cot_features', 'r_latent_mean'},
                        'unexpected cache fields')
                require(np.array_equal(z['indices'], np.arange(offset, offset + b)), 'shard index coverage')
                payload = {key: z[key] for key in z.files if key != 'indices'}
                expected_shapes = {'predicted_agri_physical': (b, 2, 16, 13, 16, 16),
                                   'geometry': (b, 2, 16, 3, 16, 16), 'cot_log1p': (b, 2, 16, 16, 16),
                                   'cot_features': (b, 2, 16, 4), 'r_latent_mean': (b, 2, 16, 16)}
                for key, shape in expected_shapes.items():
                    require(payload[key].shape == shape and payload[key].dtype == np.float32
                            and np.isfinite(payload[key]).all(), 'nonfinite/type/shape ' + key)
                geom = payload['geometry']
                require((np.abs(geom[:, :, :, :2]) <= 1.00001).all(), 'solar cosine range')
                require(np.isin(geom[:, :, :, 2], [0., 1.]).all(), 'nonbinary source day mask')
                cot = payload['cot_log1p']
                require((cot >= 0).all(), 'negative predicted log1p COT')
                flat = cot.reshape(b, 2, 16, 256)
                reconstructed = np.stack([flat.mean(-1), flat.std(-1), np.quantile(flat, .9, axis=-1), cot[..., 8, 8]], -1)
                require(np.allclose(reconstructed, payload['cot_features'], atol=2e-6, rtol=2e-5), 'explicit COT feature mismatch')
                local, station, lead = np.where(indices[s][offset:offset + b] >= 0)
                rows = indices[s][offset:offset + b][local, station, lead]
                require(not filled[rows].any(), 'duplicate feature output row')
                x = payload['predicted_agri_physical'][local, station, lead]
                c = payload['cot_features'][local, station, lead]
                g = geom[local, station, lead].mean((-1, -2), dtype=np.float64).astype(np.float32)
                arrays['x'][rows], arrays['c'][rows], arrays['g'][rows] = x, c, g
                filled[rows] = True
                if s == 'train' and len(rows):
                    for key, v, axis in [('x', x.astype(np.float64), (0, 2, 3)),
                                         ('c', c.astype(np.float64), 0), ('g', g[:, :2].astype(np.float64), 0)]:
                        totals[key] += v.sum(axis=axis)
                        squares[key] += np.square(v).sum(axis=axis)
                checked_shards.append({'split': s, **shard})
            offset += b
            if len(checked_shards) % 50 == 0:
                status = {'state': 'ACCEPTING_AND_PACKING', 'shards': len(checked_shards), 'total_shards': 1332,
                          'labeled_rows_written': int(filled.sum()), 'seconds': time.monotonic() - started}
                atomic(out / 'status.json', status)
                print(json.dumps(status), flush=True)
        require(offset == SEQUENCES[s], 'missing split sequence coverage')
    require(len(checked_shards) == 1332 and filled.all(), 'incomplete full-cache acceptance')
    norm = {key: moment_stats(totals[key], squares[key], EXPECTED['train'] * (256 if key == 'x' else 1))
            for key in totals}
    norm.update({'fit_split': 'train', 'fit_rows': EXPECTED['train'], 'x_channels': 'C01-C06,C09-C15',
                 'c_features': ['mean_log1p_cot', 'std_log1p_cot', 'p90_log1p_cot', 'center_log1p_cot'],
                 'g_features': ['patch_mean_cosSOZ', 'patch_mean_cosRAA', 'patch_mean_day_mask'],
                 'g_day_mask': 'spatial mean kept unchanged, no normalization',
                 'target': 'kt=float64 mean(GHI)/mean(clear_sky_GHI); no clipping',
                 'variance': 'population; float64 sums and squares; no validation statistics'})
    atomic(out / 'norm.json', norm)
    for start in range(0, n, 1024):
        stop = min(n, start + 1024)
        for key in ('x', 'c', 'g'):
            feature = arrays[key][start:stop]
            stats = norm[key]
            mean, std = np.array(stats['mean'], dtype=np.float32), np.array(stats['std'], dtype=np.float32)
            if key == 'x':
                feature[:] = (feature - mean[None, :, None, None]) / std[None, :, None, None]
            elif key == 'g':
                feature[:, :2] = (feature[:, :2] - mean) / std
            else:
                feature[:] = (feature - mean) / std
            require(np.isfinite(feature).all(), 'nonfinite normalized feature')
        train = arrays['split'][start:stop] == 0
        group_size = group_counts[arrays['station'][start:stop], arrays['lead'][start:stop]]
        arrays['weight'][start:stop] = np.where(train, EXPECTED['train'] / (32 * group_size), 1).astype(np.float32)
    for v in arrays.values():
        v.flush()
    kt = arrays['kt']
    as_float = kt.astype(np.float32)
    require(np.isfinite(as_float).all(), 'kt not representable in float32')
    roundtrip = as_float.astype(np.float64)
    target_audit = {}
    for s, sid in [('train', 0), ('validation', 1)]:
        select = arrays['split'] == sid
        target_audit[s] = {'min': float(kt[select].min()), 'max': float(kt[select].max()),
                           'quantile_levels': [0.5, 0.9, 0.99, 0.999, 1.0],
                           'quantiles': np.quantile(kt[select], [0.5, 0.9, 0.99, 0.999, 1.0]).tolist(),
                           'count_gt_10': int((kt[select] > 10).sum()), 'count_gt_100': int((kt[select] > 100).sum())}
    target_audit['float32_max_absolute_error'] = float(np.abs(kt - roundtrip).max())
    target_audit['float32_max_relative_error'] = float((np.abs(kt - roundtrip) / np.maximum(np.abs(kt), 1e-300)).max())
    require(sha(labels) == LABEL_SHA, 'source labels changed during pack')
    atomic(out / 'status.json', {'state': 'HASHING_OUTPUT_ARRAYS', 'shards': len(checked_shards), 'rows': n})
    artifacts = {key: {'name': key + '.npy', 'dtype': dt, 'shape': list(shape), 'sha256': sha(out / (key + '.npy'))}
                 for key, (dt, shape) in shapes.items()}
    atomic(out / 'source_shards.json', checked_shards)
    report = {'state': 'COMPLETE', 'status': 'PASS_FULL_CACHE_AND_LABEL_PACK', 'rows': n, 'counts': EXPECTED,
              'sequences': SEQUENCES, 'shards_checked': len(checked_shards), 'all_source_arrays_finite': True,
              'all_source_shapes_verified': True, 'all_source_receipts_and_sha_verified': True,
              'all_index_and_label_identities_verified': True, 'all_explicit_cot_features_reconstructed': True,
              'train_val_frames_disjoint': True, 'test_used': False, 'label_filters_added': 0, 'kt_clipping': False,
              'S_frozen': True, 'R_frozen': True, 'source_labels': str(labels), 'source_labels_sha256': LABEL_SHA,
              'source_cache': str(cache), 'source_cache_contract_sha256': sha(cache / 'contract.json'),
              'source_cache_contract': contract, 'source_shards_sha256': sha(out / 'source_shards.json'),
              'source_split_complete_sha256': {s: sha(cache / s / 'complete.json') for s in SEQUENCES},
              'rows_csv_sha256': sha(out / 'rows.csv'), 'norm_sha256': sha(out / 'norm.json'),
              'script_sha256': sha(__file__), 'arrays': artifacts, 'kt_audit': target_audit,
              'station_order': STATIONS, 'train_station_lead_counts': group_counts.tolist(),
              'weight_formula': 'Ntrain/(32*n_train_station_lead); ordinary minibatch weighted-MSE mean is unbiased equal-32-group MSE',
              'validation_weight': 'stored as 1; selection must explicitly compute equal station/lead group metric',
              'split_values': {'train': 0, 'validation': 1}, 'lead_values': '0..15 means +15..+240 min',
              'sequence_values': 'cache split-local index; combine with split for globally unique sequence',
              'seconds': time.monotonic() - started, 'completed_utc': datetime.utcnow().isoformat() + 'Z'}
    atomic(out / 'audit.json', report)
    atomic(out / 'status.json', {'state': 'COMPLETE', 'rows': n, 'shards': len(checked_shards), 'seconds': report['seconds']})
    print(json.dumps({'state': 'COMPLETE', 'counts': EXPECTED, 'kt_audit': target_audit, 'seconds': report['seconds']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    options = parser.parse_args()
    if options.self_test:
        self_test()
    else:
        require(options.project is not None and options.output is not None, 'project and output required')
        main(options)
