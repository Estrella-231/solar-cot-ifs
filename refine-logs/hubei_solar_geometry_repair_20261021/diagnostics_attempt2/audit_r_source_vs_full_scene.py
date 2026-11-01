#!/usr/bin/env python3
"""Read-only train/validation audit of R patch inputs against canonical full AGRI scenes.

Only the R rows.csv keys, each train/validation ``patch`` member, and matching
full-scene/data16 AGRI arrays are read. NPZ label members are never accessed.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path('/home/Data_Pool_3/chenyi/Auxiliary_data/HuBei_AGRI_Preprocessed')
ROWS = Path('/tmp/hubei_R_rows_trainval_attempt2.csv')
OUT = Path('/tmp/hubei_R_full_scene_source_audit_attempt1')
CHANNELS = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    rows = list(csv.DictReader(ROWS.open(newline='', encoding='utf-8-sig')))
    if len(rows) != 3479 or any(r['split'] not in {'train', 'val'} for r in rows):
        raise ValueError(f'unexpected R train/val rows: {len(rows)}')
    if len({r['time_bjt'] for r in rows}) != len(rows):
        raise ValueError('duplicate R time key')

    records = []
    misses = []
    channel_sum = np.zeros(13, np.float64)
    channel_max = np.zeros(13, np.float64)
    exact_rows = 0
    data16_exact = 0
    full_exists = 0
    for i, r in enumerate(rows):
        bjt = dt.datetime.strptime(r['time_bjt'], '%Y%m%d%H%M')
        utc = bjt - dt.timedelta(hours=8)
        rel = Path(utc.strftime('%Y/%Y%m/%Y%m%d')) / f'{utc:%Y%m%d%H%M}.npy'
        full = ROOT / 'data' / rel
        small = ROOT / 'data16' / rel
        patch_file = Path(r['path_agri'])
        if not full.is_file() or not patch_file.is_file():
            misses.append({'index': i, 'time_bjt': r['time_bjt'], 'split': r['split'],
                           'full_scene_exists': full.is_file(), 'source_patch_exists': patch_file.is_file()})
            continue
        full_exists += 1
        # Access only the AGRI patch member, never GHI/COT/label members.
        with np.load(patch_file, allow_pickle=False) as z:
            patch = np.asarray(z['patch'][CHANNELS], dtype=np.float32)
        scene = np.load(full, mmap_mode='r', allow_pickle=False)
        if scene.shape != (20, 256, 256) or patch.shape != (13, 16, 16):
            raise ValueError(f'shape mismatch at {r["time_bjt"]}: {scene.shape}, {patch.shape}')
        canonical = np.asarray(scene[CHANNELS, 103:119, 118:134], dtype=np.float32)
        delta = np.abs(patch.astype(np.float64) - canonical.astype(np.float64))
        chmax = delta.max(axis=(1, 2))
        chmean = delta.mean(axis=(1, 2))
        rowmax = float(chmax.max())
        channel_sum += delta.sum(axis=(1, 2))
        channel_max = np.maximum(channel_max, chmax)
        if rowmax <= 1e-7:
            exact_rows += 1
        small_exists = small.is_file()
        small_exact = None
        if small_exists:
            small_arr = np.load(small, mmap_mode='r', allow_pickle=False)
            if small_arr.shape == (20, 16, 16):
                small_exact = bool(np.array_equal(small_arr[CHANNELS], canonical))
                data16_exact += int(small_exact)
        records.append({
            'index': i, 'time_bjt': r['time_bjt'], 'time_utc': utc.strftime('%Y%m%d%H%M'),
            'split': r['split'], 'max_abs_by_channel': chmax.tolist(),
            'mean_abs_by_channel': chmean.tolist(), 'max_abs': rowmax,
            'patch_equals_full_scene_crop': rowmax <= 1e-7,
            'full_scene_sha256': sha256(full), 'source_patch_sha256': sha256(patch_file),
            'data16_exists': small_exists, 'data16_equals_full_scene_crop': small_exact,
        })
        if (i + 1) % 500 == 0:
            print(f'AUDITED {i+1}/{len(rows)} full={full_exists} exact={exact_rows}', flush=True)

    by_split = {s: {'rows': sum(r['split'] == s for r in rows),
                    'audited': sum(r['split'] == s for r in records),
                    'exact': sum(r['split'] == s and r['patch_equals_full_scene_crop'] for r in records),
                    'mismatched': sum(r['split'] == s and not r['patch_equals_full_scene_crop'] for r in records)}
                for s in ('train', 'val')}
    summary = {
        'state': 'R_SOURCE_VS_CANONICAL_FULL_SCENE_TRAIN_VAL_AUDIT_COMPLETE',
        'rows_csv_sha256': sha256(ROWS), 'rows': len(rows), 'rows_by_split': by_split,
        'full_scene_files_found': full_exists, 'missing_files': misses,
        'exact_rows': exact_rows, 'data16_exact_rows': data16_exact,
        'max_abs_by_channel_order': CHANNELS, 'max_abs_by_channel': channel_max.tolist(),
        'mean_abs_by_channel': (channel_sum / max(full_exists * 256, 1)).tolist(),
        'largest_mismatches': sorted(records, key=lambda x: x['max_abs'], reverse=True)[:25],
        'test_payloads_read': 0, 'buffer_payloads_read': 0, 'ghi_label_members_opened': 0,
        'note': 'Values compared only for frozen train/validation AGRI channels and station crop.'
    }
    (OUT / 'R_SOURCE_AUDIT.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (OUT / 'per_row.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['index','time_bjt','time_utc','split','max_abs','patch_equals_full_scene_crop','data16_exists','data16_equals_full_scene_crop','max_abs_C01','max_abs_C02','max_abs_C03','max_abs_C04','max_abs_C05','max_abs_C06','max_abs_C09','max_abs_C10','max_abs_C11','max_abs_C12','max_abs_C13','max_abs_C14','max_abs_C15'])
        for r in records:
            w.writerow([r['index'],r['time_bjt'],r['time_utc'],r['split'],r['max_abs'],r['patch_equals_full_scene_crop'],r['data16_exists'],r['data16_equals_full_scene_crop'],*r['max_abs_by_channel']])
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
