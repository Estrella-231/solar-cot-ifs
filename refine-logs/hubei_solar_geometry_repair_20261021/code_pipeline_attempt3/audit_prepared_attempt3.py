#!/usr/bin/env python3
"""Independent train/validation-only audit; releases attempt3 training gate."""
import argparse, csv, hashlib, json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from solar_geometry_noaa import compute_solar_angles

ROLES = ['full'] + [f'fold_{i}' for i in range(5)]

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--preparation', type=Path, required=True)
    p.add_argument('--rpack', type=Path, required=True)
    p.add_argument('--cpp-sidecar', type=Path, required=True)
    p.add_argument('--grid', type=Path, required=True)
    a = p.parse_args()
    out = a.base / 'prepared_attempt3'
    ready = json.loads((out / 'DATA_READY.json').read_text())
    assert ready['state'] == 'CORRECTED_SOLAR_TRAIN_VAL_DATA_READY'
    assert ready['test_payloads_read'] == ready['buffer_payloads_read'] == 0
    assert ready['R_input_source_alignment_rows'] == 3479
    ref = json.loads((a.base / 'NOAA_ALL_TRAIN_VAL_REFERENCE_AUDIT.json').read_text())
    assert ref['state'] == 'PASS_AGAINST_OFFICIAL_NOAA_JAVASCRIPT_GEOMETRIC_ANGLES'
    restore = json.loads((a.base / 'restored_attempt3' / 'RESTORATION_REPORT.json').read_text())
    assert restore['state'] == 'EXACT_UTC_HDF_RADIANCE_RESTORATION_COMPLETE'

    prep = a.preparation
    meta = prep / 'metadata'
    dates = {r['date_BJT'].replace('-', ''): r['split'] for r in read_csv(meta / 'day_assignment.csv')}
    fold_map = {r['date_BJT'].replace('-', ''): int(r['heldout_fold']) for r in read_csv(meta / 'train_date_folds.csv')}
    source_rows = read_csv(a.rpack / 'rows.csv')
    source_manifest = read_csv(out / 'R_input_source_alignment.csv')
    assert len(source_rows) == len(source_manifest) == 3479
    times = [r['time_bjt'] for r in source_rows]
    assert len(set(times)) == 3479
    assert all(r['split'] in ('train', 'val') and dates[r['time_bjt'][:8]] == r['split'] for r in source_rows)
    assert sum(r['split'] == 'train' for r in source_rows) == 3178
    assert sum(r['split'] == 'val' for r in source_rows) == 301
    assert all(float(r['R_S_max_abs']) == 0.0 for r in source_manifest if r['overlap_with_corrected_S'] == 'True')

    raw = np.load(out / 'R_raw_corrected_canonical_attempt3.npy', mmap_mode='r', allow_pickle=False)
    assert raw.shape == (3479, 16, 16, 16) and np.isfinite(raw).all()
    cp = np.load(a.cpp_sidecar, allow_pickle=False)
    cp_lut = {str(k): i for i, k in enumerate(cp['time_bjt'])}
    assert len(cp_lut) == 3479 and all(t in cp_lut for t in times)
    assert [str(k) for k in cp['time_bjt']] == times
    grid = dict(np.load(a.grid, allow_pickle=False))
    lat, lon, saa = (grid[k][103:119, 118:134] for k in ('lat', 'lon', 'saa'))
    expected_mask = np.empty((3479, 16, 16), dtype=bool)
    geometry_max = 0.0
    for i, row in enumerate(source_rows):
        utc = datetime.strptime(row['time_bjt'], '%Y%m%d%H%M') - timedelta(hours=8)
        z, az, day = compute_solar_angles(lat, lon, utc)
        az = np.where(day > .5, az, 0.)
        expected_geo = np.stack((np.cos(np.deg2rad(z)), np.cos(np.deg2rad(az - saa)), day)).astype(np.float32)
        geometry_max = max(geometry_max, float(np.abs(raw[i, 13:16] - expected_geo).max()))
        expected_mask[i] = cp['cot_valid_mask'][cp_lut[row['time_bjt']]].astype(bool) & (day > .5) & (expected_geo[0] > np.cos(np.deg2rad(80.)))
    assert geometry_max < 1e-6
    saved_targets = np.load(out / 'R_packs' / 'full' / 'target.npy', mmap_mode='r', allow_pickle=False)
    saved_masks = np.load(out / 'R_packs' / 'full' / 'mask.npy', mmap_mode='r', allow_pickle=False)[:, 0].astype(bool)
    expected_cot = np.stack([cp['cot'][cp_lut[t]] for t in times]).astype(np.float32)
    target_expected = np.where(expected_mask, expected_cot / 100.0, 0.0).astype(np.float32)
    assert np.array_equal(saved_masks, expected_mask)
    target_max = float(np.abs(saved_targets[:, 0] - target_expected).max())
    assert target_max == 0.0

    role_audits = {}
    for role in ROLES:
        pack = out / 'R_packs' / role
        status = json.loads((pack / 'pack_status.json').read_text())
        assert status['state'] == 'HUBEI_R_TRAIN_VAL_CORRECT_UTC_PACK_BUILT_TEST_CLOSED'
        assert status['test_payloads_read'] == status['buffer_payloads_read'] == 0
        rr = read_csv(pack / 'rows.csv')
        assert all(r['split'] in ('train', 'val') for r in rr)
        assert all(r['path_agri'] == source_manifest[times.index(r['time_bjt'])]['source_path_used']
                   for r in rr)
        indices = np.array([times.index(r['time_bjt']) for r in rr], dtype=np.int64)
        expected_indices = np.array([i for i, r in enumerate(source_rows)
                                     if r['split'] == 'val' or role == 'full'
                                     or fold_map[r['time_bjt'][:8]] != int(role[-1])], dtype=np.int64)
        assert np.array_equal(indices, expected_indices)
        train = np.array([r['split'] == 'train' for r in rr])
        x = np.load(pack / 'x_norm.npy', mmap_mode='r', allow_pickle=False)
        target = np.load(pack / 'target.npy', mmap_mode='r', allow_pickle=False)
        mask = np.load(pack / 'mask.npy', mmap_mode='r', allow_pickle=False)[:, 0].astype(bool)
        norm = json.loads((pack / 'norm.json').read_text())
        assert x.shape == (len(rr), 16, 16, 16) and target.shape == (len(rr), 1, 16, 16)
        assert np.array_equal(mask, expected_mask[indices])
        assert np.array_equal(target[:, 0], target_expected[indices])
        mu = np.asarray(norm['mean'], dtype=np.float32)
        sd = np.asarray(norm['std'], dtype=np.float32)
        expected_mu = raw[indices][train].astype(np.float64).mean(axis=(0, 2, 3)).astype(np.float32)
        expected_sd = np.maximum(raw[indices][train].astype(np.float64).std(axis=(0, 2, 3)), 1e-6).astype(np.float32)
        assert np.array_equal(mu, expected_mu) and np.array_equal(sd, expected_sd)
        denorm = x.astype(np.float64) * sd[None, :, None, None] + mu[None, :, None, None]
        roundtrip = float(np.max(np.abs(denorm - raw[indices].astype(np.float64))))
        assert roundtrip < 2e-5
        role_audits[role] = {'rows': len(rr), 'train': int(train.sum()), 'val': int((~train).sum()),
                             'normalization_fit': 'train only', 'roundtrip_max_abs': roundtrip,
                             'pack_sha256': sha(pack / 'SHA256.json')}

    manifest = read_csv(out / 'stage_manifest.csv')
    assert len(manifest) == 10473 and len({r['utc'] for r in manifest}) == 10473
    assert sum(r['split'] == 'train' for r in manifest) == 9560
    assert sum(r['split'] == 'val' for r in manifest) == 913
    result = {'state': 'PASS_PREPARED_ATTEMPT3_TRAIN_VAL_KEYS_LABELS_AGRI_GEOMETRY',
              'S_frames': len(manifest), 'S_train_frames': 9560, 'S_val_frames': 913,
              'R_rows': len(source_rows), 'R_train_rows': 3178, 'R_val_rows': 301,
              'R_S_overlap_rows': sum(r['overlap_with_corrected_S'] == 'True' for r in source_manifest),
              'R_S_overlap_max_abs': 0.0, 'R_geometry_max_abs': geometry_max,
              'target_max_abs': target_max, 'R_roles': role_audits,
              'train_val_disjoint': True, 'test_payloads_read': 0, 'buffer_payloads_read': 0,
              'ghi_label_members_opened': 0, 'DATA_READY_sha256': sha(out / 'DATA_READY.json'),
              'source_alignment_sha256': sha(out / 'R_input_source_alignment.csv')}
    (out / 'DATA_AUDIT_PASS_attempt3.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
