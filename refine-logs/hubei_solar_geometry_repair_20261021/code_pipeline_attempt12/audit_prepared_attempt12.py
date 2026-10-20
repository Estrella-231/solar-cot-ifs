#!/usr/bin/env python3
"""Independent train/validation-only audit; releases attempt12 training gate."""
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
    out = a.base / 'prepared_attempt12'
    ready = json.loads((out / 'DATA_READY.json').read_text())
    assert ready['state'] == 'CORRECTED_SOLAR_TRAIN_VAL_DATA_READY'
    assert ready['test_payloads_read'] == ready['buffer_payloads_read'] == 0
    assert ready['R_input_source_alignment_rows'] == 3479
    source_receipt=json.loads((a.base/'R_NON_S_CACHE_SOURCE_AUDIT_attempt12.json').read_text())
    transfer_receipt=json.loads((a.base/'TRANSFER_AUDIT_attempt6.json').read_text())
    assert source_receipt['state']=='PASS_R_NON_S_CACHE_SOURCES_TRAIN_VAL_SHA256' and source_receipt['count']==560
    assert transfer_receipt['state']=='PASS_TRANSFERRED_R_ONLY_TRAIN_VAL_SHA256' and transfer_receipt['count']==202 and transfer_receipt['all_hashes_match'] and transfer_receipt['source_audit_sha256']==sha(a.base/'MISSING_R_SOURCE_AUDIT_attempt6.json')
    assert source_receipt['R_only_transfer_audit_sha256']==sha(a.base/'TRANSFER_AUDIT_attempt6.json')
    assert ready['R_non_S_cache_source_audit_sha256']==sha(a.base/'R_NON_S_CACHE_SOURCE_AUDIT_attempt12.json')
    assert ready['R_only_transfer_audit_sha256']==sha(a.base/'TRANSFER_AUDIT_attempt6.json')
    ref = json.loads((a.base / 'NOAA_ALL_TRAIN_VAL_REFERENCE_AUDIT.json').read_text())
    assert ref['state'] == 'PASS_AGAINST_OFFICIAL_NOAA_JAVASCRIPT_GEOMETRIC_ANGLES'
    restore = json.loads((a.base / 'restored_attempt3' / 'RESTORATION_REPORT.json').read_text())
    assert restore['state'] == 'EXACT_UTC_HDF_RADIANCE_RESTORATION_COMPLETE'
    witness_audit=json.loads((a.base/'RESTORATION_WITNESS_ARCHIVE_AUDIT_attempt7.json').read_text())
    assert witness_audit['state']=='PASS_RESTORATION_WITNESS_ARCHIVE_MATCHES_RESTORED_REPORT'
    assert ready['restoration_witness_archive_audit_sha256']==sha(a.base/'RESTORATION_WITNESS_ARCHIVE_AUDIT_attempt7.json')
    source_witness=json.loads((a.preparation/'RESTORATION_REQUEST_WITNESS.json').read_text())
    witness_lut={r['utc']:r for r in source_witness['records']}
    assert source_witness['frames']==restore['frames']==255 and len(witness_lut)==255
    for r in restore['records']:
        assert witness_lut[r['utc']]['augmented_request_sha256']==r['request_sha256']

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
    r_only=[r for r in source_manifest if r['overlap_with_corrected_S']=='False']
    source_lut={x['utc']:x for x in source_receipt['records']}
    assert len(r_only)==len(source_lut)==560
    assert {r['time_utc'] for r in r_only}==set(source_lut)
    for r in r_only:
        src=source_lut[r['time_utc']];pth=Path(r['source_path_used'])
        assert src['bjt']==r['time_bjt'] and src['split']==r['split'] and src['relative']==(Path(r['time_utc'][:4])/r['time_utc'][:6]/r['time_utc'][:8]/f"{r['time_utc']}.npy").as_posix()
        expected=(Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/hubei_independent_trainval_stage_20261017/data')/f"{r['time_utc']}.npy") if src['source_kind']=='old_stage_flat' else (a.base/'r_only_data_attempt6'/src['relative'])
        assert pth==expected and pth.is_file() and sha(pth)==src['sha256']

    raw = np.load(out / 'R_raw_corrected_canonical_attempt12.npy', mmap_mode='r', allow_pickle=False)
    assert raw.shape == (3479, 16, 16, 16) and np.isfinite(raw).all()
    cp = np.load(a.cpp_sidecar, allow_pickle=False)
    cp_lut = {str(k): i for i, k in enumerate(cp['time_bjt'])}
    assert len(cp_lut) == 3479 and all(t in cp_lut for t in times)
    assert set(str(k) for k in cp['time_bjt']) == set(times) and len(set(str(k) for k in cp['time_bjt'])) == len(times), 'CPP/R key sets differ'
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
    # Independently verify every corrected S cache file referenced by the new indexes.
    S_geometry_max_abs = 0.0
    prep_aux_root = a.preparation / 'aux'
    for row in manifest:
        for path_key, sha_key in (('data_path', 'data_sha256'), ('aux_path', 'aux_sha256')):
            path = Path(row[path_key])
            assert path.is_file() and sha(path) == row[sha_key]
        data = np.load(row['data_path'], mmap_mode='r', allow_pickle=False)
        got = np.load(row['aux_path'], allow_pickle=False)
        ref = np.load(prep_aux_root / Path(row['aux_path']).name, allow_pickle=False)
        for key in ('cosSOZ', 'cosRAA', 'day_mask'):
            assert np.array_equal(got[key], ref[key]), (row['utc'], key, 'cache auxiliary differs from verified preparation geometry')
        errs = [float(np.abs(got['cosSOZ'] - np.cos(np.deg2rad(data[18]))).max()),
                float(np.abs(got['cosRAA'] - np.cos(np.deg2rad(data[17] - data[15]))).max()),
                float(np.abs(got['day_mask'] - data[19]).max())]
        S_geometry_max_abs = max(S_geometry_max_abs, *errs)
        assert max(errs) < 1e-6, (row['utc'], errs)
    result = {'state': 'PASS_PREPARED_ATTEMPT12_TRAIN_VAL_KEYS_LABELS_AGRI_GEOMETRY',
              'S_frames': len(manifest), 'S_train_frames': 9560, 'S_val_frames': 913,
              'R_rows': len(source_rows), 'R_train_rows': 3178, 'R_val_rows': 301,
              'R_S_overlap_rows': sum(r['overlap_with_corrected_S'] == 'True' for r in source_manifest),
              'R_S_overlap_max_abs': 0.0, 'R_geometry_max_abs': geometry_max, 'S_geometry_max_abs': S_geometry_max_abs,
              'target_max_abs': target_max, 'R_roles': role_audits,
              'train_val_disjoint': True, 'test_payloads_read': 0, 'buffer_payloads_read': 0,
              'ghi_label_members_opened': 0, 'DATA_READY_sha256': sha(out / 'DATA_READY.json'),
              'source_alignment_sha256': sha(out / 'R_input_source_alignment.csv')}
    (out / 'DATA_AUDIT_PASS_attempt12.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
