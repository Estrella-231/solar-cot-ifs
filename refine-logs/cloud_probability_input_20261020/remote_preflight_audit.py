#!/usr/bin/env python3
"""Read-only remote audit of cloud-probability geometry and four-arm keys.

Run by piping this file to ``ssh zjnu-hpc 'python -'``. It opens only the
registered train/validation GHI pack, frozen probability bank, old classifier
pack, AGRI auxiliary geometry, and grid_static; no test payload is accessed.
"""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907')
GHI = ROOT / 'experiments/cot_interval_history_20261015_v1/data_attempt2'
RBANK = ROOT / 'experiments/r_geometry_downstream_ghi_20261016/data/paired_r_cot_bank_v1'
PBANK = ROOT / 'experiments/monthly_R_station16_20261013_v1/results/conservative_gate_cot_bank_20261015_attempt1'
CLF_PACK = ROOT / 'experiments/monthly_R_station16_20261013_v1/data/pack_trainval_v1'
DATA = Path('/public/home/slfu/ttzhou/Auxiliary_data/HuNan_AGRI_Preprocessed')
GRID = DATA / 'grid_static.npz'
STATIONS = ('sili', 'zhujia')
CHANNELS = (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def solar_angles(lat, lon, dt):
    """Exact corrected nominal-UTC function used by the GHI pack contract."""
    year, month, day = dt.year, dt.month, dt.day
    hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    if month <= 2:
        year -= 1
        month += 12
    a = int(year / 100)
    b = 2 - a + int(a / 4)
    jd = (int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b
          - 1524.5 + hour / 24.0)
    jc = (jd - 2451545.0) / 36525.0
    gmls = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360.0
    gmas = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    ecc = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    gmas_rad = np.deg2rad(gmas)
    center = (np.sin(gmas_rad) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
              + np.sin(2 * gmas_rad) * (0.019993 - 0.000101 * jc)
              + np.sin(3 * gmas_rad) * 0.000289)
    true_longitude = gmls + center
    omega = 125.04 - 1934.136 * jc
    apparent_longitude = true_longitude - 0.00569 - 0.00478 * np.sin(np.deg2rad(omega))
    mean_obliq = 23.0 + (26.0 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60.0) / 60.0
    obliq = mean_obliq + 0.00256 * np.cos(np.deg2rad(omega))
    obliq_rad = np.deg2rad(obliq)
    decl = np.arcsin(np.sin(obliq_rad) * np.sin(np.deg2rad(apparent_longitude)))
    vary = np.tan(obliq_rad / 2.0) ** 2
    gmls_rad = np.deg2rad(gmls)
    eot = 4.0 * np.rad2deg(
        vary * np.sin(2 * gmls_rad) - 2 * ecc * np.sin(gmas_rad)
        + 4 * ecc * vary * np.sin(gmas_rad) * np.cos(2 * gmls_rad)
        - 0.5 * vary * vary * np.sin(4 * gmls_rad)
        - 1.25 * ecc * ecc * np.sin(2 * gmas_rad))
    utc_minutes = dt.hour * 60.0 + dt.minute + dt.second / 60.0
    tst = (utc_minutes + eot + 4.0 * lon) % 1440.0
    ha = np.deg2rad(tst / 4.0 - 180.0)
    lat_rad = np.deg2rad(lat)
    cosz = np.clip(np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(ha), -1.0, 1.0)
    soz = np.rad2deg(np.arccos(cosz))
    sinz = np.sqrt(np.maximum(1.0 - cosz ** 2, 1e-12))
    cosaz = np.clip((np.sin(decl) - np.sin(lat_rad) * cosz) / (np.cos(lat_rad) * sinz), -1.0, 1.0)
    sinaz = np.clip(-np.cos(decl) * np.sin(ha) / sinz, -1.0, 1.0)
    soa = np.rad2deg(np.arctan2(sinaz, cosaz)) % 360.0
    return soz.astype(np.float32), soa.astype(np.float32)


def aux_path(utc_text):
    dt = datetime.fromisoformat(utc_text) + timedelta(hours=8)
    matches = sorted((DATA / 'aux_256' / dt.strftime('%Y%m')).glob(dt.strftime('%Y%m%d%H%M') + '_*_aux.npz'))
    if len(matches) != 1:
        raise RuntimeError(f'aux sidecar count != 1 for {utc_text}: {matches}')
    return matches[0]


def station_rc(text):
    return tuple(int(x) for x in text.split(','))


def main():
    gm = json.loads((GHI / 'pack_manifest.json').read_text())
    rm = json.loads((RBANK / 'bank_manifest.json').read_text())
    pm = json.loads((PBANK / 'bank_manifest.json').read_text())
    assert gm['state'] == 'COMPLETE' and gm['test_payload_opened'] is False
    assert rm['state'] == 'COMPLETE_PAIRED_R_COT_HISTORY_BANK' and rm['test_payloads_read'] == 0
    assert pm['state'] == 'COMPLETE' and pm['test_payload_opened'] is False
    assert pm['probability_layout'] == '[unique_trainval_target_frame, station(sili,zhujia),16,16], float16'

    with (GHI / 'rows.csv').open(newline='') as f:
        ghi_rows = list(csv.DictReader(f))
    with (RBANK / 'frame_keys.csv').open(newline='') as f:
        rkeys = list(csv.DictReader(f))
    with (PBANK / 'frame_keys.csv').open(newline='') as f:
        pkeys = list(csv.DictReader(f))
    with (PBANK / 'classifier_assignment.csv').open(newline='') as f:
        assignments = list(csv.DictReader(f))
    assert len(ghi_rows) == 51929 and sum(r['split'] == 'train' for r in ghi_rows) == 47666
    assert len(rkeys) == 54895 and len(pkeys) == len(assignments) == 55733
    pidx = {r['target_time_utc']: int(r['index']) for r in pkeys}
    assert len(pidx) == len(pkeys)
    aidx = {int(r['frame_index']): r for r in assignments}
    assert len(aidx) == len(assignments)
    oof_dates = {r['target_date_bjt'] for r in assignments
                 if r['classifier_source'] in ('fold0', 'fold1', 'fold2')}
    rlookup = {(r['station'], r['target_time_utc']): r for r in rkeys}
    assert len(rlookup) == len(rkeys)

    prob = np.load(PBANK / 'p_cloud.npy', mmap_mode='r', allow_pickle=False)
    assert prob.shape == (55733, 2, 16, 16) and prob.dtype == np.float16
    stations_in_layout = pm.get('stations', ['sili', 'zhujia'])
    if isinstance(stations_in_layout, str):
        stations_in_layout = [x.strip() for x in stations_in_layout.split(',')]
    assert stations_in_layout == ['sili', 'zhujia'], stations_in_layout
    pmin, pmax, nfinite = 1.0, 0.0, 0
    for lo in range(0, len(prob), 1024):
        q = np.asarray(prob[lo:lo + 1024])
        nfinite += int(np.isfinite(q).sum())
        pmin, pmax = min(pmin, float(np.nanmin(q))), max(pmax, float(np.nanmax(q)))
    assert nfinite == prob.size and 0.0 <= pmin <= pmax <= 1.0

    missing, split_mismatch, station_rc_mismatch, assignment_bad = [], [], [], []
    expected_frame_keys = set()
    bad_day = 0
    for row in ghi_rows:
        if row['station'] not in STATIONS:
            raise AssertionError(row['station'])
        if row['suspect_daylight_long_zero'] == '1':
            bad_day += 1
        times = [row['earlier_time_utc'], row['previous_time_utc'], row['target_time_utc']]
        tgt = datetime.fromisoformat(row['target_time_utc'])
        assert [int((tgt - datetime.fromisoformat(t)).total_seconds() / 60) for t in times] == [30, 15, 0]
        assert row['agri_frame_relpath'] == f"{tgt:%Y/%Y%m/%Y%m%d}/{tgt:%Y%m%d%H%M}.npy"
        for t in times:
            key = (row['station'], t)
            expected_frame_keys.add(key)
            if t not in pidx:
                missing.append(key); continue
            pi = pidx[t]
            pk = pkeys[pi]
            ai = aidx[pi]
            if pk['split'] != row['split'] or ai['split'] != row['split']:
                split_mismatch.append((key, row['split'], pk['split'], ai['split']))
            expected_rc = (row['row'], row['col'])
            if station_rc(pk[f"{row['station']}_grid_rc"]) != tuple(map(int, expected_rc)):
                station_rc_mismatch.append((key, expected_rc, pk[f"{row['station']}_grid_rc"]))
            rk = rlookup.get(key)
            if rk is None or rk['split'] != row['split'] or (rk['row'], rk['col']) != expected_rc:
                station_rc_mismatch.append(('R-bank', key))
            if row['split'] == 'val' and ai['classifier_source'] != 'full_train':
                assignment_bad.append((key, ai['classifier_source']))
            if row['split'] == 'train' and ai['target_date_bjt'] in oof_dates and ai['classifier_source'] == 'full_train':
                # This is allowed only for dates with no CPP-supervised OOF fold.
                assignment_bad.append((key, ai['classifier_source'], ai['target_date_bjt']))
    if missing or split_mismatch or station_rc_mismatch or assignment_bad:
        raise AssertionError(json.dumps(dict(missing=len(missing), split_mismatch=len(split_mismatch),
            station_rc_mismatch=len(station_rc_mismatch), assignment_bad=len(assignment_bad))))

    # Check that the classifier's archived training tensors carried the old
    # producer geometry, then compare it with the corrected nominal-UTC field
    # on a deterministic time-spread sample of the exact GHI history keys.
    clf_rows_path = CLF_PACK / 'rows.csv'
    clf_x_path = CLF_PACK / 'x_raw.npy'
    clf_rows_all = list(csv.DictReader(clf_rows_path.open(newline='')))
    clf_rows = sorted((r for r in clf_rows_all if r['state'] == 'SUPERVISED'), key=lambda r: int(r['array_index']))
    clf_x = np.load(clf_x_path, mmap_mode='r', allow_pickle=False)
    assert clf_x.shape == (len(clf_rows), 16, 16, 16)
    clf_lookup = {(r['station'], r['target_time_utc']): int(r['array_index']) for r in clf_rows}
    assert len(clf_lookup) == len(clf_rows)
    with np.load(GRID, allow_pickle=False) as z:
        lat = np.asarray(z['lat'], np.float32); lon = np.asarray(z['lon'], np.float32); saa = np.asarray(z['saa'], np.float32)
    assert lat.shape == lon.shape == saa.shape == (256, 256)
    coords = {'sili': (163, 148), 'zhujia': (73, 150)}
    unique_times = sorted({t for _, t in expected_frame_keys})
    sample_times = unique_times[::max(1, len(unique_times) // 96)][:96]
    geom_rows = []
    old_cosz, new_cosz, old_cosraa, new_cosraa = [], [], [], []
    day_mismatches = 0; exact_classifier_aux = []
    for t in sample_times:
        dt = datetime.fromisoformat(t); bjt = dt + timedelta(hours=8)
        apath = aux_path(t)
        with np.load(apath, allow_pickle=False) as z:
            auxz = np.asarray(z['cosSOZ'], np.float32); auxr = np.asarray(z['cosRAA'], np.float32); auxday = np.asarray(z['day_mask'], np.float32)
        for st in STATIONS:
            rr, cc = coords[st]; sl = np.s_[rr - 8:rr + 8, cc - 8:cc + 8]
            if sl[0].start < 0 or sl[1].start < 0 or sl[0].stop > 256 or sl[1].stop > 256:
                raise AssertionError((st, rr, cc))
            old = np.stack((auxz[sl], auxr[sl], auxday[sl]))
            soz, soa = solar_angles(lat[sl], lon[sl], dt)
            soa = np.where(soz >= 90.0, 0.0, soa)
            new = np.stack((np.cos(np.deg2rad(soz)), np.cos(np.deg2rad(soa - saa[sl])), (soz < 85.0).astype(np.float32)))
            old_cosz.extend(old[0].ravel().tolist()); new_cosz.extend(new[0].ravel().tolist())
            old_cosraa.extend(old[1].ravel().tolist()); new_cosraa.extend(new[1].ravel().tolist())
            day_mismatches += int(np.count_nonzero(old[2] != new[2]))
            ix = clf_lookup.get((st, t))
            if ix is not None:
                d = np.abs(clf_x[ix, 13:16] - old)
                exact_classifier_aux.append(float(d.max()))
            geom_rows.append(dict(station=st, target_time_utc=t, old_aux= str(apath),
                cosSOZ_mean_abs_delta=float(np.mean(np.abs(old[0] - new[0]))),
                cosSOZ_max_abs_delta=float(np.max(np.abs(old[0] - new[0]))),
                cosRAA_mean_abs_delta=float(np.mean(np.abs(old[1] - new[1]))),
                cosRAA_max_abs_delta=float(np.max(np.abs(old[1] - new[1]))),
                day_mask_mismatch_pixels=int(np.count_nonzero(old[2] != new[2]))))
    def delta_summary(old, new):
        d = np.abs(np.asarray(old, np.float64) - np.asarray(new, np.float64))
        return dict(mean_abs=float(d.mean()), p95_abs=float(np.percentile(d, 95)), max_abs=float(d.max()))

    out = dict(state='PASS_EXACT_KEY_ALIGNMENT_GEOMETRY_PROVENANCE_MISMATCH_RECORDED',
        checked_at_utc=datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        operation='read-only train/validation arrays and metadata; no classifier weights loaded; no test payload accessed',
        train_val_only=True, test_payloads_read=0,
        source_paths=dict(ghi_pack=str(GHI), r_bank=str(RBANK), probability_bank=str(PBANK), classifier_pack=str(CLF_PACK), data_root=str(DATA)),
        source_states=dict(ghi=gm['state'], r=rm['state'], probability=pm['state'],
            split_id=gm['split_id'], split_consistent=(gm['split_id'] == rm['split_id'] == pm['split_id'])),
        alignment=dict(ghi_rows=len(ghi_rows), expected_unique_station_frame_keys=len(expected_frame_keys),
            probability_rows=len(pkeys), r_bank_rows=len(rkeys), probability_keys_missing=len(missing),
            split_mismatches=len(split_mismatch), station_grid_mismatches=len(station_rc_mismatch),
            classifier_assignment_errors=len(assignment_bad), rows_with_suspect_zero_flag=bad_day,
            history_offsets_min=[-30, -15, 0], p_crop='same station-centered 16x16 as classifier; read center [7:10,7:10] to match GHI/R 3x3'),
        probability=dict(shape=list(prob.shape), dtype=str(prob.dtype), finite_count=nfinite, value_range=[pmin, pmax],
            sha256=sha(PBANK / 'p_cloud.npy'), stored_sha256=pm['array_sha256']['p_cloud.npy'],
            classifier_source_sha256=pm['classifier_source_sha256'], checkpoint_hashes=pm['classifier_checkpoint_sha256'],
            oof_assignment_sha256=pm['classifier_assignment_sha256'], station_order=stations_in_layout),
        geometry=dict(classifier_training_input='cached old producer aux cosSOZ/cosRAA/day_mask; code and manifest show same R-normalized 16x16 source',
            ghi_input='corrected nominal-UTC geometry; equation-of-time trigonometric expression converted radians to degrees',
            classifier_training_rows=len(clf_rows), classifier_geometry_source_exact_aux_max_abs=(max(exact_classifier_aux) if exact_classifier_aux else None),
            sampled_unique_utc_times=len(sample_times), sampled_station_frames=len(geom_rows),
            old_vs_corrected_cosSOZ=delta_summary(old_cosz, new_cosz), old_vs_corrected_cosRAA=delta_summary(old_cosraa, new_cosraa),
            old_vs_corrected_day_mask_mismatch_pixels=day_mismatches,
            caveat='The probability bank is spatially/time keyed and split-aligned, but its frozen classifier used the old EOT geometry. Keep it as the registered legacy p_cloud feature; do not describe it as corrected-geometry classification. Feeding corrected geometry to these frozen weights would be an unvalidated distribution shift.'),
        hashes=dict(ghi_rows=sha(GHI / 'rows.csv'), ghi_manifest=sha(GHI / 'pack_manifest.json'),
            r_manifest=sha(RBANK / 'bank_manifest.json'), probability_manifest=sha(PBANK / 'bank_manifest.json'),
            probability_keys=sha(PBANK / 'frame_keys.csv'), assignment=sha(PBANK / 'classifier_assignment.csv'),
            classifier_rows=sha(clf_rows_path), classifier_x_raw=sha(clf_x_path), grid_static=sha(GRID)),
        geometry_sample=geom_rows)
    assert out['source_states']['split_consistent']
    assert out['alignment']['probability_keys_missing'] == out['alignment']['split_mismatches'] == 0
    assert out['alignment']['station_grid_mismatches'] == out['alignment']['classifier_assignment_errors'] == 0
    assert max(exact_classifier_aux, default=0.0) < 2e-5
    print(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
