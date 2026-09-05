"""Trace source NetCDF -> station CPP arrays and numerical masks.

Bounded stratified audit only; never reads GHI or changes a source dataset.
No quality flag semantics are invented when a product lacks quality flags.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import netCDF4 as nc
import numpy as np

VARIABLES = ('COT', 'CER', 'CTH', 'CLP')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_hash(array):
    a = np.asarray(array, dtype='<f4').copy()
    a[np.isnan(a)] = np.float32(np.nan)
    return hashlib.sha256(a.tobytes()).hexdigest()


def run(manifest, cpp_root, output):
    output.mkdir(parents=True, exist_ok=False)
    groups = defaultdict(list)
    with manifest.open(newline='') as f:
        for r in csv.DictReader(f):
            groups[(r['split'], r['station_id'])].append(r)
    selected = []
    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda r: r['timestamp_utc'])
        picks = [rows[0], rows[len(rows)//2], max(rows, key=lambda r: float(r['cot_max']))]
        selected.extend({r['path_npz']: r for r in picks}.values())
    report = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'selection': 'first, middle, and maximum recorded COT per split/station of existing candidate manifest',
        'manifest_sha256': sha(manifest), 'cases': [], 'errors': [],
        'scope': 'source-to-cache and numeric-mask provenance; not independent retrieval accuracy'}
    for row in selected:
        print('AUDIT', row['station_id'], row['timestamp_utc'], flush=True)
        p = Path(row['path_npz'])
        stamp = datetime.fromisoformat(row['timestamp_utc'].replace('Z', '+00:00')).strftime('%Y%m%d%H%M%S')
        src = cpp_root/stamp[:4]/stamp[:8]/f'FY4B_AGRI_{stamp}.nc'
        try:
            with np.load(p, allow_pickle=False) as z:
                cached = z['cpp'].copy()
                cached_mask = z['cpp_valid_mask'].copy()
                present = bool(z['cpp_present'])
                bounds = z['cpp_crop_bounds'].tolist()
                agri = z['agri'].copy()
                gridlat, gridlon = z['grid_lat'].copy(), z['grid_lon'].copy()
            r0, r1, c0, c1 = bounds
            with nc.Dataset(src) as ds:
                raw = np.stack([np.ma.filled(ds[v][r0:r1, c0:c1], np.nan).astype(np.float32) for v in VARIABLES])
                lat, lon = np.asarray(ds['LAT'][r0:r1]), np.asarray(ds['LON'][c0:c1])
                all_lat, all_lon = np.asarray(ds['LAT'][:]), np.asarray(ds['LON'][:])
                rr = np.abs(all_lat[:, None]-gridlat[:, 0][None, :]).argmin(axis=0)
                cc = np.abs(all_lon[:, None]-gridlon[0, :][None, :]).argmin(axis=0)
                aligned = np.stack([np.ma.filled(ds[v][rr, cc], np.nan).astype(np.float32) for v in VARIABLES])
                aligned_lat_error = float(np.max(np.abs(gridlat-all_lat[rr, None])))
                aligned_lon_error = float(np.max(np.abs(gridlon-all_lon[None, cc])))
                meta = {'global_attributes': {a: ds.getncattr(a) for a in ds.ncattrs()},
                    'variables': {n: {'shape': list(v.shape), 'dtype': str(v.dtype),
                        'attributes': {a: v.getncattr(a) for a in v.ncattrs()}} for n, v in ds.variables.items()}}
            cot, phase = raw[0], raw[3]
            finite = np.isfinite(cot)
            day = agri[19] > 0.5
            soz = agri[18]
            case = {'npz': str(p), 'npz_sha256': sha(p), 'source': str(src),
                'source_bytes': src.stat().st_size, 'source_metadata': meta,
                'source_patch_sha256': normalized_hash(raw), 'cached_patch_sha256': normalized_hash(cached),
                'split': row['split'], 'station': row['station_id'], 'timestamp_utc': row['timestamp_utc'],
                'bounds': bounds, 'present': present,
                'values_exact_match': bool(np.array_equal(cached, raw, equal_nan=True)),
                'mask_equals_source_finite': bool(np.array_equal(cached_mask, np.isfinite(raw))),
                'lat_max_abs_error_deg': float(np.max(np.abs(gridlat-lat[:, None]))),
                'lon_max_abs_error_deg': float(np.max(np.abs(gridlon-lon[None, :]))),
                'lat_signed_offset_deg': float(np.mean(lat[:, None]-gridlat)),
                'lon_signed_offset_deg': float(np.mean(lon[None, :]-gridlon)),
                'aligned_rows': rr.tolist(), 'aligned_cols': cc.tolist(),
                'aligned_lat_max_abs_error_deg': aligned_lat_error,
                'aligned_lon_max_abs_error_deg': aligned_lon_error,
                'aligned_vs_cached_finite_pairs': int((np.isfinite(aligned[0]) & np.isfinite(cached[0])).sum()),
                'aligned_vs_cached_cot_mae': float(np.nanmean(np.abs(aligned[0]-cached[0]))),
                'cot_counts': {'finite': int(finite.sum()), 'negative': int((finite & (cot<0)).sum()),
                    'over85_to100': int((finite & (cot>85) & (cot<=100)).sum()),
                    'over100': int((finite & (cot>100)).sum()),
                    'zero': int((finite & (cot==0)).sum()),
                    'clear_phase_finite_cot': int((finite & (phase==0)).sum()),
                    'clear_phase_positive_cot': int((finite & (phase==0) & (cot>0)).sum()),
                    'legacy_training_valid': int((finite & day & (cot>=0) & (cot<=85)).sum()),
                    'range100_day_soz80': int((finite & day & (cot>=0) & (cot<=100) & (soz<80)).sum())},
                'phase_values_counts': {str(v): int(n) for v, n in zip(*np.unique(phase[np.isfinite(phase)], return_counts=True))},
                'quality_variable_names': [n for n in meta['variables'] if any(x in n.lower() for x in ('quality','flag','uncert','confidence','dqf'))]}
            report['cases'].append(case)
            if aligned_lat_error > 1e-4 or aligned_lon_error > 1e-4:
                raise ValueError('nearest CPP coordinates cannot match AGRI grid within tolerance')
            np.savez_compressed(output/f'{row["station_id"]}_{stamp}_aligned_cpp.npz',
                cpp=aligned, cpp_finite_mask=np.isfinite(aligned),
                cpp_rows=rr, cpp_cols=cc, lat=gridlat, lon=gridlon,
                source_path=str(src), timestamp_utc=row['timestamp_utc'],
                note='Audited aligned reference only; no independent retrieval QA is present')
        except Exception as e:
            report['errors'].append({'source': str(src), 'npz': str(p), 'error': repr(e)})
        (output/'cpp_quality_audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    report['summary'] = {'cases': len(report['cases']), 'errors': len(report['errors']),
        'exact_value_matches': sum(c['values_exact_match'] for c in report['cases']),
        'finite_mask_matches': sum(c['mask_equals_source_finite'] for c in report['cases']),
        'source_quality_flag_present_cases': sum(bool(c['quality_variable_names']) for c in report['cases']),
        'sample_cot_gt85_le100_pixels': sum(c['cot_counts']['over85_to100'] for c in report['cases'])}
    (output/'cpp_quality_audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--cpp-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.manifest, a.cpp_root, a.output)
