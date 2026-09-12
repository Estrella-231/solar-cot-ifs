"""Independent bounded NC -> sidecar -> packed COT pixel audit.

Does not import production mapping/mask/packing functions or model code.
Two modes: export-pack on HPC; check-source on FD-107 using the SHA-bound
small export. Only the frozen twelve train/validation samples are decoded.
"""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np

SELECTION_FILE_SHA = 'd2557c7e37d6d55d60a295fb3c7c16a207783cd3928b6e9a8a37f0fb1054a49c'
SELECTION_SHA = 'ae0cbeab4116ba3452f4f0e329b4ca545844c04fa8ad75457e0b6a9c4b50cb2b'
PACK_HASHES = {
    'SHA256.json': 'f0216e4a5b14ca7dc2ef54da4816273e0c935da486c951889d838749e4a3c898',
    'rows.csv': '7741fda8e5a232c23598b9824811bf67ca5ba48d50b392f1c99a21b298177c69',
    'target.npy': 'e7e330fd7245212ff09b58f0cbb841a60020e583f4e4121e90b76958d595ba33',
    'mask.npy': 'c134bc31d54cbbab89523a9825bf32cefc226d89b75d85bf6c588295b63547c4',
}
ABS_DEG = 1e-4
CPP_ROOT = Path('/home/Data_Pool_3/data/Global_CPP/FY/FY4B/Result/CPP')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def stamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(result.utcoffset() == timedelta(0), 'expected aware UTC timestamp')
    return result


def load_selection(path):
    require(sha(path) == SELECTION_FILE_SHA, 'frozen selection file changed')
    selected = json.loads(Path(path).read_text())
    digest = hashlib.sha256(json.dumps(selected['samples'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(digest == selected['selection_sha256'] == SELECTION_SHA, 'selection payload changed')
    require(len(selected['samples']) == 12 and selected['test_used'] is False, 'twelve train/val samples required')
    require(all(s['split'] in ('train', 'validation') for s in selected['samples']), 'test prohibited')
    return selected


def write_json(path, report):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
    temporary.replace(path)


def save_values(path, arrays):
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size,
                shapes={k: list(v.shape) for k, v in arrays.items()})


def export_pack(args, selected, report, value_path):
    pack = Path(selected['source_rows_csv']).parent
    require(pack == args.pack.resolve(), 'canonical R pack path differs')
    for name, digest in PACK_HASHES.items():
        require(sha(pack / name) == digest, f'pack SHA mismatch: {name}')
    receipts = json.loads((pack / 'SHA256.json').read_text())
    require(all(receipts[n] == PACK_HASHES[n] for n in ('rows.csv', 'target.npy', 'mask.npy')), 'pack recorded hashes')
    with (pack / 'rows.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 18514, 'R pack length')
    for sample in selected['samples']:
        row = rows[int(sample['index'])]
        require(all(row[k] == sample[k] for k in row), 'selected row binding')
        group = sorted((r for r in rows if r['station_id'] == sample['station_id'] and r['split'] == sample['split']),
                       key=lambda r: stamp(r['timestamp_utc']))
        ranks = {'first': 0, 'middle': (len(group) - 1) // 2, 'last': len(group) - 1}
        require(len(group) == sample['group_size'] and ranks[sample['selection_slot']] == sample['selection_rank'], 'fixed rank')
        require(group[sample['selection_rank']]['index'] == sample['index'], 'rank-selected identity')
    ids = np.asarray([int(s['index']) for s in selected['samples']], dtype=np.int64)
    arrays = {'r_pack_index': ids}
    for name, shape, dtype in (('target', (18514, 1, 16, 16), np.float32), ('mask', (18514, 1, 16, 16), np.bool_)):
        source = np.load(pack / (name + '.npy'), mmap_mode='r', allow_pickle=False)
        require(source.shape == shape and source.dtype == dtype, f'pack {name} shape/dtype')
        arrays[name] = np.array(source[ids], copy=True)
        del source
    report['pack'] = dict(path=str(pack), verified_file_sha256=PACK_HASHES, selected_rows=12,
                          semantic_payload_bytes=sum(v.nbytes for v in arrays.values()))
    report['values'] = save_values(value_path, arrays)
    report['samples'] = selected['samples']
    report['state'] = 'PASS_12_SELECTED_PACK_TARGET_MASK_EXPORT'


def coordinate_error(lat, lon, target_lat, target_lon):
    return dict(latitude_max_abs_deg=float(np.max(np.abs(lat - target_lat))),
                longitude_max_abs_deg=float(np.max(np.abs(lon - target_lon))))


def independent_indices(lat, lon, target_lat, target_lon):
    require(lat.ndim == lon.ndim == 1 and target_lat.shape == target_lon.shape == (16, 16), 'coordinate dimensions')
    require(all(np.isfinite(v).all() for v in (lat, lon, target_lat, target_lon)), 'nonfinite coordinates')
    require(all(np.all(np.diff(v) > 0) or np.all(np.diff(v) < 0) for v in (lat, lon)), 'nonmonotonic source axes')
    # Every target pixel independently chooses its nearest axis coordinate.
    # Sorted-axis bracketing avoids a large axis-by-patch temporary and does
    # not trust the production first-column/first-row mapping shortcut.
    def nearest(axis, wanted):
        order = np.argsort(axis)
        ascending = axis[order]
        upper = np.searchsorted(ascending, wanted)
        lo, hi = np.clip(upper - 1, 0, len(axis) - 1), np.clip(upper, 0, len(axis) - 1)
        low_error, high_error = np.abs(ascending[lo] - wanted), np.abs(ascending[hi] - wanted)
        use_high = (high_error < low_error) | ((high_error == low_error) & (order[hi] < order[lo]))
        return np.where(use_high, order[hi], order[lo])
    rows, cols = nearest(lat, target_lat), nearest(lon, target_lon)
    require(np.array_equal(rows, np.repeat(rows[:, :1], 16, axis=1)), 'target LAT is not row-rectilinear')
    require(np.array_equal(cols, np.repeat(cols[:1], 16, axis=0)), 'target LON is not column-rectilinear')
    require(len(np.unique(rows[:, 0])) == 16 and len(np.unique(cols[0])) == 16, 'collapsed target pixels')
    error = coordinate_error(lat[rows], lon[cols], target_lat, target_lon)
    require(max(error.values()) <= ABS_DEG, 'coordinate-derived grid is not coincident')
    alternatives = {}
    transforms = {
        'transpose_target_patch': (rows.T, cols.T),
        'vertical_flip': (rows[::-1], cols[::-1]),
        'horizontal_flip': (rows[:, ::-1], cols[:, ::-1]),
        'both_flips': (rows[::-1, ::-1], cols[::-1, ::-1]),
        'row_minus_1': (rows - 1, cols), 'row_plus_1': (rows + 1, cols),
        'col_minus_1': (rows, cols - 1), 'col_plus_1': (rows, cols + 1),
        'one_based_indices_as_zero_based': (rows + 1, cols + 1),
        'zero_based_indices_as_one_based': (rows - 1, cols - 1),
    }
    for name, (r, c) in transforms.items():
        in_bounds = bool((r >= 0).all() and (r < len(lat)).all() and (c >= 0).all() and (c < len(lon)).all())
        value = dict(in_bounds=in_bounds)
        if in_bounds:
            value.update(coordinate_error(lat[r], lon[c], target_lat, target_lon))
            value['within_coordinate_tolerance'] = max(value['latitude_max_abs_deg'], value['longitude_max_abs_deg']) <= ABS_DEG
        alternatives[name] = value
    return rows, cols, error, alternatives


def check_one(sample, position, packed, arrays, nc_hash_cache):
    import netCDF4
    record = dict(position=position, r_pack_index=int(sample['index']), station=sample['station_id'],
                  split=sample['split'], timestamp_utc=sample['timestamp_utc'], selection_slot=sample['selection_slot'])
    paths = [Path(sample['original_path']), Path(sample['sidecar_path'])]
    for path, key in zip(paths, ('original_sha256', 'sidecar_sha256')):
        require(sha(path) == sample[key], f'{key} mismatch')
    with np.load(paths[0], allow_pickle=False) as original, np.load(paths[1], allow_pickle=False) as sidecar:
        for source in (original, sidecar):
            require(str(source['station_id'].item()) == sample['station_id'], 'station identity')
            require(stamp(str(source['timestamp_utc'].item())) == stamp(sample['timestamp_utc']), 'UTC identity')
        require(str(sidecar['original_sha256'].item()) == sample['original_sha256'], 'sidecar original binding')
        require(str(sidecar['schema_version'].item()) == 'cpp_aligned_sidecar_v1', 'sidecar schema')
        require(list(sidecar['channel_names']) == ['COT', 'CER', 'CTH', 'CLP'], 'sidecar COT channel')
        target_lat, target_lon = original['grid_lat'].copy(), original['grid_lon'].copy()
        require(np.array_equal(target_lat, sidecar['grid_lat']) and np.array_equal(target_lon, sidecar['grid_lon']), 'original/sidecar grid')
        agri = original['agri']  # Only day/SOZ are consumed; no GHI members opened.
        require(agri.shape == (20, 16, 16), 'original AGRI shape')
        day, soz = agri[19].copy(), agri[18].copy()
        cot = sidecar['cpp'][0].copy()
        finite = sidecar['cpp_valid_mask'][0].copy()
        mask = sidecar['cot_reference_mask'].copy()
        saved_rows, saved_cols = sidecar['cpp_rows'].copy(), sidecar['cpp_cols'].copy()
        source_path = CPP_ROOT / stamp(sample['timestamp_utc']).strftime('%Y/%Y%m%d/FY4B_AGRI_%Y%m%d%H%M%S.nc')
        require(str(source_path) == str(sidecar['source_nc'].item()), 'UTC-to-NC filename binding')
        recorded_present = bool(sidecar['cpp_present'])
    record['source_nc'] = str(source_path)
    record['original_sha256'], record['sidecar_sha256'] = sample['original_sha256'], sample['sidecar_sha256']
    prefix = f's{position:02d}_'
    arrays.update({prefix + k: v for k, v in dict(target_lat=target_lat, target_lon=target_lon,
        sidecar_COT=cot, sidecar_finite=finite, sidecar_mask=mask, sidecar_rows=saved_rows,
        sidecar_cols=saved_cols, day=day, solar_zenith_deg=soz, pack_target=packed['target'][position, 0],
        pack_mask=packed['mask'][position, 0]).items()})
    if not source_path.is_file():
        record.update(state='MISSING_SOURCE_RETAINED_NO_REPLACEMENT', source_recorded_present=recorded_present)
        return record
    require(recorded_present, 'source present now but selected sidecar marked missing')
    before = source_path.stat()
    cache_key = (str(source_path), before.st_size, before.st_mtime_ns)
    if cache_key not in nc_hash_cache:
        nc_hash_cache[cache_key] = sha(source_path)
    record['source_nc_sha256'] = nc_hash_cache[cache_key]
    with netCDF4.Dataset(source_path) as ds:
        lat_var, lon_var, cot_var = ds.variables['LAT'], ds.variables['LON'], ds.variables['COT']
        lat = np.asarray(np.ma.filled(lat_var[:], np.nan), dtype=np.float64)
        lon = np.asarray(np.ma.filled(lon_var[:], np.nan), dtype=np.float64)
        require(len(lat_var.dimensions) == len(lon_var.dimensions) == 1, 'LAT/LON variable dimensions')
        lat_dim, lon_dim = lat_var.dimensions[0], lon_var.dimensions[0]
        require(lat_dim != lon_dim and len(cot_var.dimensions) == 2
                and set(cot_var.dimensions) == {lat_dim, lon_dim}, 'COT dimensions do not identify latitude/longitude')
        require(str(cot_var.getncattr('Range')).strip() == '0 - 100', 'declared COT range changed')
        rows, cols, error, alternatives = independent_indices(lat, lon, target_lat, target_lon)
        record['netcdf'] = dict(COT_dimensions=list(cot_var.dimensions), COT_shape=list(cot_var.shape),
            COT_dtype=str(cot_var.dtype), LAT_dimension=lat_dim, LON_dimension=lon_dim,
            LAT_length=len(lat), LON_length=len(lon), LAT_direction='increasing' if lat[-1] > lat[0] else 'decreasing',
            LON_direction='increasing' if lon[-1] > lon[0] else 'decreasing',
            attributes={k: str(cot_var.getncattr(k)) for k in cot_var.ncattrs()},
            production_lat_then_lon_dimension_order=tuple(cot_var.dimensions) == (lat_dim, lon_dim))
        # One bounded rectangle avoids re-decompressing a large source chunk
        # for every scalar. Basic slices plus NumPy paired indices have an
        # explicit meaning independent of production netCDF fancy indexing.
        r0, r1, c0, c1 = int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1
        require((r1 - r0) * (c1 - c0) <= 4096, 'coordinate-bound source rectangle unexpectedly large')
        if tuple(cot_var.dimensions) == (lat_dim, lon_dim):
            rectangle = np.ma.filled(cot_var[r0:r1, c0:c1], np.nan)
        else:
            rectangle = np.ma.filled(cot_var[c0:c1, r0:r1], np.nan).T
        source_cot = np.asarray(rectangle[rows - r0, cols - c0], dtype=np.float32)
        record['source_read_rectangle'] = dict(zero_based_bounds=[r0, r1, c0, c1],
            returned_pixels=int(rectangle.size), source_chunks=cot_var.chunking(),
            strategy='single basic-slice hyperslab; paired indexing in NumPy; no alternate-coordinate value search')
    after = source_path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'NC changed during hash/read')
    require(cot.shape == finite.shape == mask.shape == (16, 16), 'sidecar shape')
    require(cot.dtype == np.float32 and finite.dtype == mask.dtype == np.bool_, 'sidecar dtype')
    expected_finite = np.isfinite(source_cot)
    expected_mask = (expected_finite & (source_cot >= 0) & (source_cot <= 100)
                     & np.isfinite(day) & (day > 0.5) & np.isfinite(soz)
                     & (soz >= 0) & (soz <= 180) & (soz < 80))
    expected_target = np.where(expected_mask, source_cot / np.float32(100.0), np.float32(0)).astype(np.float32)
    checks = dict(original_sidecar_grid_exact=True,
        cpp_rows_zero_based_exact=np.array_equal(saved_rows, rows[:, 0]),
        cpp_cols_zero_based_exact=np.array_equal(saved_cols, cols[0]),
        source_to_sidecar_COT_exact=np.array_equal(source_cot, cot, equal_nan=True),
        source_to_sidecar_finite_exact=np.array_equal(expected_finite, finite),
        independently_rebuilt_sidecar_mask_exact=np.array_equal(expected_mask, mask),
        independently_rebuilt_pack_mask_exact=np.array_equal(expected_mask, packed['mask'][position, 0]),
        independently_rebuilt_pack_target_exact=np.array_equal(expected_target, packed['target'][position, 0]),
        recorded_valid_pixels_exact=int(expected_mask.sum()) == int(sample['valid_pixels']),
        COT_dimension_order_matches_production=record['netcdf']['production_lat_then_lon_dimension_order'])
    record['checks'] = {k: bool(v) for k, v in checks.items()}
    record['coordinate_error'] = error
    record['coordinate_only_alternative_hypotheses'] = alternatives
    record['derived_zero_based_rows'], record['derived_zero_based_cols'] = rows[:, 0].tolist(), cols[0].tolist()
    record['valid_pixels'] = int(expected_mask.sum())
    round_trip = (expected_target * np.float32(100.0)).astype(np.float64)
    record['float32_div100_mul100_rounding_max_abs_valid_COT'] = float(np.max(np.abs(
        round_trip[expected_mask] - source_cot.astype(np.float64)[expected_mask]))) if expected_mask.any() else None
    finite_both = np.isfinite(source_cot) & np.isfinite(cot)
    record['source_sidecar_max_abs_finite_COT'] = float(np.max(np.abs(source_cot[finite_both].astype(np.float64)
        - cot[finite_both].astype(np.float64)))) if finite_both.any() else None
    arrays.update({prefix + k: v for k, v in dict(source_LAT_axis=lat, source_LON_axis=lon,
        derived_rows=rows, derived_cols=cols, source_COT=source_cot,
        expected_mask=expected_mask, expected_target=expected_target).items()})
    record['state'] = 'PASS_SOURCE_SIDECAR_PACK_PIXEL_CHAIN' if all(checks.values()) else 'FAIL_PIXEL_CHAIN'
    return record


def check_source(args, selected, report, value_path):
    require(sha(args.pack_report) == args.pack_report_sha256, 'pack export report SHA')
    receipt = json.loads(args.pack_report.read_text())
    require(receipt['state'] == 'PASS_12_SELECTED_PACK_TARGET_MASK_EXPORT'
            and receipt['selection_sha256'] == SELECTION_SHA, 'pack export not accepted')
    require(receipt['samples'] == selected['samples'], 'export sample identity')
    require(sha(args.pack_values) == receipt['values']['sha256'], 'pack export values SHA')
    with np.load(args.pack_values, allow_pickle=False) as source:
        packed = {name: source[name] for name in source.files}
    require(packed['target'].shape == packed['mask'].shape == (12, 1, 16, 16), 'export shape')
    require(packed['target'].dtype == np.float32 and packed['mask'].dtype == np.bool_, 'export dtype')
    require(np.array_equal(packed['r_pack_index'], [int(s['index']) for s in selected['samples']]), 'export row indices')
    report['pack_export'] = dict(report_sha256=args.pack_report_sha256, values_sha256=receipt['values']['sha256'],
                                 source_pack=receipt['pack'])
    arrays, nc_hash_cache = {}, {}
    for position, sample in enumerate(selected['samples']):
        try:
            row = check_one(sample, position, packed, arrays, nc_hash_cache)
        except Exception as error:
            row = dict(position=position, r_pack_index=int(sample['index']), station=sample['station_id'],
                split=sample['split'], timestamp_utc=sample['timestamp_utc'], state='ERROR_SAMPLE_RETAINED',
                error_type=type(error).__name__, error=str(error))
        report['samples'].append(row)
        print('SPATIAL_PIXEL_CHAIN', position, row['state'], flush=True)
    report['values'] = save_values(value_path, arrays)
    report['source_NC_files_hashed'] = len(nc_hash_cache)
    report['pass_count'] = sum(r['state'] == 'PASS_SOURCE_SIDECAR_PACK_PIXEL_CHAIN' for r in report['samples'])
    report['missing_source_count'] = sum(r['state'] == 'MISSING_SOURCE_RETAINED_NO_REPLACEMENT' for r in report['samples'])
    report['state'] = 'PASS_12_SOURCE_SIDECAR_PACK_PIXEL_CHAINS' if report['pass_count'] == 12 else 'COMPLETE_WITH_UNRESOLVED_PIXEL_CHAINS'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('export-pack', 'check-source'), required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--prefix', type=Path, required=True, help='NEW report/value file prefix; adds .json and .npz')
    parser.add_argument('--pack', type=Path)
    parser.add_argument('--pack-report', type=Path)
    parser.add_argument('--pack-report-sha256')
    parser.add_argument('--pack-values', type=Path)
    args = parser.parse_args()
    report_path, value_path = args.prefix.with_suffix('.json'), args.prefix.with_suffix('.npz')
    require(not report_path.exists() and not value_path.exists(), 'refuse to overwrite previous audit evidence')
    args.prefix.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    report = dict(state='RUNNING', mode=args.mode, script_sha256=sha(__file__),
        selection_file_sha256=SELECTION_FILE_SHA, selection_sha256=SELECTION_SHA,
        started_utc=datetime.now(timezone.utc).isoformat(), absolute_coordinate_tolerance_deg=ABS_DEG,
        test_used=False, GHI_decoded=False, models_loaded=0, GPU_used=False, old_assets_modified=False,
        mapping='independent nearest source LAT/LON for every target pixel; no interpolation, no error-based coordinate choice',
        mask='source finite, COT [0,100], finite day>0.5, finite SOZ in [0,180] and SOZ<80; original contract unchanged',
        scope='twelve preselected train/validation rows; does not certify the complete dataset or upstream retrieval teacher',
        runtime=dict(python=platform.python_version(), numpy=np.__version__), samples=[])
    try:
        selected = load_selection(args.selection)
        if args.mode == 'export-pack':
            require(args.pack is not None, '--pack required')
            export_pack(args, selected, report, value_path)
        else:
            require(args.pack_report is not None and args.pack_values is not None
                    and args.pack_report_sha256 is not None, 'pack export arguments required')
            check_source(args, selected, report, value_path)
    except Exception as error:
        report.update(state='ERROR_INCOMPLETE_AUDIT', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        report['elapsed_seconds'] = time.monotonic() - start
        write_json(report_path, report)
        print(report['state'], flush=True)


if __name__ == '__main__':
    main()
