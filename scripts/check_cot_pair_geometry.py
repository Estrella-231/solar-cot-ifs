"""Bounded CPU geometry gate for a frozen real/forecast COT pair inventory.

No model imports, inference, label/mask reads, or test inputs. Full selected
files are streamed only for their SHA receipts. Semantic data reads are the
64 selected real [13:16] patches and 872 selected forecast geometry patches.
The abs-only tolerance is fixed here before any geometry values are read.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                  'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_variable] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np

SELECTION_SHA = '8a06a7f06cf3339cd8f067111d5397c173ba63868598c6904348362994b08318'
ATOL = 1e-5
CHANNELS = ('cosSOZ', 'cosRAA', 'day_mask')
QUANTILES = (0, .5, .9, .95, .99, 1)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dt(text):
    # Python 3.10 needs explicit normalization of ISO-8601 Z. The original
    # preflight compared raw strings and rejected equal Z/+00:00 instants.
    value = datetime.fromisoformat(text.replace('Z', '+00:00'))
    require(value.utcoffset() == timedelta(0), 'target/init must be UTC')
    return value


def channel_stats(real, forecast):
    """Both arrays [3,N], all samples included; no target/CPP/day filtering."""
    require(real.shape == forecast.shape and real.shape[0] == 3, 'stats shape')
    result = {}
    for c, name in enumerate(CHANNELS):
        r, f = real[c].astype(np.float64), forecast[c].astype(np.float64)
        valid = np.isfinite(r) & np.isfinite(f)
        absolute = np.abs(f[valid] - r[valid])
        signed = f[valid] - r[valid]
        result[name] = dict(elements=int(r.size), finite_pairs=int(valid.sum()),
            real_nonfinite=int((~np.isfinite(r)).sum()),
            forecast_nonfinite=int((~np.isfinite(f)).sum()),
            max_abs=float(absolute.max()) if absolute.size else None,
            mae=float(absolute.mean()) if absolute.size else None,
            rmse=float(np.sqrt(np.mean(signed ** 2))) if absolute.size else None,
            signed_mean_forecast_minus_real=float(signed.mean()) if absolute.size else None,
            abs_quantiles={str(q): float(v) for q, v in zip(QUANTILES, np.quantile(absolute, QUANTILES))} if absolute.size else {},
            within_abs_tolerance=int(np.sum(absolute <= ATOL)),
            outside_abs_tolerance=int(np.sum(absolute > ATOL)),
            exact_equal=int(np.sum(r[valid] == f[valid])))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--inventory', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--values', required=True, type=Path)
    parser.add_argument('--previous-failure', required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists() and not args.values.exists(), 'outputs must be new')
    start = time.monotonic()
    report = dict(state='RUNNING', created_utc=datetime.now(timezone.utc).isoformat(),
        selection_sha256=SELECTION_SHA, absolute_tolerance=ATOL, relative_tolerance=0,
        quantiles=list(QUANTILES), threshold_chosen_before_geometry_read=True,
        script_sha256=sha(Path(__file__)), inventory_sha256=sha(args.inventory),
        no_geometry_override=True, models_loaded=0, training_or_GPU_used=False,
        test_rows_read=0, test_payloads_read=0, GHI_labels_or_performance_read=False,
        CPP_COT_target_or_mask_read=False, predicted_AGRI_semantically_read=False,
        source_coordinates_read=False, coordinate_identity_proven=False,
        metadata_verified=[], full_file_SHA_verified=[], semantic_reads=[])
    try:
        prior_sha = sha(args.previous_failure)
        require(prior_sha == '078a1f1eef2c1c8864df5c44d14cb3b246a4118f625225c9a415b4c0586737f1', 'prior failure receipt mismatch')
        report['prior_failure'] = dict(path=str(args.previous_failure.resolve()), sha256=prior_sha,
            original_state='FAIL_CONTRACT', original_failure='selected R identity',
            cause='Strict string comparison rejected the same explicit UTC instant encoded as 2025-06-30T22:30:00Z and 2025-06-30T22:30:00+00:00.',
            correction='Compare parsed aware UTC instants, normalize Z for Python 3.10; reject naive or non-UTC timestamps. Selection and values unchanged; prior failure preserved.')
        require(hasattr(os, 'sched_getaffinity'), 'Linux affinity support required')
        cpus = sorted(os.sched_getaffinity(0))[:2]
        os.sched_setaffinity(0, set(cpus))
        report['cpu_affinity'] = sorted(os.sched_getaffinity(0))
        require(1 <= len(report['cpu_affinity']) <= 2, 'CPU affinity cap')
        report['thread_limit_environment'] = {v: os.environ[v] for v in
            ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}
        root = args.root.resolve()
        inventory = json.loads(args.inventory.read_text())
        require(inventory['root'] == str(root), 'inventory root mismatch')
        selection = inventory['selection']
        canonical = json.dumps(dict(rule=inventory['selection_rule'], targets=selection),
                               sort_keys=True, separators=(',', ':')).encode()
        require(hashlib.sha256(canonical).hexdigest() == inventory['selection_sha256'] == SELECTION_SHA,
                'frozen selection SHA mismatch')
        require(len(selection) == 64, 'must preserve 64 targets')
        require(sum(len(t['pairs']) for t in selection) == 872, 'must preserve 872 pairs')
        require(sum(len(t['missing_leads']) for t in selection) == 152, 'must preserve 152 unavailable leads')
        require(sum(not t['pairs'] for t in selection) == 6, 'must preserve six unmatched targets')
        require(len(inventory['selected_shards']) == 59, 'must preserve 59 selected shards')
        require(inventory['selection_rule']['split'] == 'validation', 'validation only')
        require(inventory['forecast']['station_order'] == ['sili', 'zhujia'], 'station order')
        require(inventory['R_pack']['input_features'][13:] == list(CHANNELS), 'R geometry channel order')

        def allowed(path):
            path = Path(path).resolve()
            require(path.is_relative_to(root), f'path outside root: {path}')
            require('test' not in path.parts and not path.name.startswith('test') and path.name != 'labels.csv',
                    f'forbidden data path: {path}')
            return path

        # Verify exactly the prior metadata receipts; headers and tensor entries
        # are excluded here. In particular target.npy/mask.npy are never opened.
        metadata_bytes = {}
        for receipt in inventory['metadata_accessed']:
            if receipt['mode'] != 'metadata':
                continue
            path = allowed(receipt['path'])
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            require(len(raw) == receipt['bytes'] and digest == receipt['sha256'], f'metadata receipt mismatch: {path}')
            report['metadata_verified'].append(dict(path=str(path), bytes=len(raw), sha256=digest))
            metadata_bytes[str(path)] = raw
        pack = allowed(inventory['R_pack']['path'])
        cache = allowed(inventory['forecast']['path'])
        rows = list(csv.DictReader(io.StringIO(metadata_bytes[str(pack/'rows.csv')].decode('utf-8-sig'))))
        indices = list(csv.DictReader(io.StringIO(metadata_bytes[str(cache/'val/index.csv')].decode('utf-8-sig'))))
        require(len(rows) == 18514 and len(indices) == 4258, 'row count contract')
        for target in selection:
            row = rows[target['r_pack_index']]
            require(row['split'] == 'validation', 'selected R split')
            require(int(row['index']) == target['r_pack_index'], 'selected R row index')
            require(row['station_id'] == target['station'] and dt(row['timestamp_utc']) == dt(target['target_time_utc']), 'selected R identity')
            for field in ('original_path', 'original_sha256', 'sidecar_path', 'sidecar_sha256'):
                require(row[field] == target[field], f'selected R {field}')
            require(inventory['forecast']['station_order'][target['station_index']] == target['station'], 'station axis')
            require(sorted([p['lead_minutes'] for p in target['pairs']] + [p['lead_minutes'] for p in target['missing_leads']]) == list(range(15, 241, 15)), 'all 16 leads retained')
            for pair in target['pairs']:
                idx = indices[pair['cache_index']]
                require(idx['split'] == 'val' and int(idx['index']) == pair['cache_index'], 'forecast index identity')
                require(idx['seq_id'] == pair['seq_id'] and dt(idx['init_time_utc']) == dt(pair['init_time_utc']), 'forecast init identity')
                require(pair['lead_minutes'] == 15*(pair['lead_index']+1), 'canonical lead')
                require(dt(target['target_time_utc']) == dt(pair['init_time_utc']) + timedelta(minutes=pair['lead_minutes']), 'same target time')

        x_path = allowed(pack/'x_raw.npy')
        checks = [dict(path=str(x_path), file_bytes=inventory['R_pack']['arrays']['x_raw']['file_bytes'],
                       recorded_sha256=inventory['R_pack']['recorded_array_sha256']['x_raw.npy'])]
        checks.extend(inventory['selected_shards'])
        for i, item in enumerate(checks):
            path = allowed(item['path'])
            require(path.stat().st_size == item['file_bytes'], f'file size mismatch: {path}')
            digest = sha(path)
            require(digest == item['recorded_sha256'], f'full file SHA mismatch: {path}')
            report['full_file_SHA_verified'].append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest))
            if i % 10 == 0 or i == len(checks)-1:
                print(f'FULL_SHA_VERIFIED {i+1}/{len(checks)}', flush=True)
        report['full_file_SHA_stream_bytes'] = sum(x['bytes'] for x in report['full_file_SHA_verified'])

        x_raw = np.load(x_path, mmap_mode='r', allow_pickle=False)
        require(x_raw.shape == (18514, 16, 16, 16) and x_raw.dtype.str == '<f4', 'R shape/dtype')
        real = np.stack([np.array(x_raw[t['r_pack_index'], 13:16], copy=True) for t in selection])
        del x_raw
        report['semantic_reads'].append(dict(path=str(x_path), members='64 selected rows, channels 13:16 only', bytes=real.nbytes))
        forecast = np.empty((872, 3, 16, 16), dtype=np.float32)
        pair_records, by_shard = [], defaultdict(list)
        for target_position, target in enumerate(selection):
            for pair in target['pairs']:
                pair_position = len(pair_records)
                record = dict(pair_position=pair_position, target_position=target_position,
                    station=target['station'], station_index=target['station_index'],
                    r_pack_index=target['r_pack_index'], target_time_utc=target['target_time_utc'], **pair)
                pair_records.append(record)
                by_shard[pair['shard']].append(record)
        shard_specs = {Path(s['path']).name: s for s in inventory['selected_shards']}
        require(set(by_shard) == set(shard_specs), 'selected shard set equality')
        for name, records in by_shard.items():
            spec = shard_specs[name]
            path = allowed(spec['path'])
            require(path == cache/'val'/name, 'validation shard path')
            gspec, ispec = spec['members']['geometry.npy'], spec['members']['indices.npy']
            require(gspec['shape'] == [spec['samples'], 2, 16, 3, 16, 16] and gspec['dtype'] == '<f4', 'forecast geometry shape/dtype')
            require(ispec['shape'] == [spec['samples']] and ispec['dtype'] == '<i8', 'forecast index shape/dtype')
            require(gspec['compression'] == ispec['compression'] == 'ZIP_STORED', 'seek requires uncompressed members')
            # Full-file hash already binds these inventory header offsets to
            # the exact shard bytes. Seek only selected row/station/lead patches.
            with path.open('rb') as stream:
                index_rows = sorted({r['shard_row'] for r in records})
                actual_indices = {}
                for shard_row in index_rows:
                    require(0 <= shard_row < spec['samples'], 'shard row bound')
                    stream.seek(ispec['data_file_offset'] + shard_row * 8)
                    raw = stream.read(8)
                    require(len(raw) == 8, 'index short read')
                    actual_indices[shard_row] = int(np.frombuffer(raw, dtype='<i8')[0])
                for record in records:
                    require(actual_indices[record['shard_row']] == record['cache_index'], 'shard/index disagreement')
                    offset = ((record['shard_row']*2 + record['station_index'])*16 + record['lead_index'])*3*16*16*4
                    stream.seek(gspec['data_file_offset'] + offset)
                    raw = stream.read(3*16*16*4)
                    require(len(raw) == 3*16*16*4, 'geometry short read')
                    forecast[record['pair_position']] = np.frombuffer(raw, dtype='<f4').reshape(3,16,16)
            report['semantic_reads'].append(dict(path=str(path), members='selected indices.npy values and selected geometry.npy patches only', index_rows=len(index_rows), geometry_pairs=len(records), bytes=len(index_rows)*8 + len(records)*3*16*16*4))

        real_pair = real[np.array([r['target_position'] for r in pair_records])]
        aggregate = channel_stats(real_pair.transpose(1,0,2,3).reshape(3,-1), forecast.transpose(1,0,2,3).reshape(3,-1))
        for c, name in enumerate(CHANNELS):
            diff = np.abs(forecast[:,c].astype(np.float64) - real_pair[:,c].astype(np.float64))
            comparable = np.isfinite(diff)
            if comparable.any():
                position, row, col = np.unravel_index(np.where(comparable,diff,-np.inf).argmax(), diff.shape)
                aggregate[name]['worst_location'] = dict(**{k:v for k,v in pair_records[position].items() if k not in ('init_time_bjt',)}, patch_row=int(row), patch_col=int(col), real_value=float(real_pair[position,c,row,col]), forecast_value=float(forecast[position,c,row,col]))
        for record in pair_records:
            position = record['pair_position']
            record['channels'] = channel_stats(real_pair[position].reshape(3,-1), forecast[position].reshape(3,-1))
            record['all_elements_within_abs_tolerance'] = all(s['finite_pairs'] == s['elements'] == s['within_abs_tolerance'] for s in record['channels'].values())
            for c, name in enumerate(CHANNELS):
                diff = np.abs(forecast[position,c].astype(np.float64) - real_pair[position,c].astype(np.float64))
                comparable = np.isfinite(diff)
                if comparable.any():
                    row, col = np.unravel_index(np.where(comparable,diff,-np.inf).argmax(), diff.shape)
                    record['channels'][name]['worst_patch_pixel'] = dict(row=int(row), col=int(col), real_value=float(real_pair[position,c,row,col]), forecast_value=float(forecast[position,c,row,col]))
        targets = []
        for target_position, target in enumerate(selection):
            positions = [r['pair_position'] for r in pair_records if r['target_position'] == target_position]
            entry = dict(target_position=target_position, station=target['station'], target_time_utc=target['target_time_utc'], r_pack_index=target['r_pack_index'], pair_positions=positions, missing_leads=target['missing_leads'], real_geometry_finite=bool(np.isfinite(real[target_position]).all()), real_day_mask_binary=bool(np.isin(real[target_position,2], (0.,1.)).all()))
            if positions:
                values = forecast[positions].astype(np.float64)
                finite = bool(np.isfinite(values).all())
                spread = np.max(values,axis=0)-np.min(values,axis=0)
                entry.update(forecast_geometry_finite=finite, forecast_day_mask_binary=bool(np.isin(values[:,2],(0.,1.)).all()),
                    cross_init_max_range_by_channel={name: float(spread[c].max()) if finite else None for c,name in enumerate(CHANNELS)},
                    cross_init_exact_equal=finite and bool(np.all(spread == 0)),
                    cross_init_all_within_abs_tolerance=finite and bool(np.all(spread <= ATOL)))
            else:
                entry['forecast_status'] = 'UNAVAILABLE_NO_FORECAST_PAIRS_RETAINED'
            targets.append(entry)
        checks = dict(all_selected_real_geometry_finite=bool(np.isfinite(real).all()),
            all_selected_forecast_geometry_finite=bool(np.isfinite(forecast).all()),
            all_selected_real_day_mask_binary=bool(np.isin(real[:,2],(0.,1.)).all()),
            all_selected_forecast_day_mask_binary=bool(np.isin(forecast[:,2],(0.,1.)).all()),
            all_pairs_within_abs_tolerance=all(r['all_elements_within_abs_tolerance'] for r in pair_records),
            all_available_targets_cross_init_exact_equal=all(t['cross_init_exact_equal'] for t in targets if t['pair_positions']),
            all_available_targets_cross_init_within_abs_tolerance=all(t['cross_init_all_within_abs_tolerance'] for t in targets if t['pair_positions']))
        with args.values.open('xb') as stream:
            np.savez_compressed(stream, real_geometry=real, forecast_geometry=forecast,
                pair_target_position=np.array([r['target_position'] for r in pair_records],dtype=np.int64),
                r_pack_index=np.array([t['r_pack_index'] for t in selection],dtype=np.int64),
                selection_sha256=np.array(SELECTION_SHA), channels=np.array(CHANNELS), atol=np.array(ATOL))
        require(args.values.stat().st_size < 4*1024*1024, 'saved geometry budget exceeded')
        report.update(state='PASS_GEOMETRY_EQUALITY_COORDINATE_GATE_PENDING' if all(checks.values()) else 'WARN_GEOMETRY_MISMATCH',
            checks=checks, targets_selected=64, targets_with_any_pairs=58, targets_without_pairs=6,
            matched_pairs=872, unavailable_leads=152, selected_shards=59,
            pairs_all_elements_within_tolerance=sum(r['all_elements_within_abs_tolerance'] for r in pair_records),
            aggregate_channels=aggregate, targets=targets, pairs=pair_records,
            semantic_tensor_bytes_read=sum(r['bytes'] for r in report['semantic_reads']),
            values_file=dict(path=str(args.values.resolve()), sha256=sha(args.values), bytes=args.values.stat().st_size,
                             semantic='geometry only; real array includes all 64 targets and forecast array follows pairs order'),
            interpretation='Geometry equality is a necessary nuisance-control gate, not proof of identical coordinates/pixels or AGRI-only error attribution. No R inference, COT target comparison, or accuracy result is produced.',
            next_gate='Verify selected original/sidecar grid_lat/grid_lon against frozen station grid patches; if geometry differs, trace source formula/time basis and stop before any AGRI-only diagnostic.')
    except Exception as error:
        report.update(state='FAIL_CONTRACT', original_failure_type=type(error).__name__, original_failure=str(error))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - start
        with args.output.open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        print(json.dumps({k: report[k] for k in ('state','selection_sha256','elapsed_seconds')},indent=2),flush=True)
    print(json.dumps(dict(checks=report['checks'], aggregate_channels=report['aggregate_channels'], pairs_all_elements_within_tolerance=report['pairs_all_elements_within_tolerance']),indent=2),flush=True)


if __name__ == '__main__':
    main()
