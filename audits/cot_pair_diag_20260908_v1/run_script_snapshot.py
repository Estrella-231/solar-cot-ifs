"""Frozen 64-target/872-pair CPU CPP-reference diagnostic; never train or use test.

Outputs in a NEW --output-dir: report.json and
cot_pair_predictions_20260908_v1.npz. Both R paths are recomputed on CPU;
saved GPU COT is never used. Read the frozen protocol before running.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import platform
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
from train_cot_repaired import COTUNet


SELECTION_SHA = '8a06a7f06cf3339cd8f067111d5397c173ba63868598c6904348362994b08318'
RECEIPTS = {
    'cot_pair_inventory_20260908.json': '7e7f5f87cced284797ed46b47104c16ad62770c62eb6f611f88036637fa8cf2d',
    'cot_pair_coordinates_20260908.json': '4298da1f31e36dcba206bc5ebbd3e12298ad5a39748e9f9f6140e99b633a551d',
    'cot_pair_geometry_20260908_v2.json': 'ebe0ee3e5e06143685eb0a0c6cc4f494d01e7df039b7f90fd61cd500bc31c796',
    'cot_pair_geometry_20260908.json': '078a1f1eef2c1c8864df5c44d14cb3b246a4118f625225c9a415b4c0586737f1',
    'r_cache_numeric_witness_20260908.json': '482424f4f18a31822cb00a36b605b0abe3464b4cc5f02be0c18f020891e5b3b3',
}
PROTOCOL = 'docs/COT_PAIR_DIAGNOSTIC_PROTOCOL_20260908.md'
PROTOCOL_SHA = '74a94abda6c1b7054b26baf586bf3b7e2e94fe62c71efc2ac5a0f3ee3998ccf5'
BATCH = 32
STATIONS = ('sili', 'zhujia')
LEADS = tuple(range(15, 241, 15))
FEATURES = [f'C{i:02d}' for i in (1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15)] + [
    'cosSOZ', 'cosRAA', 'day_mask']


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def utc(value):
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(stamp.utcoffset() == timedelta(0), 'naive or non-UTC target/init')
    return stamp


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def checked_hash(path, expected, report):
    path = Path(path)
    digest = sha(path)
    require(digest == expected, f'SHA mismatch: {path}')
    report['verified_files'].append(dict(path=str(path), sha256=digest, bytes=path.stat().st_size))
    return digest


def load_contracts(root, protocol_sha, report):
    require(len(protocol_sha) == 64 and all(c in '0123456789abcdef' for c in protocol_sha),
            'provide the reviewed frozen protocol SHA-256')
    checked_hash(root / PROTOCOL, protocol_sha, report)
    report['protocol_sha256'] = protocol_sha
    gates = {}
    for name, digest in RECEIPTS.items():
        checked_hash(root / 'audits' / name, digest, report)
        gates[name] = read_json(root / 'audits' / name)
    inv = gates['cot_pair_inventory_20260908.json']
    coordinate = gates['cot_pair_coordinates_20260908.json']
    geometry = gates['cot_pair_geometry_20260908_v2.json']
    require(inv['root'] == str(root), 'inventory root')
    selection = inv['selection']
    canonical = json.dumps(dict(rule=inv['selection_rule'], targets=selection),
                           sort_keys=True, separators=(',', ':')).encode()
    require(hashlib.sha256(canonical).hexdigest() == inv['selection_sha256'] == SELECTION_SHA,
            'fixed selection changed')
    require(len(selection) == 64 and sum(bool(t['pairs']) for t in selection) == 58, '64/58 target contract')
    require(sum(len(t['pairs']) for t in selection) == 872, '872 pairs contract')
    require(sum(len(t['missing_leads']) for t in selection) == 152, '152 unavailable leads contract')
    require(len(inv['selected_shards']) == 59 and inv['selection_rule']['split'] == 'validation', 'val/shard scope')
    require(inv['forecast']['station_order'] == list(STATIONS), 'station order')
    require(inv['R_pack']['input_features'] == FEATURES, 'channel order')
    for gate in (coordinate, geometry):
        require(gate['inventory_sha256'] == RECEIPTS['cot_pair_inventory_20260908.json']
                and gate['selection_sha256'] == SELECTION_SHA, 'gate selection binding')
    require(coordinate['state'] == 'PASS_58_TARGET_COORDINATE_BINDINGS'
            and coordinate['matched_targets'] == 58 and coordinate['test_used'] is False
            and coordinate['all_coordinates_exact_equal'] and coordinate['max_abs_degrees'] == 0,
            'coordinate gate did not pass exactly')
    require(geometry['state'] == 'PASS_GEOMETRY_EQUALITY_COORDINATE_GATE_PENDING'
            and geometry['matched_pairs'] == 872 and geometry['test_payloads_read'] == 0,
            'geometry gate did not complete')
    require(all(v['max_abs'] == 0 and v['exact_equal'] == v['elements']
                for v in geometry['aggregate_channels'].values()), 'geometry gate not exact')
    require(geometry['prior_failure']['sha256'] == RECEIPTS['cot_pair_geometry_20260908.json'],
            'prior parser-failure evidence changed')
    report['gates'] = dict(coordinates=coordinate['state'], geometry=geometry['state'],
        both_actual_checks_pass=True, prior_geometry_failure_preserved=True,
        old_cache_numeric_warning='max 0.0010333061218261719 > original 0.001 gate; remains open')
    coord_rows = {(t['station'], t['target_time_utc']): t for t in coordinate['targets']}
    for target in selection:
        if not target['pairs']:
            continue
        row = coord_rows[(target['station'], target['target_time_utc'])]
        require(row['r_pack_index'] == target['r_pack_index'] and row['pairs'] == len(target['pairs']),
                'coordinate selected row binding')
        require(row['S_patch_bounds'] == inv['forecast']['station_patches'][target['station']], 'coordinate crop')
        for kind in ('original', 'sidecar'):
            matches = [s for s in row['sources'] if s['kind'] == kind]
            require(len(matches) == 1 and matches[0]['sha256'] == target[kind + '_sha256']
                    and matches[0]['path'] == target[kind + '_path'], 'coordinate source binding')

    metadata = {}
    for receipt in inv['metadata_accessed']:
        if receipt['mode'] != 'metadata':
            continue
        path = Path(receipt['path']).resolve()
        require(path.is_relative_to(root), f'metadata outside project: {path}')
        raw = path.read_bytes()
        require(len(raw) == receipt['bytes'] and hashlib.sha256(raw).hexdigest() == receipt['sha256'],
                f'metadata changed: {path}')
        report['verified_files'].append(dict(path=str(path), sha256=receipt['sha256'], bytes=len(raw)))
        metadata[str(path)] = raw
    pack, cache = Path(inv['R_pack']['path']), Path(inv['forecast']['path'])
    require(pack == root / 'data/cot_repaired_pack_20260907_v1', 'R pack path')
    require(cache == root / 'data/frozen_forecast_trainval_20260907_v1', 'cache path')
    rows = list(csv.DictReader(io.StringIO(metadata[str(pack / 'rows.csv')].decode('utf-8-sig'))))
    indices = list(csv.DictReader(io.StringIO(metadata[str(cache / 'val/index.csv')].decode('utf-8-sig'))))
    require(len(rows) == 18514 and len(indices) == 4258, 'pack/validation row counts')
    contract = read_json(cache / 'contract.json')
    rc, sc = (read_json(root / f'configs/{name}_frozen_{suffix}.json')
              for name, suffix in (('r', 'repaired_seed42'), ('s', 'hunan_seed42')))
    require(rc == contract['R'] and sc == contract['S'] and not rc['test_used'] and not sc['test_used'],
            'frozen S/R config binding')
    for name, config in (('R', rc), ('S', sc)):
        require(config['checkpoint_sha256'] == inv['forecast'][name + '_checkpoint_sha256'], 'checkpoint inventory binding')
        checked_hash(config['checkpoint'], config['checkpoint_sha256'], report)
    for key in ('normalization', 'manifest'):
        checked_hash(sc[key], sc[key + '_sha256'], report)
    checked_hash(root / 'scripts/cache_frozen_forecast.py', contract['script_sha256'], report)
    checked_hash(inspect.getfile(COTUNet), contract['R_source_sha256'], report)
    s_sources = root.parent / 'HunanSimVPDDP_20260907/scripts'
    for name, digest in contract['imported_source_sha256'].items():
        checked_hash(s_sources / name, digest, report)
    pack_config = read_json(pack / 'pack_config.json')
    for name, digest in pack_config['code_sha256'].items():
        checked_hash(root / 'scripts' / name, digest, report)
    checkpoint = torch.load(rc['checkpoint'], map_location='cpu')
    meta, norm = checkpoint['metadata'], read_json(pack / 'norm.json')
    require(meta['test_used'] is False and meta['source_code_sha256'] == contract['R_source_sha256'], 'R metadata scope/source')
    require(Path(meta['pack']) == pack and meta['norm'] == norm, 'R pack/norm identity')
    require(norm['test_used'] is False and norm['train_samples'] == 11978
            and norm['feature_names'] == FEATURES and norm['cot_scale'] == 100, 'R train normalization')
    require(meta['nonfinite_input_imputation'] == 'normalized zero (train mean)', 'R imputation contract')
    require(np.isfinite(norm['mean']).all() and np.isfinite(norm['std']).all()
            and (np.asarray(norm['std']) > 0).all(), 'R normalization finite/positive')
    hashes = read_json(pack / 'SHA256.json')
    for name in ('x_raw.npy', 'target.npy', 'mask.npy', 'rows.csv', 'norm.json', 'pack_config.json'):
        require(hashes[name] == meta['pack_sha256'][name], f'R checkpoint/pack binding: {name}')
        if name.endswith('.npy'):
            require(hashes[name] == inv['R_pack']['recorded_array_sha256'][name], 'inventory array binding')
        checked_hash(pack / name, hashes[name], report)
    report['R_checkpoint_sha256'] = rc['checkpoint_sha256']
    report['S_checkpoint_sha256'] = sc['checkpoint_sha256']
    report['R_source_sha256'] = contract['R_source_sha256']
    report['R_norm_sha256'] = hashes['norm.json']

    pair_records, unique_targets = [], []
    for target_position, target in enumerate(selection):
        row = rows[target['r_pack_index']]
        require(row['split'] == 'validation' and int(row['index']) == target['r_pack_index'], 'selected R validation row')
        require(row['station_id'] == target['station'] and utc(row['timestamp_utc']) == utc(target['target_time_utc']), 'selected R identity')
        for key in ('original_path', 'original_sha256', 'sidecar_path', 'sidecar_sha256'):
            require(row[key] == target[key], 'selected source identity')
        require(STATIONS[target['station_index']] == target['station'], 'station axis')
        require(sorted([p['lead_minutes'] for p in target['pairs']] +
                       [p['lead_minutes'] for p in target['missing_leads']]) == list(LEADS), 'all leads retained')
        if not target['pairs']:
            continue
        unique_index = len(unique_targets)
        unique_targets.append(dict(target_position=target_position, valid_pixels=int(row['valid_pixels']), **target))
        for pair in target['pairs']:
            idx = indices[pair['cache_index']]
            require(idx['split'] == 'val' and int(idx['index']) == pair['cache_index'], 'forecast validation index')
            require(idx['seq_id'] == pair['seq_id'] and utc(idx['init_time_utc']) == utc(pair['init_time_utc']), 'forecast init identity')
            require(pair['lead_minutes'] == 15 * (pair['lead_index'] + 1), 'canonical lead')
            require(utc(target['target_time_utc']) == utc(pair['init_time_utc']) + timedelta(minutes=pair['lead_minutes']), 'same target')
            record = dict(pair_position=len(pair_records), unique_index=unique_index,
                          target_position=target_position, station=target['station'], station_index=target['station_index'],
                          r_pack_index=target['r_pack_index'], target_time_utc=target['target_time_utc'], **pair)
            old = geometry['pairs'][len(pair_records)]
            require(all(old[k] == v for k, v in record.items() if k != 'unique_index'), 'geometry pair order/key binding')
            pair_records.append(record)
    require(len(pair_records) == 872 and len(unique_targets) == 58, 'fixed matched count')
    report['selection'] = selection  # Includes all six unmatched targets and 152 missing leads.
    report['coordinate_source_binding_scope'] = 'prior gate verified all 116 source-file SHAs; selected pack rows bind to those exact SHAs'
    return inv, pack, cache, checkpoint, norm, unique_targets, pair_records


def load_selected_inputs(inv, pack, cache, targets, pairs, report):
    chosen = np.asarray([t['r_pack_index'] for t in targets], dtype=np.int64)
    selected = {}
    for name, expected_shape, dtype in (
        ('x_raw', (18514, 16, 16, 16), np.float32),
        ('target', (18514, 1, 16, 16), np.float32),
        ('mask', (18514, 1, 16, 16), np.bool_),
    ):
        array = np.load(pack / (name + '.npy'), mmap_mode='r', allow_pickle=False)
        require(array.shape == expected_shape and array.dtype == dtype, f'R array shape/dtype: {name}')
        selected[name] = np.array(array[chosen], copy=True)
        del array
    real = selected['x_raw']
    # Match training exactly: multiply serialized float32 by 100 in float32,
    # then convert to float64. No reconstruction from a different CPP source.
    reference = (selected['target'] * np.float32(100.0)).astype(np.float64)[:, 0]
    mask = selected['mask'][:, 0]
    require(np.isfinite(reference).all() and ((reference >= 0) & (reference <= 100)).all(), 'reference range/finite')
    require(mask.reshape(58, -1).any(1).all(), 'original selected masks must retain their valid pixels')
    require(np.array_equal(mask.sum((1, 2)), [t['valid_pixels'] for t in targets]), 'original mask count binding')
    require(np.isfinite(real[:, 13:]).all(), 'real geometry finite')
    forecast = np.empty((872, 16, 16, 16), dtype=np.float32)
    grouped = defaultdict(list)
    for record in pairs:
        grouped[record['shard']].append(record)
    specs = {Path(s['path']).name: s for s in inv['selected_shards']}
    require(set(grouped) == set(specs), 'exact selected shard set')
    complete = read_json(cache / 'val/complete.json')
    require(complete['state'] == 'COMPLETE' and complete['test_used'] is False, 'validation completion')
    receipts = {s['name']: s for s in complete['shards']}
    for number, (name, records) in enumerate(grouped.items(), 1):
        spec, path = specs[name], cache / 'val' / name
        require(Path(spec['path']) == path and path.stat().st_size == spec['file_bytes'], 'shard path/size')
        receipt = read_json(path.with_suffix('.json'))
        require(receipt == receipts[name] and receipt['sha256'] == spec['recorded_sha256']
                and receipt['samples'] == spec['samples'], 'shard sidecar/complete binding')
        checked_hash(path, spec['recorded_sha256'], report)
        with np.load(path, allow_pickle=False) as shard:
            # Deliberately do not open cot_log1p/cot_features/r_latent_mean.
            agri, geom, indices = shard['predicted_agri_physical'], shard['geometry'], shard['indices']
        n = spec['samples']
        require(agri.shape == (n, 2, 16, 13, 16, 16) and agri.dtype == np.float32, 'forecast AGRI contract')
        require(geom.shape == (n, 2, 16, 3, 16, 16) and geom.dtype == np.float32, 'forecast geometry contract')
        require(indices.shape == (n,) and indices.dtype == np.int64, 'shard indices contract')
        for rec in records:
            r, s, lead = rec['shard_row'], rec['station_index'], rec['lead_index']
            require(0 <= r < n and indices[r] == rec['cache_index'], 'selected shard sequence binding')
            g = geom[r, s, lead]
            require(np.array_equal(g, real[rec['unique_index'], 13:]), 'selected pair geometry must remain exactly equal')
            forecast[rec['pair_position']] = np.concatenate((agri[r, s, lead], g), axis=0)
        del agri, geom, indices
        print(f'INPUT_SHARD_VERIFIED {number}/59', flush=True)
    report['selected_inputs'] = dict(unique_real_rows=58, forecast_pairs=872, shards=59,
        raw_real_sha256=hashlib.sha256(real.tobytes()).hexdigest(),
        raw_forecast_sha256=hashlib.sha256(forecast.tobytes()).hexdigest(),
        geometry_exact_equal_pairs=872, original_reference_mask_unchanged=True,
        reference_conversion='float32(target.npy * float32(100)) then float64',
        original_masks_valid_pixels_unique=int(mask.sum()))
    return real, forecast, reference, mask


@torch.no_grad()
def predict(model, raw, norm, branch, report):
    mean = np.asarray(norm['mean'], np.float32)[None, :, None, None]
    std = np.asarray(norm['std'], np.float32)[None, :, None, None]
    normalized = (raw - mean) / std
    bad = ~np.isfinite(normalized)
    report['normalization'][branch] = dict(nonfinite_values_imputed=int(bad.sum()),
        imputation='normalized zero (train mean), frozen training rule', clipping=False)
    normalized[bad] = 0
    require(normalized.dtype == np.float32 and np.isfinite(normalized).all(), 'normalized input contract')
    outputs = []
    for start in range(0, len(normalized), BATCH):
        batch = normalized[start:start + BATCH]
        count = len(batch)
        if count < BATCH:
            batch = np.concatenate((batch, np.repeat(batch[-1:], BATCH - count, axis=0)), axis=0)
        tensor = torch.from_numpy(np.ascontiguousarray(batch))
        require(tensor.device.type == 'cpu' and tensor.dtype == torch.float32, 'CPU FP32 input')
        value = model(tensor)
        require(value.device.type == 'cpu' and value.dtype == torch.float32
                and tuple(value.shape) == (BATCH, 1, 16, 16), 'CPU FP32 output')
        result = value[:count, 0].numpy().copy()
        require(np.isfinite(result).all() and (result >= 0).all(), 'nonfinite/negative log1p output')
        outputs.append(result)
    return np.concatenate(outputs)


def metrics(log_prediction, reference, mask):
    n = int(mask.sum())
    if not n:
        return dict(valid_pixels=0, cot_mae=None, cot_rmse=None, cot_bias=None,
                    log1p_mae=None, log1p_rmse=None)
    log_pred = log_prediction.astype(np.float64)[mask]
    truth = reference.astype(np.float64)[mask]
    physical = np.expm1(log_pred)
    require(np.isfinite(physical).all(), 'nonfinite physical COT; do not clip or drop')
    error, log_error = physical - truth, log_pred - np.log1p(truth)
    return dict(valid_pixels=n, cot_mae=float(np.mean(np.abs(error))),
        cot_rmse=float(np.sqrt(np.mean(error ** 2))), cot_bias=float(np.mean(error)),
        log1p_mae=float(np.mean(np.abs(log_error))), log1p_rmse=float(np.sqrt(np.mean(log_error ** 2))))


def summarize_saved(saved):
    """Compute metrics only from reloaded serialized arrays, using NumPy FP64."""
    result = dict(metric_source='reloaded NPZ arrays; NumPy float64',
        pair_aggregation='valid-pixel pooled; target references repeat across available leads',
        unique_real_aggregation='one original reference and real-R prediction per station/target',
        bias_definition='predicted physical COT minus reference physical COT',
        by_station_lead=[], unique_real_targets=[], unique_real_by_station=[])

    def paired_group(select):
        rows = np.flatnonzero(select)
        return dict(n_pair=len(rows), n_unique_target=len(np.unique(saved['r_pack_index'][rows])),
            real=metrics(saved['real_log1p'][rows], saved['reference_COT'][rows], saved['mask'][rows]),
            forecast=metrics(saved['predicted_log1p'][rows], saved['reference_COT'][rows], saved['mask'][rows]))

    for station in STATIONS:
        for lead in LEADS:
            group = paired_group((saved['station'] == station) & (saved['lead_minutes'] == lead))
            group.update(station=station, lead_minutes=lead, selected_targets=32,
                         unavailable_targets=32 - group['n_pair'])
            result['by_station_lead'].append(group)
    result['all_pairs'] = paired_group(np.ones(872, dtype=bool))
    unique_ids, first = np.unique(saved['r_pack_index'], return_index=True)
    require(len(unique_ids) == 58, 'saved unique target count')
    for ident, index in zip(unique_ids, first):
        rows = np.flatnonzero(saved['r_pack_index'] == ident)
        for key in ('real_log1p', 'reference_COT', 'mask', 'station', 'target_time_utc'):
            require(all(np.array_equal(saved[key][index], saved[key][r]) for r in rows),
                    f'duplicate target changed: {key}')
        result['unique_real_targets'].append(dict(r_pack_index=int(ident), station=str(saved['station'][index]),
            target_time_utc=str(saved['target_time_utc'][index]), available_leads=len(rows),
            metrics=metrics(saved['real_log1p'][index:index + 1], saved['reference_COT'][index:index + 1], saved['mask'][index:index + 1])))
    for station in STATIONS:
        select = first[saved['station'][first] == station]
        result['unique_real_by_station'].append(dict(station=station, n_unique_target=len(select),
            metrics=metrics(saved['real_log1p'][select], saved['reference_COT'][select], saved['mask'][select])))
    result['unique_real_all'] = dict(n_unique_target=58,
        metrics=metrics(saved['real_log1p'][first], saved['reference_COT'][first], saved['mask'][first]))
    require(sum(row['n_pair'] for row in result['by_station_lead']) == 872, 'metric pair count')
    require(sum(row['unavailable_targets'] for row in result['by_station_lead']) == 152, 'metric missing lead count')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output_dir.resolve()
    require(out.is_relative_to(root / 'audits'), 'output must be a new directory under this project audits')
    require(not out.exists(), 'refuse existing output directory, including any failed prior diagnostic')
    out.mkdir(parents=True, exist_ok=False)
    report_path = out / 'report.json'
    started = time.monotonic()
    report = dict(state='IN_PROGRESS', started_utc=datetime.now(timezone.utc).isoformat(),
        root=str(root), script_sha256=sha(Path(__file__)), selection_sha256=SELECTION_SHA,
        test_used=False, test_payloads_read=0, GHI_read=False, training=False, GPU_used=False,
        models_or_cache_modified=False, old_saved_R_outputs_used=False, normalization={}, verified_files=[],
        CPP_reference='retrieval reference, not independent physical truth; source NC/checkpoint binding unverified',
        interpretation='descriptive validation input-propagation diagnostic; R was validation-selected; repeated targets are not independent',
        output_names=dict(report='report.json', predictions='cot_pair_predictions_20260908_v1.npz'))
    atomic_json(report_path, report)
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        if hasattr(os, 'sched_getaffinity'):
            cpus = sorted(os.sched_getaffinity(0))[:2]
            os.sched_setaffinity(0, set(cpus))
        require(torch.get_default_dtype() == torch.float32, 'default FP32 contract')
        report['runtime'] = dict(python=platform.python_version(), platform=platform.platform(),
            torch=torch.__version__, numpy=np.__version__, cpu_threads=torch.get_num_threads(),
            interop_threads=torch.get_num_interop_threads(), mkldnn_enabled=torch.backends.mkldnn.enabled,
            cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], default_dtype=str(torch.get_default_dtype()),
            cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
            batch=BATCH, partial_batch='repeat final input to 32, discard padded predictions',
            device='cpu', autocast=False, eval=True, no_grad=True)
        loaded = load_contracts(root, PROTOCOL_SHA, report)
        inv, pack, cache, checkpoint, norm, targets, pairs = loaded
        report['state'] = 'METADATA_AND_GATES_VERIFIED'
        atomic_json(report_path, report)
        real, forecast, reference, mask = load_selected_inputs(inv, pack, cache, targets, pairs, report)
        model = COTUNet().float().eval().requires_grad_(False)
        model.load_state_dict(checkpoint['model'], strict=True)
        require(all(p.device.type == 'cpu' and p.dtype == torch.float32 and not p.requires_grad
                    for p in model.parameters()), 'frozen CPU FP32 R')
        real_log = predict(model, real, norm, 'real', report)
        forecast_log = predict(model, forecast, norm, 'forecast', report)
        unique_index = np.asarray([r['unique_index'] for r in pairs], dtype=np.int64)
        arrays = dict(predicted_log1p=forecast_log, real_log1p=real_log[unique_index],
            reference_COT=reference[unique_index], mask=mask[unique_index], unique_index=unique_index)
        for key in ('station', 'target_time_utc', 'init_time_utc', 'seq_id', 'shard'):
            arrays[key] = np.asarray([r[key] for r in pairs], dtype=str)
        for key in ('station_index', 'lead_minutes', 'lead_index', 'r_pack_index', 'target_position', 'pair_position', 'shard_row'):
            arrays[key] = np.asarray([r[key] for r in pairs], dtype=np.int64)
        arrays['sequence'] = np.asarray([r['cache_index'] for r in pairs], dtype=np.int64)
        predictions = out / 'cot_pair_predictions_20260908_v1.npz'
        with predictions.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        report['state'] = 'PREDICTIONS_SAVED_PENDING_NUMPY_RECOMPUTE'
        report['predictions'] = dict(path=str(predictions), sha256=sha(predictions), bytes=predictions.stat().st_size,
            shapes={k: list(v.shape) for k, v in arrays.items()}, dtypes={k: str(v.dtype) for k, v in arrays.items()})
        atomic_json(report_path, report)
        with np.load(predictions, allow_pickle=False) as source:
            require(set(source.files) == set(arrays), 'saved array key set')
            saved = {key: source[key] for key in source.files}
        for key, value in arrays.items():
            require(saved[key].dtype == value.dtype and np.array_equal(saved[key], value), f'saved reload mismatch: {key}')
        report['metrics'] = summarize_saved(saved)
        checked_hash(root / PROTOCOL, PROTOCOL_SHA, report)
        require(sha(Path(__file__)) == report['script_sha256'], 'script changed during run')
        require(sha(predictions) == report['predictions']['sha256'], 'predictions changed during metric computation')
        report['serialization_verification'] = 'all output arrays reload exactly; all metrics computed from saved arrays with NumPy float64'
        report['state'] = 'COMPLETE_CPU_CPP_REFERENCE_DIAGNOSTIC_WITH_CACHE_WARNING'
    except Exception as error:
        report.update(state='ERROR_INCOMPLETE_DIAGNOSTIC', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        report['elapsed_seconds'] = time.monotonic() - started
        atomic_json(report_path, report)
        print(report['state'], flush=True)


if __name__ == '__main__':
    main()
