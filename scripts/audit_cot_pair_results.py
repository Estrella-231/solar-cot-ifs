"""Independent NumPy-only audit of frozen CPU same-target COT diagnostics.

No import of the inference runner, torch, or model; no prediction recompute.
Reopens only the saved pair arrays and selected train/validation-pack target
and mask rows. Full SHA streams bind the immutable source/code/weight files.
"""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np

SELECTION_SHA = '8a06a7f06cf3339cd8f067111d5397c173ba63868598c6904348362994b08318'
INVENTORY_SHA = '7e7f5f87cced284797ed46b47104c16ad62770c62eb6f611f88036637fa8cf2d'
PROTOCOL_SHA = '74a94abda6c1b7054b26baf586bf3b7e2e94fe62c71efc2ac5a0f3ee3998ccf5'
ATOL = 1e-10
METRICS = ('cot_mae', 'cot_rmse', 'cot_bias', 'log1p_mae', 'log1p_rmse')
STATIONS = ('sili', 'zhujia')
LEADS = tuple(range(15, 241, 15))


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def stamp(value):
    result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    require(result.utcoffset() == timedelta(0), 'UTC timestamp required')
    return result


def measure(log_values, physical_reference, validity):
    """Independent pooled float64 sums/counts; original mask is the only filter."""
    selected_log = np.asarray(log_values, np.float64)[validity]
    selected_reference = np.asarray(physical_reference, np.float64)[validity]
    count = selected_reference.size
    if not count:
        return dict(valid_pixels=0, **{key: None for key in METRICS})
    physical_prediction = np.expm1(selected_log)
    require(np.isfinite(physical_prediction).all(), 'finite un-clipped physical predictions')
    error = physical_prediction - selected_reference
    log_error = selected_log - np.log1p(selected_reference)
    return dict(valid_pixels=int(count),
        cot_mae=float(np.sum(np.abs(error), dtype=np.float64) / count),
        cot_rmse=float(np.sqrt(np.sum(np.square(error), dtype=np.float64) / count)),
        cot_bias=float(np.sum(error, dtype=np.float64) / count),
        log1p_mae=float(np.sum(np.abs(log_error), dtype=np.float64) / count),
        log1p_rmse=float(np.sqrt(np.sum(np.square(log_error), dtype=np.float64) / count)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--diagnostic-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'audit output must be new')
    start = time.monotonic()
    audit = dict(state='RUNNING', created_utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=sha(Path(__file__)), tolerance=ATOL, rtol=0, prediction_recomputed=False,
        model_loaded=False, torch_imported=False, test_used=False, test_payloads_read=0,
        GHI_read=False, GPU_used=False, training=False, verified_files=[], metric_comparisons=[],
        scientific_scope='fixed validation descriptive CPP-reference diagnostic; no GHI mechanism or independent physical truth claim')
    try:
        require(hasattr(os, 'sched_getaffinity'), 'Linux CPU affinity expected')
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
        audit['cpu_affinity'] = sorted(os.sched_getaffinity(0))
        root = args.root.resolve()
        run = args.diagnostic_dir.resolve()
        require(run == root/'audits/cot_pair_diag_20260908_v2', 'fixed diagnostic attempt directory')
        require(args.output.resolve().is_relative_to(root/'audits'), 'audit output scope')
        raw_report = (run/'report.json').read_bytes()
        report = json.loads(raw_report)
        require(report['state'] == 'COMPLETE_CPU_CPP_REFERENCE_DIAGNOSTIC_WITH_CACHE_WARNING',
                'diagnostic not complete; do not audit partial arrays')
        require(report['root'] == str(root), 'diagnostic root')
        require(report['selection_sha256'] == SELECTION_SHA and report['protocol_sha256'] == PROTOCOL_SHA, 'frozen protocol/selection')
        for name in ('test_used', 'GHI_read', 'training', 'GPU_used', 'models_or_cache_modified', 'old_saved_R_outputs_used'):
            require(report[name] is False, f'forbidden source operation: {name}')
        require(report['test_payloads_read'] == 0, 'test reads')
        runtime = report['runtime']
        require(runtime['device'] == 'cpu' and runtime['default_dtype'] == 'torch.float32'
                and runtime['cpu_threads'] == 2 and runtime['batch'] == 32
                and runtime['autocast'] is False and runtime['eval'] is True and runtime['no_grad'] is True,
                'fixed runtime contract')
        audit['source_runtime'] = runtime
        audit['source_report'] = dict(path=str(run/'report.json'), sha256=hashlib.sha256(raw_report).hexdigest(), bytes=len(raw_report))

        def verify(path, expected, role):
            path = Path(path).resolve()
            require(not any(p == 'test' or p.startswith('test_') for p in path.parts), f'forbidden test source: {path}')
            require(path.name != 'labels.csv', 'GHI labels forbidden')
            actual = sha(path)
            require(actual == expected, f'{role} SHA changed: {path}')
            audit['verified_files'].append(dict(role=role, path=str(path), sha256=actual, bytes=path.stat().st_size))
            return path

        inv_path = verify(root/'audits/cot_pair_inventory_20260908.json', INVENTORY_SHA, 'frozen inventory')
        inv = json.loads(inv_path.read_text())
        selection = inv['selection']
        serialized = json.dumps(dict(rule=inv['selection_rule'], targets=selection), sort_keys=True, separators=(',', ':')).encode()
        require(hashlib.sha256(serialized).hexdigest() == SELECTION_SHA, 'selection canonical SHA')
        require(report['selection'] == selection, 'all original selected/missing entries unchanged')
        require(len(selection) == 64 and sum(bool(t['pairs']) for t in selection) == 58,
                '64 targets/58 matched contract')
        require(sum(len(t['pairs']) for t in selection) == 872
                and sum(len(t['missing_leads']) for t in selection) == 152, '872/152 contract')
        verify(root/'docs/COT_PAIR_DIAGNOSTIC_PROTOCOL_20260908.md', PROTOCOL_SHA, 'pre-value protocol')
        verify(root/'scripts/run_cot_pair_diagnostic.py', report['script_sha256'], 'inference source')
        checkpoint = verify(inv['forecast']['R_checkpoint'], inv['forecast']['R_checkpoint_sha256'], 'frozen R weight bytes only')
        require(report['R_checkpoint_sha256'] == inv['forecast']['R_checkpoint_sha256'], 'R checkpoint report binding')
        sources = [v for v in report['verified_files'] if v['sha256'] == report['R_source_sha256']]
        require(len(sources) == 1, 'unique recorded R source')
        verify(sources[0]['path'], report['R_source_sha256'], 'R model source bytes only')
        for filename in ('cot_pair_coordinates_20260908.json', 'cot_pair_geometry_20260908_v2.json'):
            matches = [v for v in report['verified_files'] if Path(v['path']) == root/'audits'/filename]
            require(len(matches) == 1, f'prior gate binding {filename}')
            verify(matches[0]['path'], matches[0]['sha256'], 'prior coordinate/geometry evidence')
        pack = Path(inv['R_pack']['path']).resolve()
        require(pack == root/'data/cot_repaired_pack_20260907_v1', 'fixed R pack')
        verify(pack/'rows.csv', inv['R_pack']['rows_sha256'], 'R row identities')
        verify(pack/'norm.json', inv['R_pack']['norm_sha256'], 'R train normalization')
        for name in ('target.npy', 'mask.npy'):
            verify(pack/name, inv['R_pack']['recorded_array_sha256'][name], 'original reference/mask byte stream')
        rows = list(csv.DictReader(io.StringIO((pack/'rows.csv').read_text())))
        require(len(rows) == 18514 and all(row['split'] in ('train','validation') for row in rows), 'no test rows in pack')
        prediction_path = Path(report['predictions']['path']).resolve()
        require(prediction_path == run/'cot_pair_predictions_20260908_v1.npz', 'fixed prediction file')
        verify(prediction_path, report['predictions']['sha256'], 'saved predictions and references')
        with np.load(prediction_path, allow_pickle=False) as data:
            saved = {name: data[name] for name in data.files}
        keys = set(('predicted_log1p','real_log1p','reference_COT','mask','unique_index','station','target_time_utc',
            'init_time_utc','seq_id','shard','station_index','lead_minutes','lead_index','r_pack_index',
            'target_position','pair_position','shard_row','sequence'))
        require(set(saved) == keys, 'saved exact key set')
        for key, array in saved.items():
            require(array.shape == (872,16,16) if key in ('predicted_log1p','real_log1p','reference_COT','mask') else array.shape == (872,), f'saved shape {key}')
            require(list(array.shape) == report['predictions']['shapes'][key]
                    and str(array.dtype) == report['predictions']['dtypes'][key], f'saved schema receipt {key}')
        require(saved['predicted_log1p'].dtype == saved['real_log1p'].dtype == np.float32, 'FP32 saved predictions')
        require(saved['reference_COT'].dtype == np.float64 and saved['mask'].dtype == np.bool_, 'reference/mask dtype')
        for key in ('predicted_log1p','real_log1p','reference_COT'):
            require(np.isfinite(saved[key]).all() and (saved[key] >= 0).all(), f'finite nonnegative {key}')
        require((saved['reference_COT'] <= 100).all(), 'reference physical range')
        expected_pairs, selected_targets = [], []
        for target_position, target in enumerate(selection):
            require(sorted([p['lead_minutes'] for p in target['pairs']] + [p['lead_minutes'] for p in target['missing_leads']]) == list(LEADS), 'all 16 original lead slots')
            if not target['pairs']:
                continue
            unique_index = len(selected_targets)
            selected_targets.append(target)
            for pair in target['pairs']:
                expected_pairs.append(dict(pair_position=len(expected_pairs), unique_index=unique_index,
                    target_position=target_position, station=target['station'], station_index=target['station_index'],
                    r_pack_index=target['r_pack_index'], target_time_utc=target['target_time_utc'],
                    sequence=pair['cache_index'], **{k: pair[k] for k in ('lead_minutes','lead_index','init_time_utc','seq_id','shard','shard_row')}))
        for name in keys - {'predicted_log1p','real_log1p','reference_COT','mask'}:
            require(np.array_equal(saved[name], np.asarray([p[name] for p in expected_pairs])), f'exact inventory key match: {name}')
        identity_keys = list(zip(saved['station'].tolist(), saved['init_time_utc'].tolist(), saved['target_time_utc'].tolist()))
        require(len(set(identity_keys)) == 872, 'unique station/init/target pairs')
        for record in expected_pairs:
            require(stamp(record['target_time_utc']) - stamp(record['init_time_utc']) == timedelta(minutes=record['lead_minutes']), 'canonical target/init/lead equality')
        chosen = [t['r_pack_index'] for t in selected_targets]
        pack_target = np.load(pack/'target.npy', mmap_mode='r', allow_pickle=False)
        pack_mask = np.load(pack/'mask.npy', mmap_mode='r', allow_pickle=False)
        require(pack_target.shape == pack_mask.shape == (18514,1,16,16), 'original target/mask shape')
        require(pack_target.dtype == np.float32 and pack_mask.dtype == np.bool_, 'original target/mask dtype')
        target_rows = np.asarray(pack_target[chosen]).copy()
        mask_rows = np.asarray(pack_mask[chosen]).copy()[:,0]
        del pack_target, pack_mask
        # Critical representation contract: the multiplication precedes the
        # cast to float64. This checks the original training reference exactly.
        reference_rows = np.multiply(target_rows, np.float32(100), dtype=np.float32).astype(np.float64)[:,0]
        for j, target in enumerate(selected_targets):
            row = rows[target['r_pack_index']]
            require(int(row['index']) == target['r_pack_index'] and row['split'] == 'validation'
                    and row['station_id'] == target['station'] and stamp(row['timestamp_utc']) == stamp(target['target_time_utc']), 'selected original reference identity')
            for key in ('original_path','original_sha256','sidecar_path','sidecar_sha256'):
                require(row[key] == target[key], f'original source binding {key}')
            require(int(mask_rows[j].sum()) == int(row['valid_pixels']), 'original valid-pixel count')
        require(np.array_equal(saved['reference_COT'], reference_rows[saved['unique_index']]), 'all saved references equal selected original conversion')
        require(np.array_equal(saved['mask'], mask_rows[saved['unique_index']]), 'all saved masks equal original selected masks')
        first_by_id = {}
        for i, row_id in enumerate(saved['r_pack_index']):
            original = first_by_id.setdefault(int(row_id), i)
            for key in ('real_log1p','reference_COT','mask','station','target_time_utc'):
                require(np.array_equal(saved[key][i], saved[key][original]), f'unchanged duplicate real/reference/mask: {key}')
        require(len(first_by_id) == 58, '58 unique real references')
        audit['selection'] = dict(selection_sha256=SELECTION_SHA, targets_selected=64,
            matched_targets=58, matched_pairs=872, unavailable_leads=152, unmatched_targets=[t for t in selection if not t['pairs']],
            all_saved_keys_exact_inventory_match=True, repeated_real_reference_mask_exact_equal=True,
            selected_original_reference_and_mask_exact_equal=True,
            original_reference_conversion='multiply float32 target by float32(100), then float64',
            original_valid_pixels_unique=int(mask_rows.sum()), selected_reference_bytes=target_rows.nbytes + mask_rows.nbytes)

        def group(where):
            index = np.flatnonzero(where)
            return dict(n_pair=int(index.size), n_unique_target=int(np.unique(saved['r_pack_index'][index]).size),
                real=measure(saved['real_log1p'][index], saved['reference_COT'][index], saved['mask'][index]),
                forecast=measure(saved['predicted_log1p'][index], saved['reference_COT'][index], saved['mask'][index]))

        def compare(actual, recorded, label):
            if isinstance(actual, dict):
                for key in actual:
                    require(key in recorded, f'metric field absent: {label}.{key}')
                    compare(actual[key], recorded[key], label+'.'+key)
            elif isinstance(actual, (int, str)) or actual is None:
                require(actual == recorded, f'exact summary mismatch: {label}')
            else:
                require(isinstance(recorded, (float,int)), f'nonnumeric metric: {label}')
                difference = abs(actual-recorded)
                audit['metric_comparisons'].append(dict(key=label, absolute_difference=difference))
                require(np.isfinite(recorded) and difference <= ATOL, f'metric differs by {difference}: {label}')

        supplied = report['metrics']
        per_lead = []
        require(len(supplied['by_station_lead']) == 32, '32 original station/lead summaries')
        supplied_by_key = {(x['station'],x['lead_minutes']):x for x in supplied['by_station_lead']}
        require(len(supplied_by_key) == 32, 'no duplicate station/lead summaries')
        for station in STATIONS:
            for lead in LEADS:
                result = group((saved['station']==station) & (saved['lead_minutes']==lead))
                result.update(station=station,lead_minutes=lead,selected_targets=32,unavailable_targets=32-result['n_pair'])
                compare(result,supplied_by_key[(station,lead)],f'by_station_lead.{station}.{lead}')
                result['forecast_minus_real'] = {key:result['forecast'][key]-result['real'][key] for key in METRICS}
                per_lead.append(result)
        all_pairs = group(np.ones(872,dtype=bool))
        compare(all_pairs,supplied['all_pairs'],'all_pairs')
        by_station = []
        for station in STATIONS:
            result = dict(station=station, **group(saved['station']==station))
            result['forecast_minus_real'] = {key:result['forecast'][key]-result['real'][key] for key in METRICS}
            result['lead_direction_counts'] = {}
            station_leads = [r for r in per_lead if r['station']==station]
            for metric in ('cot_mae','cot_rmse','log1p_mae','log1p_rmse'):
                delta=np.array([r['forecast_minus_real'][metric] for r in station_leads])
                result['lead_direction_counts'][metric] = dict(forecast_lower=int((delta<0).sum()),forecast_higher=int((delta>0).sum()),equal=int((delta==0).sum()))
            by_station.append(result)
        first = np.array([first_by_id[key] for key in sorted(first_by_id)],dtype=np.int64)
        unique_targets = []
        require(len(supplied['unique_real_targets'])==58, '58 original unique metrics')
        supplied_unique = {r['r_pack_index']:r for r in supplied['unique_real_targets']}
        require(len(supplied_unique)==58,'unique report target identities')
        for row_id,index in zip(sorted(first_by_id),first):
            row = dict(r_pack_index=row_id,station=str(saved['station'][index]),target_time_utc=str(saved['target_time_utc'][index]),
                available_leads=int((saved['r_pack_index']==row_id).sum()),
                metrics=measure(saved['real_log1p'][index:index+1],saved['reference_COT'][index:index+1],saved['mask'][index:index+1]))
            compare(row,supplied_unique[row_id],f'unique_real_targets.{row_id}')
            unique_targets.append(row)
        unique_by_station=[]
        require(len(supplied['unique_real_by_station'])==2,'2 unique station summaries')
        station_unique = {r['station']:r for r in supplied['unique_real_by_station']}
        for station in STATIONS:
            take=first[saved['station'][first]==station]
            row=dict(station=station,n_unique_target=int(take.size),metrics=measure(saved['real_log1p'][take],saved['reference_COT'][take],saved['mask'][take]))
            compare(row,station_unique[station],f'unique_real_by_station.{station}')
            unique_by_station.append(row)
        unique_all=dict(n_unique_target=58,metrics=measure(saved['real_log1p'][first],saved['reference_COT'][first],saved['mask'][first]))
        compare(unique_all,supplied['unique_real_all'],'unique_real_all')
        require(sum(row['unavailable_targets'] for row in per_lead)==152,'all unavailable lead slots retained in metrics')
        require(sum(row['n_pair'] for row in per_lead)==872,'all pairs retained in metrics')
        require(sha(prediction_path)==report['predictions']['sha256'],'prediction file changed during audit')
        require((run/'report.json').read_bytes()==raw_report,'source report changed during audit')
        audit.update(state='PASS_INDEPENDENT_SAVED_COT_RESULTS_AUDIT',
            metric_comparison_count=len(audit['metric_comparisons']),
            max_metric_absolute_difference=max(r['absolute_difference'] for r in audit['metric_comparisons']),
            pair_weighted_by_station=by_station,by_station_lead=per_lead,all_pairs=all_pairs,
            unique_real_targets=unique_targets,unique_real_by_station=unique_by_station,unique_real_all=unique_all,
            metric_definition='physical expm1(float64 saved log1p) minus original selected COT; original mask only; pooled sum/count; bias prediction minus reference',
            pair_weighting='Each available target/lead contributes its original valid pixels. Compare pair-weighted real versus pair-weighted forecast; unique-real summary is descriptive separately.',
            limitations=['Does not rerun or prove model-to-prediction numerical binding; validates frozen provenance receipts and saved-array metrics.',
                'CPP source NC/generating-checkpoint linkage and independent physical truth remain unproven.',
                'Fixed validation selected R; repeated target/lead pairs are not independent generalization samples.',
                'Does not establish GHI impact or the sole cause of AC head behavior; old GPU-cache numeric warning remains open.'])
    except Exception as error:
        audit.update(state='FAIL_INDEPENDENT_COT_RESULTS_AUDIT',original_failure_type=type(error).__name__,original_failure=str(error))
        raise
    finally:
        audit['elapsed_seconds']=time.monotonic()-start
        with args.output.open('x') as stream:
            json.dump(audit,stream,indent=2,allow_nan=False)
        print(json.dumps({key:audit[key] for key in ('state','elapsed_seconds')},indent=2),flush=True)
    print(json.dumps(dict(metric_comparison_count=audit['metric_comparison_count'],max_metric_absolute_difference=audit['max_metric_absolute_difference'],pair_weighted_by_station=audit['pair_weighted_by_station']),indent=2),flush=True)


if __name__ == '__main__':
    main()
