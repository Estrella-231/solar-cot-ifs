"""Metadata/header-only real/forecast COT pairing inventory; no model or test reads.

Reads only the completed R train/validation rows and forecast train/val indices.
It never imports torch, opens labels.csv, or accesses an array data element.
The fixed next-stage selection is 32 evenly spaced unique validation targets
per station, chosen from timestamps alone before inspecting CPP values. All
available leads are included and every unavailable lead is reported.
"""
import argparse
import bisect
import csv
import hashlib
import io
import json
import struct
import zipfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

BJT = timezone(timedelta(hours=8))
STATIONS = ('sili', 'zhujia')
LEADS = tuple(range(15, 241, 15))
TARGETS_PER_STATION = 32


def stamp(text):
    value = datetime.fromisoformat(text.replace('Z', '+00:00'))
    assert value.tzinfo is not None
    return value


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'inventory output must be new'
    root = args.root.resolve()
    pack = root / 'data/cot_repaired_pack_20260907_v1'
    cache = root / 'data/frozen_forecast_trainval_20260907_v1'
    accessed = []

    def read_metadata(path, kind='json'):
        assert 'test' not in path.name and path.name != 'labels.csv'
        raw = path.read_bytes()
        accessed.append(dict(path=str(path), mode='metadata', bytes=len(raw), sha256=digest(raw)))
        if kind == 'json':
            return json.loads(raw)
        reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
        return list(reader), reader.fieldnames

    def array_header(stream):
        start = stream.tell()
        version = np.lib.format.read_magic(stream)
        assert version in ((1, 0), (2, 0))
        fn = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = fn(stream)
        assert not fortran and not dtype.hasobject
        return dict(shape=list(shape), dtype=dtype.str, fortran_order=fortran,
                    npy_header_bytes=stream.tell()-start,
                    payload_bytes=int(np.prod(shape))*dtype.itemsize)

    def npy_header(path):
        with path.open('rb') as stream:
            spec = array_header(stream)
        accessed.append(dict(path=str(path), mode='npy_header_only', bytes=spec['npy_header_bytes']))
        spec['file_bytes'] = path.stat().st_size
        assert spec['file_bytes'] == spec['npy_header_bytes'] + spec['payload_bytes']
        return spec

    def npz_headers(path):
        # Parse ZIP_STORED member headers directly. ZipExtFile may prefetch
        # data, so it is deliberately not used for this metadata-only pass.
        members = {}
        with zipfile.ZipFile(path) as archive, path.open('rb') as stream:
            for member in archive.infolist():
                assert member.compress_type == zipfile.ZIP_STORED
                stream.seek(member.header_offset)
                header = stream.read(30)
                fields = struct.unpack('<4s5H3I2H', header)
                assert fields[0] == b'PK\x03\x04'
                stream.seek(fields[-2] + fields[-1], 1)
                spec = array_header(stream)
                spec.update(data_file_offset=stream.tell(), crc32=member.CRC,
                            compression='ZIP_STORED', member_bytes=member.file_size)
                assert member.file_size == spec['npy_header_bytes'] + spec['payload_bytes']
                members[member.filename] = spec
        accessed.append(dict(path=str(path), mode='zip_directory_and_npy_headers_only',
                             member_header_bytes=sum(x['npy_header_bytes'] for x in members.values()),
                             file_bytes=path.stat().st_size))
        return members

    status = read_metadata(pack / 'pack_status.json')
    assert status['state'] == 'COMPLETE' and status['test_payloads_read'] == 0
    hashes = read_metadata(pack / 'SHA256.json')  # Hash registry only, not test references.
    pack_config = read_metadata(pack / 'pack_config.json')
    norm = read_metadata(pack / 'norm.json')
    rows, row_schema = read_metadata(pack / 'rows.csv', 'csv')
    assert accessed[-1]['sha256'] == hashes['rows.csv']
    assert digest((pack / 'norm.json').read_bytes()) == hashes['norm.json']
    assert digest((pack / 'pack_config.json').read_bytes()) == hashes['pack_config.json']
    assert len(rows) == status['total'] == 18514
    headers = {name: npy_header(pack / (name + '.npy')) for name in ('x_raw', 'target', 'mask')}
    assert headers['x_raw']['shape'] == [len(rows), 16, 16, 16]
    assert headers['target']['shape'] == headers['mask']['shape'] == [len(rows), 1, 16, 16]
    assert headers['x_raw']['dtype'] == headers['target']['dtype'] == '<f4'
    assert headers['mask']['dtype'] == '|b1'
    contract = read_metadata(cache / 'contract.json')
    cache_status = read_metadata(cache / 'complete.json')
    assert cache_status['state'] == 'COMPLETE_FEATURE_CACHE' and not cache_status['test_used']
    sc = read_metadata(root / 'configs/s_frozen_hunan_seed42.json')
    rc = read_metadata(root / 'configs/r_frozen_repaired_seed42.json')
    assert sc == contract['S'] and rc == contract['R']
    assert tuple(sc['station_patches']) == STATIONS
    assert sc['lead_minutes'] == list(LEADS)
    assert norm['test_used'] is False and norm['cot_scale'] == 100
    assert norm['feature_names'] == [f'C{i:02}' for i in (1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15)] + ['cosSOZ', 'cosRAA', 'day_mask']

    # Register the sampling rule before any joins or selected shard headers.
    selection_rule = dict(split='validation', targets_per_station=TARGETS_PER_STATION,
        eligible='all unique validation station/targets in the completed R pack; forecast availability does not select targets',
        order='ascending timestamp_utc separately within each station',
        ranks='floor(k*(n-1)/31), k=0,...,31; no error/CPP-value/valid-pixel-count/lead-coverage ranking',
        includes='all available leads for each selected target, explicitly record unavailable leads; same R row/CPP mask reused across leads',
        purpose='single separately named CPP-reference input-propagation diagnostic; no GHI cohort change')
    rmap, rcounts, rminutes = {}, Counter(), Counter()
    for i, row in enumerate(rows):
        assert int(row['index']) == i and row['split'] in ('train', 'validation')
        assert row['station_id'] in STATIONS
        utc, bjt = stamp(row['timestamp_utc']), stamp(row['timestamp_bjt'])
        assert utc.utcoffset() == timedelta(0) and bjt.utcoffset() == timedelta(hours=8) and utc == bjt
        assert utc.minute in (0, 15, 30, 45) and utc.second == 0
        rminutes[(row['split'], row['station_id'], utc.minute)] += 1
        expected = 'train' if datetime(2024, 4, 2, tzinfo=BJT) <= bjt < datetime(2025, 7, 1, tzinfo=BJT) else 'validation' if datetime(2025, 7, 1, tzinfo=BJT) <= bjt < datetime(2025, 10, 1, tzinfo=BJT) else None
        assert row['split'] == expected
        assert datetime.strptime(Path(row['original_path']).stem, '%Y%m%d%H%M').replace(tzinfo=timezone.utc) == utc
        assert Path(row['original_path']).name == Path(row['sidecar_path']).name
        key = (row['split'], row['station_id'], utc)
        assert key not in rmap
        rmap[key] = row
        rcounts[(row['split'], row['station_id'])] += 1

    indices, completes, starts, seq_summary = {}, {}, {}, {}
    for disk_split, split in (('train', 'train'), ('val', 'validation')):
        complete = read_metadata(cache / disk_split / 'complete.json')
        idx, index_schema = read_metadata(cache / disk_split / 'index.csv', 'csv')
        assert accessed[-1]['sha256'] == complete['index_sha256']
        assert complete['state'] == 'COMPLETE' and complete['test_used'] is False
        assert complete['stations'] == list(STATIONS) and len(idx) == complete['samples']
        mapping, seen_seq = {}, set()
        for i, row in enumerate(idx):
            assert int(row['index']) == i and row['split'] == disk_split
            init = stamp(row['init_time_utc'])
            assert init.utcoffset() == timedelta(0) and init not in mapping
            seq_start = datetime.strptime(row['seq_id'], 'hunan_%Y%m%d%H%M').replace(tzinfo=timezone.utc)
            assert init == seq_start + timedelta(minutes=105)
            assert row['seq_id'] not in seen_seq
            seen_seq.add(row['seq_id'])
            mapping[init] = row
        indices[split], completes[split] = mapping, complete
        boundaries = [0]
        for shard in complete['shards']:
            boundaries.append(boundaries[-1] + shard['samples'])
        assert boundaries[-1] == len(idx)
        starts[split] = boundaries
        seq_summary[split] = dict(sequences=len(idx), shards=len(complete['shards']), index_schema=index_schema,
            init_utc_min=min(mapping).isoformat(), init_utc_max=max(mapping).isoformat(),
            total_station_lead_candidates=len(idx)*2*16, index_sha256=complete['index_sha256'])

    overlap, eligible, all_pairs_keys = [], {}, set()
    for split in ('train', 'validation'):
        for station in STATIONS:
            targets = sorted(t for sp, st, t in rmap if (sp, st) == (split, station))
            per_lead = []
            covered = {t: [] for t in targets}
            for lead in LEADS:
                matches = [t for t in targets if t-timedelta(minutes=lead) in indices[split]]
                for target in matches:
                    init = target-timedelta(minutes=lead)
                    key = (split, station, init, target)
                    assert key not in all_pairs_keys
                    all_pairs_keys.add(key)
                    covered[target].append(lead)
                per_lead.append(dict(lead_minutes=lead, matched_pairs=len(matches),
                                     unmatched_R_targets=len(targets)-len(matches)))
            full = [t for t, ls in covered.items() if len(ls) == 16]
            eligible[(split, station)] = full
            overlap.append(dict(split=split, station=station, R_targets=len(targets),
                target_utc_min=targets[0].isoformat(), target_utc_max=targets[-1].isoformat(),
                any_lead_targets=sum(bool(ls) for ls in covered.values()), all_16_lead_targets=len(full),
                no_lead_targets=sum(not ls for ls in covered.values()),
                matched_pairs=sum(x['matched_pairs'] for x in per_lead), by_lead=per_lead))

    selection, shard_specs = [], {}
    for station_index, station in enumerate(STATIONS):
        targets = sorted(t for sp, st, t in rmap if (sp, st) == ('validation', station))
        assert len(targets) >= TARGETS_PER_STATION
        ranks = [k*(len(targets)-1)//(TARGETS_PER_STATION-1) for k in range(TARGETS_PER_STATION)]
        for rank in ranks:
            target = targets[rank]
            rrow = rmap[('validation', station, target)]
            pairs, missing_leads = [], []
            for lead_index, lead in enumerate(LEADS):
                init = target-timedelta(minutes=lead)
                if init not in indices['validation']:
                    missing_leads.append(dict(lead_minutes=lead, expected_init_time_utc=init.isoformat(),
                                              reason='init absent from completed validation forecast index'))
                    continue
                seq = indices['validation'][init]
                cache_index = int(seq['index'])
                shardpos = bisect.bisect_right(starts['validation'], cache_index)-1
                shard = completes['validation']['shards'][shardpos]
                path = cache/'val'/shard['name']
                shard_specs.setdefault(str(path), dict(path=str(path), recorded_sha256=shard['sha256'],
                                                       samples=shard['samples'], file_bytes=path.stat().st_size))
                pairs.append(dict(lead_minutes=lead, lead_index=lead_index, init_time_utc=init.isoformat(),
                    init_time_bjt=init.astimezone(BJT).isoformat(), seq_id=seq['seq_id'], cache_index=cache_index,
                    shard=shard['name'], shard_row=cache_index-starts['validation'][shardpos]))
            selection.append(dict(station=station, station_index=station_index, rank_0based=rank,
                eligible_target_count=len(targets), r_pack_index=int(rrow['index']),
                target_time_utc=target.isoformat(), target_time_bjt=target.astimezone(BJT).isoformat(),
                original_path=rrow['original_path'], original_sha256=rrow['original_sha256'],
                sidecar_path=rrow['sidecar_path'], sidecar_sha256=rrow['sidecar_sha256'],
                pairs=pairs, missing_leads=missing_leads))
    # Header-only schema checks of the selected validation shards (not payload).
    for spec in shard_specs.values():
        spec['members'] = npz_headers(Path(spec['path']))
        b = spec['samples']
        expected = {'indices.npy': [b], 'predicted_agri_physical.npy': [b, 2, 16, 13, 16, 16],
                    'geometry.npy': [b, 2, 16, 3, 16, 16], 'cot_log1p.npy': [b, 2, 16, 16, 16],
                    'cot_features.npy': [b, 2, 16, 4], 'r_latent_mean.npy': [b, 2, 16, 16]}
        assert set(spec['members']) == set(expected)
        for name, shape in expected.items():
            assert spec['members'][name]['shape'] == shape

    selected_targets = len(selection)
    selected_pairs = sum(len(x['pairs']) for x in selection)
    selection_sha = digest(json.dumps(dict(rule=selection_rule, targets=selection), sort_keys=True, separators=(',', ':')).encode())
    report = dict(state='COMPLETE_METADATA_INVENTORY_WITH_COORDINATE_BINDING_GATE',
        created_utc=datetime.now(timezone.utc).isoformat(), root=str(root),
        test_rows_read=0, test_payloads_read=0, GHI_labels_or_performance_read=False,
        tensor_payload_bytes_read=0, models_loaded=0, training_or_GPU_used=False,
        script_sha256=digest(Path(__file__).read_bytes()),
        R_pack=dict(path=str(pack), status=status, rows_schema=row_schema, arrays=headers,
            recorded_array_sha256={n: hashes[n] for n in ('x_raw.npy', 'target.npy', 'mask.npy')},
            rows_sha256=hashes['rows.csv'], norm_sha256=hashes['norm.json'], config=pack_config,
            input_features=norm['feature_names'], target='physical COT = target.npy * 100; use only mask.npy true pixels',
            mask='original repaired sidecar cot_reference_mask; numerical finite COT in [0,100], day > .5, solar zenith < 80 deg',
            payload_hash_status='recorded prior completed-pack SHA only; not reread arrays in this inventory'),
        forecast=dict(path=str(cache), contract_sha256=next(x['sha256'] for x in accessed if x['path']==str(cache/'contract.json')),
            S_checkpoint= sc['checkpoint'], S_checkpoint_sha256=sc['checkpoint_sha256'],
            R_checkpoint=rc['checkpoint'], R_checkpoint_sha256=rc['checkpoint_sha256'],
            station_order=list(STATIONS), station_patches=sc['station_patches'], grid_sha256=sc['grid_sha256'],
            center_pixel=sc['center_pixel'], sequence_summary=seq_summary,
            mapping='index.csv index is split-local cache_index; seq_id UTC start +105 min = canonical init UTC; target=init+15*(lead_index+1); station axis follows station_order',
            payload_hash_status='selected shard receipts copied; no complete shard SHA reread'),
        checks=dict(R_station_target_unique=True, R_rows_index_contiguous=True, R_BJT_UTC_equal=True,
            R_filename_UTC_equal=True, R_time_split_verified=True,
            R_minute_counts=[dict(split=sp, station=st, minute=minute, rows=n)
                             for (sp, st, minute), n in sorted(rminutes.items())],
            forecast_split_index_contiguous=True, forecast_init_and_seq_unique=True,
            forecast_seq_to_init_offset_minutes=105, pair_keys_unique=True),
        overlap=overlap, total_matched_pairs=len(all_pairs_keys),
        selection_rule=selection_rule, selection_sha256=selection_sha, selection=selection,
        selected_shards=list(shard_specs.values()),
        proposed_read_budget=dict(targets=selected_targets, pairs=selected_pairs,
            R_unique_rows_data_bytes=selected_targets*(16*16*16*4+16*16*4+16*16),
            forecast_selected_patch_data_bytes=selected_pairs*(16*16*16*4+16*16*4),
            selected_shards=len(shard_specs),
            full_selected_shard_SHA_stream_bytes=sum(x['file_bytes'] for x in shard_specs.values()),
            future_read_strategy='SHA-stream only selected uncompressed shards against receipts, then seek directly to selected contiguous station/lead patches using checked ZIP_STORED NPY offsets; memmap only selected R rows; no full grids',
            extra_coordinate_reads='before numerical comparison, read only grid_lat/grid_lon and scalar identity from selected original/sidecar NPZ members on FD-107, and matching static-grid station patches; no AGRI/GHI/CPP field access for this coordinate gate'),
        unresolved=['R pack rows/arrays omit per-row grid_lat/grid_lon and geometry-source provenance. Prior source/sidecar grid equality checks are recorded in pack code, but selected real-vs-forecast coordinate identity must be verified from coordinate-only original/sidecar members and frozen static-grid patches before interpreting errors.',
            'Stored real-input geometry and forecast deterministic target geometry must be compared separately. If different, do not attribute the whole R difference to AGRI; record the mismatch and stop or predeclare a separate common-geometry diagnostic.',
            'R backend numerical witness is an external prerequisite; this inventory does not pass it.',
            'CPP source NC batch-to-generating-checkpoint binding remains unverified; CPP is retrieval reference, not independent truth.',
            'The R references retain prior CPP availability/mask selection; this diagnostic cannot define or replace the 15-minute GHI cohort.',
            'Validation selected R itself; a same-validation retrieval diagnostic is descriptive, not new generalization evidence.'],
        metadata_accessed=accessed)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(state=report['state'], overlap=overlap, selection_sha256=selection_sha,
        proposed_read_budget=report['proposed_read_budget']), indent=2))


if __name__ == '__main__':
    main()
