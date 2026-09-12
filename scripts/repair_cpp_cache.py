"""Build immutable CPP sidecars for ALL existing Combined samples.

Read-only original AGRI/GHI; independent output. JSONL ledger resumes failures
without treating a truncated output as success. No GHI-based selection.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import netCDF4 as nc
import numpy as np
from cpp_grid_mapping import map_cpp_grid
from cpp_quality_mask import cot_reference_mask
from repartition_cot_manifest import paper_split

CHANNELS = ('COT', 'CER', 'CTH', 'CLP')


def digest(b):
    return hashlib.sha256(b).hexdigest()


def atomic_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, path)


def repair_one(task):
    original, relative, cpp_root, output, contract_sha = task
    row = {'original_path': original, 'relative_path': relative}
    try:
        raw = Path(original).read_bytes()
        with np.load(io.BytesIO(raw), allow_pickle=False) as z:
            stamp = str(z['timestamp_utc'].item())
            station = str(z['station_id'].item())
            lat, lon = z['grid_lat'].copy(), z['grid_lon'].copy()
            agri = z['agri']
            day, soz = agri[19].copy(), agri[18].copy()
        dt = datetime.fromisoformat(stamp.replace('Z', '+00:00'))
        split, bjt = paper_split(stamp)
        if station != Path(relative).parts[0] or Path(original).stem != dt.strftime('%Y%m%d%H%M'):
            raise ValueError('station or UTC filename disagrees with embedded timestamp')
        if lat.shape != (16, 16) or agri.shape != (20, 16, 16):
            raise ValueError('unexpected Combined geometry/channel shape')
        src = Path(cpp_root)/dt.strftime('%Y/%Y%m%d/FY4B_AGRI_%Y%m%d%H%M%S.nc')
        row.update(station_id=station, timestamp_utc=stamp, timestamp_bjt=bjt,
                   split=split or 'outside_paper_dates', source_nc=str(src),
                   original_sha256=digest(raw))
        present = src.is_file()
        cpp = np.full((4, 16, 16), np.nan, dtype=np.float32)
        rows = cols = np.array([], dtype=np.int64)
        err = {}
        metadata = {}
        if present:
            before = src.stat()
            with nc.Dataset(src) as ds:
                if str(ds['COT'].getncattr('Range')).strip() != '0 - 100':
                    raise ValueError('unrecognized COT Range; requires contract review')
                qa = [n for n in ds.variables if any(s in n.lower() for s in
                      ('quality', 'flag', 'uncert', 'confidence', 'dqf'))]
                if qa:
                    raise ValueError('new QA fields require explicit decoding: '+str(qa))
                rows, cols, err = map_cpp_grid(np.ma.filled(ds['LAT'][:], np.nan),
                    np.ma.filled(ds['LON'][:], np.nan), lat, lon)
                cpp = np.stack([np.ma.filled(ds[n][rows, cols], np.nan).astype(np.float32)
                                for n in CHANNELS])
                metadata = {n: {k: str(ds[n].getncattr(k)) for k in ds[n].ncattrs()}
                            for n in CHANNELS}
            after = src.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('source NC changed during read')
            row.update(source_bytes=after.st_size, source_mtime_ns=after.st_mtime_ns, **err)
        finite = np.isfinite(cpp)
        valid, reasons = cot_reference_mask(cpp[0], finite[0], day, soz)
        dst = Path(output)/'samples'/relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        np.savez_compressed(buf, cpp=cpp, cpp_valid_mask=finite, cpp_present=np.bool_(present),
            cot_reference_mask=valid, cpp_rows=rows, cpp_cols=cols,
            grid_lat=lat, grid_lon=lon, station_id=station, timestamp_utc=stamp,
            original_path=original, original_sha256=row['original_sha256'],
            source_nc=str(src), contract_sha256=contract_sha,
            channel_names=np.asarray(CHANNELS), source_metadata_json=json.dumps(metadata),
            schema_version='cpp_aligned_sidecar_v1')
        result = buf.getvalue()
        tmp = dst.with_suffix('.npz.tmp')
        tmp.write_bytes(result)
        os.replace(tmp, dst)
        # Read back every output, not just a write-success count.
        check = dst.read_bytes()
        if digest(check) != digest(result):
            raise ValueError('output readback hash mismatch')
        with np.load(io.BytesIO(check), allow_pickle=False) as z:
            if not np.array_equal(z['cpp'], cpp, equal_nan=True):
                raise ValueError('output readback values mismatch')
            if not np.array_equal(z['cot_reference_mask'], valid):
                raise ValueError('output readback mask mismatch')
        row.update(status='ok' if present else 'missing_source', sidecar_path=str(dst),
                   sidecar_sha256=digest(result), valid_pixels=int(valid.sum()),
                   cot_gt85_le100_valid_pixels=int((valid & (cpp[0]>85)).sum()),
                   exclusions={k: int(v.sum()) for k,v in reasons.items()},
                   cot_candidate=bool(split and valid.any()))
    except Exception as exc:
        row.update(status='error', error=repr(exc))
    return row


def read_ledger(path):
    records = {}
    if not path.exists():
        return records
    data = path.read_bytes()
    complete = data.rfind(b'\n')+1
    for line in data[:complete].splitlines():
        r = json.loads(line)
        records[r['relative_path']] = r
    if complete != len(data):
        # Only discard a crash-truncated final record, inside this run output.
        with path.open('r+b') as f:
            f.truncate(complete)
    return records


def export(records, output, total, started, state):
    counts, candidates = Counter(), Counter()
    fields = ['station_id','timestamp_utc','timestamp_bjt','split','original_path',
              'sidecar_path','original_sha256','sidecar_sha256','status','valid_pixels']
    with (output/'inventory_status.jsonl').open('w', encoding='utf-8') as log, \
         (output/'cot_candidates.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fields, extrasaction='ignore')
        writer.writeheader()
        for key in sorted(records):
            r = records[key]
            log.write(json.dumps(r)+'\n')
            counts[r['status']] += 1
            if r.get('cot_candidate'):
                candidates[r['split']+'/'+r['station_id']] += 1
                writer.writerow(r)
    result = dict(state=state, total_inventory=total, processed=sum(counts.values()),
        counts=dict(counts), cot_candidates=dict(candidates), elapsed_seconds=time.time()-started,
        scope='all existing Combined files; not missing AGRI reconstruction',
        original_files_modified=False, upstream_teacher_provenance_verified=False)
    atomic_json(output/'status.json', result)
    print(json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--combined-root', type=Path, required=True)
    p.add_argument('--cpp-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--contract', type=Path, required=True)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--limit', type=int)
    p.add_argument('--inventory-source', type=Path, help='explicit relative paths for bounded smoke only')
    p.add_argument('--retry-missing', action='store_true', help='retry previously absent source files')
    a = p.parse_args()
    original, output = a.combined_root.resolve(), a.output.resolve()
    if original == output or original in output.parents or output in original.parents:
        raise ValueError('output must be independent of original tree')
    output.mkdir(parents=True, exist_ok=True)
    # OS lock is released on exit/crash; excludes concurrent writers on resume.
    import fcntl
    lock = (output/'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    contract_sha = digest(a.contract.read_bytes())
    run_config = dict(combined_root=str(original), cpp_root=str(a.cpp_root.resolve()),
                      contract_sha256=contract_sha, limit=a.limit,
                      inventory_source_sha256=digest(a.inventory_source.read_bytes()) if a.inventory_source else None,
                      code_sha256={n: digest(Path(__file__).with_name(n).read_bytes()) for n in
                       ('repair_cpp_cache.py','cpp_grid_mapping.py','cpp_quality_mask.py','repartition_cot_manifest.py')})
    config = output/'run_config.json'
    if config.exists() and json.loads(config.read_text()) != run_config:
        raise ValueError('resume config/code differs; use a new output directory')
    atomic_json(config, run_config)
    started = time.time()
    inventory = output/'inventory.txt'
    if not inventory.exists():
        paths = []
        for station in (() if a.inventory_source else ('sili','zhujia')):
            for year in sorted((original/station).glob('[0-9][0-9][0-9][0-9]')):
                for month in sorted(year.glob('[0-9]'*6)):
                    found = sorted(month.glob('[0-9]'*8+'/*.npz'))
                    paths.extend(str(x.relative_to(original)) for x in found)
                    print('INVENTORY', station, month.name, len(found), flush=True)
        if a.inventory_source:
            paths = a.inventory_source.read_text().splitlines()
        if any(Path(r).is_absolute() or '..' in Path(r).parts for r in paths):
            raise ValueError('inventory paths must remain inside Combined root')
        if len(paths) != len(set(paths)) or not paths:
            raise ValueError('empty or duplicate inventory')
        if a.limit:
            # Evenly spaced across inventory rather than first night only.
            indices = np.linspace(0,len(paths)-1,min(a.limit,len(paths)),dtype=int)
            paths = [paths[i] for i in indices]
        tmp = inventory.with_suffix('.tmp')
        tmp.write_text('\n'.join(paths)+'\n')
        os.replace(tmp, inventory)
    paths = inventory.read_text().splitlines()
    ledger_path = output/'ledger.jsonl'
    records = read_ledger(ledger_path)
    accepted = ('ok',) if a.retry_missing else ('ok','missing_source')
    done = {path for path,row in records.items() if row['status'] in accepted}
    tasks = [(str(original/rel),rel,str(a.cpp_root),str(output),contract_sha)
             for rel in paths if rel not in done]
    export(records,output,len(paths),started,'RUNNING')
    with ledger_path.open('a', encoding='utf-8') as ledger, ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,row in enumerate(pool.map(repair_one,tasks,chunksize=1),1):
            ledger.write(json.dumps(row)+'\n')
            records[row['relative_path']] = row
            if i % 25 == 0 or i == 1:
                ledger.flush()
                os.fsync(ledger.fileno())
            if i % 100 == 0 or i == 1:
                export(records,output,len(paths),started,'RUNNING')
        ledger.flush()
        os.fsync(ledger.fileno())
    result = export(records,output,len(paths),started,'FINISHED')
    if result['counts'].get('error',0):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
