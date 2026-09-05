"""Read-only bounded P200 inventory. No training, no test score evaluation.

Source paths are explicit; only immediate directory entries are listed.
Writes a new output directory and refuses to replace an earlier audit.
"""
import argparse
import csv
import hashlib
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def inspect(path):
    p = Path(path)
    result = {'path': str(p), 'exists': p.exists()}
    if not p.exists():
        return result
    if p.is_dir():
        result['entries'] = sorted(x.name for x in p.iterdir())[:120]
        result['listing_scope'] = 'first 120 sorted immediate children; not exhaustive recursive search'
    else:
        result['bytes'] = p.stat().st_size
        if p.suffix in {'.json', '.csv', '.md', '.py'} and p.stat().st_size < 2_000_000:
            result['sha256'] = digest(p)
            if p.suffix == '.json':
                result['content'] = json.loads(p.read_text())
            elif p.suffix == '.csv':
                with p.open(newline='') as f:
                    reader = csv.DictReader(f)
                    result['columns'] = reader.fieldnames
                    result['first_rows'] = [row for _, row in zip(range(3), reader)]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path('/home/Data_Pool_3')
    m0 = root / 'wangyc/irradiance_paper/hunan_shortterm_m0_20260903_r101_r103_v1'
    aux = root / 'chenyi/Auxiliary_data'
    ghi = root / 'chenyi/GHI'
    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'hostname': platform.node(), 'scope': 'bounded asset inventory; no experiment acceptance',
              'assets': {}, 'errors': {}}
    targets = {
        'm0_summary': m0/'audit_summary.json',
        'm0_spatial': m0/'spatial_audit.json',
        'station_schema': aux/'HuNan_Station16_Combined/schema.json',
        'agri_meta': aux/'HuNan_AGRI_Preprocessed/dataset_meta.json',
        'agri_channels': aux/'HuNan_AGRI_Preprocessed/channel_info.json',
        'cot_cache': aux/'HuNan_Solar_Ablation_COT_Cache_20260808/READY.json',
        'cot_cache_manifest': aux/'HuNan_Solar_Ablation_COT_Cache_20260808/manifest.csv',
        'cot_root': ghi/'cloud_retrieval_hunan',
        'cot_run': ghi/'cloud_retrieval_hunan/results/cot_unet_agri13_geom_v1',
        'simvp_runs': ghi/'SimVPv2/work_dirs',
        'ifs_station_summary': aux/'HuNan_IFS_Station16/summary.json',
        'ifs_station_channels': aux/'HuNan_IFS_Station16/channel_info.json',
        'ifs_station_index': aux/'HuNan_IFS_Station16/index.csv',
        'ifs_raw_apr2024': root/'chensr/ifs_hres_china/raw_nc/2024/202404',
    }
    for name, path in targets.items():
        print('INSPECT', name, str(path), flush=True)
        try:
            report['assets'][name] = inspect(path)
        except Exception as e:
            report['errors'][name] = repr(e)
        (args.output/'inventory.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print('VERIFY_M0_SHA', flush=True)
    checks = []
    for line in (m0/'SHA256SUMS').read_text().splitlines():
        expected, rel = line.split(maxsplit=1)
        p = m0/rel.lstrip('*')
        actual = digest(p)
        checks.append({'path': str(p), 'expected': expected, 'actual': actual, 'match': expected == actual})
    report['m0_sha_checks'] = checks
    with (m0/'sequence_manifest_8to16.csv').open(newline='') as f:
        reader = csv.DictReader(f)
        rows = Counter()
        examples = {}
        for row in reader:
            split = row['split']
            rows[split] += 1
            examples.setdefault(split, row)
        report['m0_sequences'] = {'counts': dict(rows), 'examples': examples, 'columns': reader.fieldnames}

    print('INSPECT_RAW_IFS', flush=True)
    try:
        import xarray as xr
        p = root/'chensr/ifs_hres_china/raw_nc/2024/202404/ecmwf_hres_hubei_hunan_20240402_t0000.nc'
        with xr.open_dataset(p, decode_times=False) as ds:
            report['ifs_sample'] = {'path': str(p), 'attrs': dict(ds.attrs),
                'dimensions': dict(ds.sizes),
                'variables': {k: {'dims': list(v.dims), 'attrs': dict(v.attrs)} for k, v in ds.variables.items()},
                'time_values': {k: ds[k].values.ravel()[:25].tolist() for k in ds.coords
                                if any(x in k.lower() for x in ('time', 'step'))}}
    except Exception as e:
        report['errors']['ifs_sample'] = repr(e)
    report['m0_integrity'] = 'PASS' if checks and all(c['match'] for c in checks) else 'FAIL'
    report['training_gate'] = 'NOT_EVALUATED_inventory_only'
    (args.output/'inventory.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print('AUDIT_SAVED', args.output, report['m0_integrity'], flush=True)


if __name__ == '__main__':
    main()
