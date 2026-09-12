"""Audit M0 identities/labels and freeze base GHI keys without looking at future CPP.

This is a base cohort, not a release-causal IFS intersection or a ready kt dataset.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--m0', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    checks = {}
    for line in (a.m0/'SHA256SUMS').read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.lstrip('*')
        actual = sha(a.m0/name)
        assert actual == expected, name
        checks[name] = actual
    summary = json.loads((a.m0/'audit_summary.json').read_text())
    assert summary['status'] == 'PASS'
    assert summary['source_audit']['pass'] and summary['spatial_audit']['status'] == 'PASS'
    sequences = {}
    with (a.m0/'sequence_manifest_8to16.csv').open(newline='') as f:
        for r in csv.DictReader(f):
            assert r['sequence_id'] not in sequences
            sequences[r['sequence_id']] = (r['split'], r['init_time_bjt'])
    keys = set()
    counts = Counter()
    source_rows = 0
    columns = ['sequence_id', 'split', 'station', 'latitude', 'longitude', 'init_time_bjt',
               'init_time_utc', 'target_time_bjt', 'target_time_utc', 'lead_minutes',
               'observed_ghi_15min_wm2', 'label_record_times_bjt', 'solar_cosine',
               'day_mask_sza_lt_90']
    with (a.output/'base_cohort.csv').open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        with (a.m0/'canonical_station_target_manifest.csv').open(newline='') as f:
            for r in csv.DictReader(f):
                source_rows += 1
                assert sequences[r['sequence_id']] == (r['split'], r['init_time_bjt'])
                init = datetime.fromisoformat(r['init_time_bjt'])
                target = datetime.fromisoformat(r['target_time_bjt'])
                lead = int(r['lead_minutes'])
                assert lead in range(15, 241, 15)
                assert target-init == timedelta(minutes=lead)
                assert init == datetime.fromisoformat(r['init_time_utc'])
                assert target == datetime.fromisoformat(r['target_time_utc'])
                record_times = [datetime.fromisoformat(t) for t in r['label_record_times_bjt'].split('|')]
                assert record_times == [target-timedelta(minutes=k) for k in (10, 5, 0)]
                key = (r['station'], r['init_time_bjt'], r['target_time_bjt'])
                assert key not in keys
                keys.add(key)
                valid = r['label_valid'] == 'True' and r['day_mask_sza_lt_90'] == 'True'
                assert valid == (r['valid_daylight'] == 'True')
                if not valid:
                    continue
                label = float(r['observed_ghi_15min_wm2'])
                raw = [float(v) for v in r['raw_ghi_5min_wm2'].split('|')]
                assert len(raw) == 3 and all(math.isfinite(v) and v >= 0 for v in raw)
                assert abs(sum(raw)/3-label) < 1e-4
                assert r['station'] in ('sili', 'zhujia')
                writer.writerow({k:r[k] for k in columns})
                counts[(r['split'], r['station'], lead)] += 1
    assert sum(counts.values()) == summary['manifest']['valid_daylight_targets']
    with (a.output/'counts.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['split', 'station', 'lead_minutes', 'count'])
        w.writerows((*k,v) for k,v in sorted(counts.items()))
    audit = dict(state='BASE_COHORT_VERIFIED', source_rows=source_rows, base_rows=sum(counts.values()),
                 source_m0=str(a.m0), source_hashes=checks, source_code_sha256=sha(Path(__file__)),
                 output_sha256=sha(a.output/'base_cohort.csv'),
                 keys='station,init_time,target_time', labels='M0 authoritative 15min end intervals',
                 future_CPP_used_for_filtering=False, source_payloads_opened=False,
                 test_metrics_computed=False, clear_sky_and_kt='PENDING',
                 forecast_AGRI_cache='PENDING_FROZEN_SIMVP', IFS_intersection='PENDING_RELEASE_CONTRACT')
    (a.output/'audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
