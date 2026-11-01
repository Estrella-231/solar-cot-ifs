"""Resume the unchanged CPP builder with bounded HDF retries, then audit shards."""
import argparse
import collections
import csv
import fcntl
import hashlib
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def audit(bank, manifest):
    state = json.loads((bank / 'index.json').read_text())
    assert state['state'] == 'COMPLETE_EXPLORATORY_CPP_BANK'
    assert state['test_used'] is False
    assert sha(manifest) == state['identity']['manifest_sha256']
    missing = json.loads((bank / 'missing.json').read_text())
    expected = {'train': set(), 'val': set()}
    with manifest.open(newline='') as f:
        for row in csv.DictReader(f):
            if row['split'] in expected:
                expected[row['split']].update(Path(p).stem for p in row['data_relpaths'].split('|'))
    assert not expected['train'] & expected['val']
    actual = {'train': set(), 'val': set()}
    monthly = collections.defaultdict(lambda: {'frames': 0, 'valid_pixels': 0, 'total_pixels': 0,
                                               'cot_sum': 0., 'zero_pixels': 0, 'thick_ge30_pixels': 0})
    progress = []
    for item in state['shards']:
        path = bank / item['file']
        assert sha(path) == item['sha256'], ('SHA', path.name)
        with np.load(path, allow_pickle=False) as arr:
            cot, valid = arr['cot'], arr['valid']
            assert cot.shape == valid.shape == (item['frames'], 256, 256)
            assert valid.dtype == bool and np.isfinite(cot).all()
            assert np.all((cot[valid] >= 0) & (cot[valid] <= 100))
            for j, (stamp, split) in enumerate(zip(arr['utc'], arr['split'])):
                stamp, split = stamp.decode(), split.decode()
                assert split in actual and stamp in expected[split]
                assert stamp not in actual[split], ('duplicate', stamp)
                actual[split].add(stamp)
                key = split + '/' + stamp[:6]
                rec = monthly[key]
                values = cot[j][valid[j]]
                rec['frames'] += 1
                rec['valid_pixels'] += len(values)
                rec['total_pixels'] += 256 * 256
                rec['cot_sum'] += float(values.sum(dtype=np.float64))
                rec['zero_pixels'] += int((values == 0).sum())
                rec['thick_ge30_pixels'] += int((values >= 30).sum())
        progress.append(item['file'])
        atomic(bank / 'acceptance.progress.json', {'verified_shards': len(progress), 'total_shards': len(state['shards'])})
    for split in expected:
        absent = set(missing[split])
        assert not absent & actual[split]
        assert expected[split] == actual[split] | absent
    assert not actual['train'] & actual['val']
    for rec in monthly.values():
        n = rec['valid_pixels']
        rec['valid_fraction'] = n / rec['total_pixels']
        rec['mean_cot_valid'] = rec['cot_sum'] / n if n else None
        rec['zero_fraction_valid'] = rec['zero_pixels'] / n if n else None
        rec['thick_ge30_fraction_valid'] = rec['thick_ge30_pixels'] / n if n else None
    report = {'state': 'PASS_EXPLORATORY_CPP_BANK_INTEGRITY', 'test_used': False,
              'identity': state['identity'], 'verified_shards': len(progress),
              'splits': {s: {'available': len(actual[s]), 'missing': len(missing[s]),
                             'expected': len(expected[s]), 'coverage': len(actual[s]) / len(expected[s])}
                         for s in expected}, 'monthly': dict(sorted(monthly.items())),
              'limits': 'CPP-only integrity audit; daytime/AGRI pairing and producer/physical QA remain unverified'}
    atomic(bank / 'acceptance_20260926.json', report)
    print(json.dumps({'state': report['state'], 'shards': len(progress), 'splits': report['splits']}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--builder', type=Path, required=True)
    for name in ('manifest', 'grid', 'cpp-root', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    lock = (a.output / 'resume_audit_20260926.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    spec = importlib.util.spec_from_file_location('original_cpp_builder', a.builder)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.argv = [str(a.builder), '--manifest', str(a.manifest), '--grid', str(a.grid),
                '--cpp-root', str(a.cpp_root), '--output', str(a.output)]
    for attempt in range(1, 6):
        try:
            module.main()
            break
        except RuntimeError as error:
            traceback.print_exc()
            if 'HDF error' not in str(error) or attempt == 5:
                raise
            print(json.dumps({'action': 'retry_unchanged_builder_from_registered_shard',
                              'attempt': attempt, 'reason': str(error)}), flush=True)
            time.sleep(10)
    audit(a.output, a.manifest)


if __name__ == '__main__':
    main()
