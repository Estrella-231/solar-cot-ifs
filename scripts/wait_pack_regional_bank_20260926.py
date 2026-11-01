"""Verify incoming regional shards, then run real-AGRI pairing once complete."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    for key in ('cpp-bank', 'data-root', 'manifest', 'hunan-data-code', 'output', 'packer', 'status'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    lock = a.status.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    start = time.monotonic()
    verified = {}
    while time.monotonic() - start < 4 * 3600:
        index, accepted = a.cpp_bank / 'index.json', a.cpp_bank / 'acceptance_20260926.json'
        try:
            state = json.loads(index.read_text())
            acceptance = json.loads(accepted.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            atomic(a.status, {'stage': 'WAITING_TRANSFER_METADATA', 'test_used': False})
            time.sleep(20)
            continue
        assert state['state'] == 'COMPLETE_EXPLORATORY_CPP_BANK' and state['test_used'] is False
        assert acceptance['state'] == 'PASS_EXPLORATORY_CPP_BANK_INTEGRITY'
        assert acceptance['identity'] == state['identity']
        for item in state['shards']:
            path = a.cpp_bank / item['file']
            if not path.is_file():
                continue
            stat = path.stat()
            identity = (stat.st_size, stat.st_mtime_ns)
            if verified.get(path.name) == identity:
                continue
            verified.pop(path.name, None)
            if sha(path) == item['sha256'] and identity == (path.stat().st_size, path.stat().st_mtime_ns):
                verified[path.name] = identity
        atomic(a.status, {'stage': 'VERIFYING_TRANSFER', 'verified_shards': len(verified),
                          'total_shards': len(state['shards']), 'test_used': False})
        if len(verified) == len(state['shards']):
            break
        time.sleep(20)
    else:
        atomic(a.status, {'stage': 'TRANSFER_TIMEOUT', 'verified_shards': len(verified), 'test_used': False})
        raise RuntimeError('transfer not complete within four hours; no pairing started')
    atomic(a.status, {'stage': 'PACKING_REAL_AGRI_CPP', 'verified_shards': len(verified), 'test_used': False})
    spec = importlib.util.spec_from_file_location('regional_packer', a.packer)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.argv = [str(a.packer)]
    for key in ('cpp-bank', 'data-root', 'manifest', 'hunan-data-code', 'output'):
        sys.argv.extend(['--' + key, str(getattr(a, key.replace('-', '_')))])
    module.main()
    atomic(a.status, {'stage': 'PAIRED_CACHE_COMPLETE_REQUIRES_DISTRIBUTION_REVIEW',
                      'verified_shards': len(verified), 'test_used': False})


if __name__ == '__main__':
    main()
