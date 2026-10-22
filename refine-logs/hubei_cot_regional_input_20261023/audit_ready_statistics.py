"""Independent regional CPP package check and train-only log-COT moments.

This prepares data statistics; it neither trains R nor scores GHI.
"""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ready', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    assert not a.out.exists(), 'Preserve completed audit outputs'
    report = json.loads((a.ready/'REPORT.json').read_text())
    assert report['state'] == 'PASS_REGIONAL_CPP_SOURCE_AND_REFERENCE'
    rows = list(csv.DictReader(a.manifest.open()))
    expected = {r['time_bjt']: r for r in rows}
    records = json.loads((a.ready/'frames.json').read_text())
    assert len(expected) == len(rows) == len(records)
    assert {r['time_bjt'] for r in records} == set(expected)
    count = 0
    total = total2 = 0.0
    coverage = {}
    for r in records:
        key = r['time_bjt']; split = r['split']
        assert split in ('train', 'val') and expected[key]['split'] == split
        p = a.ready/'payload'/split/(key+'.npz')
        assert hashlib.sha256(p.read_bytes()).hexdigest() == r['output_sha256']
        with np.load(p, allow_pickle=False) as z:
            cot, clp, mask = z['cot'], z['clp'], z['mask']
            assert cot.shape == clp.shape == mask.shape == (256, 256)
            assert mask.dtype == np.bool_
            assert np.all(np.isfinite(cot[mask]) & (cot[mask] >= 0) & (cot[mask] <= 100))
            assert np.all(np.isin(clp[mask], [0, 1, 2]))
            assert int(mask.sum()) == r['valid_pixels']
            group = coverage.setdefault(split+'/'+key[:6], dict(frames=0, valid_pixels=0, clear=0, water=0, ice=0))
            group['frames'] += 1; group['valid_pixels'] += int(mask.sum())
            for v, name in enumerate(('clear', 'water', 'ice')):
                group[name] += int(((clp == v) & mask).sum())
            if split == 'train':
                x = np.log1p(cot[mask].astype(np.float64))
                count += x.size; total += float(x.sum()); total2 += float(np.dot(x, x))
    assert count > 0
    mean = total/count; std = float(np.sqrt(max(0., total2/count-mean*mean)))
    assert std > 0
    result = dict(state='PASS_INDEPENDENT_PACKAGE_AND_TRAIN_STATISTICS', frames=len(rows),
                  train_log1p_cot=dict(count=count, mean=mean, population_std=std,
                                       fitting_scope='train valid pixels only; not yet an adopted R input normalization'),
                  coverage=coverage, test_payloads_read=0, GHI_labels_read=0,
                  training_started=False, validation_used_for_statistics=False)
    a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
