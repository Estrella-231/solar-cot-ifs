"""Build a node-local regional R pack after independent source verification.

Never opens GHI labels, future sequences, test or buffer files. Fits input
moments on training supervised pixels, then applies the same transform to val.
"""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

CH = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]
NAMES = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C09', 'C10',
         'C11', 'C12', 'C13', 'C14', 'C15', 'cosSOZ', 'cosRAA', 'day_mask']


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--cpp-ready', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    source = json.loads((a.cpp_ready/'REPORT.json').read_text())
    audit = json.loads((a.cpp_ready/'INDEPENDENT_STATISTICS.json').read_text())
    assert source['state'] == 'PASS_REGIONAL_CPP_SOURCE_AND_REFERENCE'
    assert audit['state'] == 'PASS_INDEPENDENT_PACKAGE_AND_TRAIN_STATISTICS'
    rows = list(csv.DictReader(a.manifest.open()))
    assert rows and all(r['split'] in ('train', 'val') for r in rows)
    assert len({r['time_bjt'] for r in rows}) == len(rows) == source['frames'] == audit['frames']
    rows.sort(key=lambda r: (r['split'] != 'train', r['time_bjt']))
    assert {r['split'] for r in rows} == {'train', 'val'}
    records = {r['time_bjt']: r for r in json.loads((a.cpp_ready/'frames.json').read_text())}
    assert set(records) == {r['time_bjt'] for r in rows}
    a.out.mkdir(exist_ok=False, parents=True)
    n = len(rows)
    x = np.lib.format.open_memmap(a.out/'x_norm.npy', mode='w+', dtype='float32', shape=(n, 16, 256, 256))
    y = np.lib.format.open_memmap(a.out/'target_log1p.npy', mode='w+', dtype='float32', shape=(n, 1, 256, 256))
    m = np.lib.format.open_memmap(a.out/'mask.npy', mode='w+', dtype='bool', shape=(n, 1, 256, 256))
    sums = np.zeros(16, np.float64); squares = sums.copy(); count = 0
    support = []; excluded_ir = 0
    for i, row in enumerate(rows):
        key = row['time_bjt']; rec = records[key]
        assert rec['split'] == row['split']
        for path, digest in ((row['AGRI_path'], row['AGRI_sha256']), (row['aux_path'], row['aux_sha256'])):
            assert sha(path) == digest, (key, path)
        raw = np.load(row['AGRI_path'], mmap_mode='r', allow_pickle=False)
        assert raw.shape == (20, 256, 256)
        with np.load(row['aux_path'], allow_pickle=False) as z:
            geometry = np.stack([z[k] for k in ('cosSOZ', 'cosRAA', 'day_mask')])
        v = np.concatenate((raw[CH], geometry)).astype(np.float32)
        assert v.shape == (16, 256, 256) and np.isfinite(v).all()
        cpp = a.cpp_ready/'payload'/row['split']/(key+'.npz')
        assert sha(cpp) == rec['output_sha256']
        with np.load(cpp, allow_pickle=False) as z:
            cot = z['cot']; valid = z['mask'].copy()
        assert cot.shape == valid.shape == (256, 256) and valid.dtype == np.bool_
        spectral = (raw[8:15] > 0).all(axis=0)
        excluded_ir += int((valid & ~spectral).sum()); valid &= spectral
        assert np.all(np.isfinite(cot[valid]) & (cot[valid] >= 0) & (cot[valid] <= 100))
        x[i] = v; y[i, 0] = np.log1p(np.where(valid, cot, 0)); m[i, 0] = valid
        support.append(dict(time_bjt=key, split=row['split'], valid_pixels=int(valid.sum()),
                            spectral_invalid_pixels=int((~spectral).sum())))
        if row['split'] == 'train' and valid.any():
            vals = v[:, valid].astype(np.float64)
            count += vals.shape[1]; sums += vals.sum(axis=1); squares += (vals*vals).sum(axis=1)
        if (i+1) % 100 == 0:
            print('PACKED', i+1, n, flush=True)
    assert count > 0
    mean = sums/count; std = np.maximum(np.sqrt(np.maximum(squares/count-mean*mean, 0)), 1e-6)
    for i in range(n):
        x[i] = (x[i]-mean.astype(np.float32)[:, None, None])/std.astype(np.float32)[:, None, None]
        assert np.isfinite(x[i]).all()
    x.flush(); y.flush(); m.flush()
    del x, y, m
    with (a.out/'rows.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['index']+list(rows[0])); w.writeheader()
        w.writerows(dict(index=i, **r) for i, r in enumerate(rows))
    norm = dict(input_channels=NAMES, mean=mean.tolist(), std=std.tolist(),
                fit_split='train only', fit_pixels=count,
                fit_mask='CPP valid daylight SZA<80 and all selected thermal channels >0',
                target_transform='log1p(physical COT); no target standardization', test_used=False)
    (a.out/'norm.json').write_text(json.dumps(norm, indent=2))
    (a.out/'support.json').write_text(json.dumps(support))
    hashes = {p.name: sha(p) for p in a.out.iterdir() if p.is_file()}
    (a.out/'SHA256.json').write_text(json.dumps(hashes, indent=2))
    report = dict(state='REGIONAL_R_PACK_BUILT_NOT_TRAINED', frames=n,
                  counts={s: sum(r['split'] == s for r in rows) for s in ('train', 'val')},
                  valid_pixels={s: sum(r['valid_pixels'] for r in support if r['split'] == s) for s in ('train', 'val')},
                  excluded_thermal_invalid_supervision_pixels=excluded_ir,
                  manifest_sha256=sha(a.manifest), cpp_report_sha256=sha(a.cpp_ready/'REPORT.json'),
                  test_payloads_read=0, GHI_labels_read=0, trained=False)
    (a.out/'REPORT.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
