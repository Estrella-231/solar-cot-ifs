"""Freeze date-grouped internal R fit/select roles from outer-train keys."""
import argparse, csv, hashlib, json
from datetime import datetime, timedelta
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); a = ap.parse_args()
    assert not a.out.exists()
    rows = list(csv.DictReader(a.manifest.open()))
    assert all(r['split'] in ('train', 'val') for r in rows)
    train_dates = sorted({r['time_bjt'][:8] for r in rows if r['split'] == 'train'})
    selected = []
    for month in sorted({d[:6] for d in train_dates}):
        dd = [d for d in train_dates if d.startswith(month)]
        if len(dd) < 5:
            continue
        dd.sort(key=lambda d: hashlib.sha256(('regional_R_seed42_'+d).encode()).hexdigest())
        selected.extend(dd[:max(1, len(dd)//7)])
    selected = set(selected); purged = set()
    for d in selected:
        dt = datetime.strptime(d, '%Y%m%d')
        purged.update((dt+timedelta(days=k)).strftime('%Y%m%d') for k in (-1, 1))
    purged = purged.intersection(train_dates)-selected
    fit = set(train_dates)-selected-purged
    assert fit and selected and not (fit & (selected | purged))
    mapped = []
    for r in rows:
        d = r['time_bjt'][:8]
        role = 'monthly_val' if r['split'] == 'val' else ('R_select' if d in selected else 'purged' if d in purged else 'R_fit')
        mapped.append(dict(time_bjt=r['time_bjt'], outer_split=r['split'], role=role))
    result = dict(state='FROZEN_R_INTERNAL_DATE_ROLES', rule='per-month train-only SHA256 order; floor(n_dates/7), min1 if>=5; +/-1 day purge',
                  seed=42, fit_dates=sorted(fit), select_dates=sorted(selected), purged_dates=sorted(purged),
                  row_counts={k: sum(r['role'] == k for r in mapped) for k in ('R_fit', 'R_select', 'purged', 'monthly_val')},
                  roles=mapped, source_manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
                  GHI_labels_read=0, CPP_payloads_read=0, test_payloads_read=0,
                  input_norm_scope='outer train supported pixels, including unsupervised internal select inputs; no val')
    a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'roles'}), flush=True)


if __name__ == '__main__':
    main()
