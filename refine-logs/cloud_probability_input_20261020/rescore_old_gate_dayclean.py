"""Read-only FP64 rescore of saved raw/gate predictions under the current GHI QC rule."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from station_day_quality import ghi_day_keep_mask, POLICY

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def metric(y, p):
    e = p.astype(np.float64) - y.astype(np.float64)
    return dict(n=len(e), rmse=float(np.sqrt(np.mean(e**2))),
                mae=float(np.mean(np.abs(e))), bias=float(np.mean(e)))

def main():
    run = ROOT / 'results/paired_gate_ghi_seed42_20261015_attempt1'
    stations = ['sili', 'zhujia']
    data, point, sources = {}, {}, []
    for st in stations:
        path = run / st / 'paired_validation_predictions.npz'
        with np.load(path, allow_pickle=False) as z:
            dates = z['target_time_bjt'].astype(str)
            dates = np.asarray([v[:10] for v in dates])
            y = z['ghi'].astype(np.float64)
            pred = np.stack([z['raw_prediction_ghi_wm2'], z['gate_prediction_ghi_wm2']]).astype(np.float64)
        keep = ghi_day_keep_mask(np.repeat(st, len(y)), bjt_dates=dates)
        assert np.isfinite(y).all() and np.isfinite(pred).all()
        if (~keep).any():
            assert st == 'sili' and np.all(dates[~keep] == '2024-09-05') and np.all(y[~keep] == 0)
        data[st] = (dates[keep], (pred[:, keep] - y[keep])**2)
        point[st] = dict(original_n=len(y), removed=int((~keep).sum()), kept=int(keep.sum()),
                         dates=int(len(np.unique(dates[keep]))),
                         original={arm: metric(y, pred[i]) for i, arm in enumerate(['raw', 'gate'])},
                         clean={arm: metric(y[keep], pred[i, keep]) for i, arm in enumerate(['raw', 'gate'])})
        for scope in ['original', 'clean']:
            point[st][scope]['gate_minus_raw_rmse'] = point[st][scope]['gate']['rmse'] - point[st][scope]['raw']['rmse']
        sources.append(dict(path=str(path), sha256=sha(path)))

    # Shared calendar-date resampling preserves cross-station weather dependence.
    days = np.unique(np.concatenate([data[s][0] for s in stations]))
    sums, counts = [], []
    for st in stations:
        dates, error2 = data[st]
        sums.append(np.asarray([[error2[a, dates == d].sum() for d in days] for a in range(2)]))
        counts.append(np.asarray([(dates == d).sum() for d in days]))
    sums, counts = np.asarray(sums), np.asarray(counts)
    rng = np.random.default_rng(20261020)
    boot = []
    for _ in range(2000):
        w = np.bincount(rng.integers(0, len(days), len(days)), minlength=len(days))
        den = counts @ w
        if np.any(den == 0):
            raise ValueError('empty station in bootstrap draw')
        scores = np.sqrt((sums @ w) / den[:, None])
        boot.append(float(np.mean(scores[:, 1] - scores[:, 0])))
    equal = {scope: {arm: float(np.mean([point[s][scope][arm]['rmse'] for s in stations]))
                     for arm in ['raw', 'gate']} for scope in ['original', 'clean']}
    for scope in equal:
        equal[scope]['gate_minus_raw_rmse'] = equal[scope]['gate'] - equal[scope]['raw']
    out = dict(state='COMPLETE_FIXED_WEIGHT_VALIDATION_RESCORE', date='2026-10-20',
               sources=sources, quality_policy=dict(path=str(POLICY), sha256=sha(POLICY)),
               stations=point, equal_station=equal, validation_only=True, test_payloads_read=0,
               retrained=False, reselected=False, seed=42,
               bootstrap=dict(draws=2000, seed=20261020, days=days.tolist(),
                 method='shared BJT target-date union, paired methods, equal station RMSE',
                 ci95_gate_minus_raw=[float(v) for v in np.quantile(boot, [.025, .975])]),
               caveat='Old checkpoints selected with the original validation set. New CI uses shared dates, unlike the old station-independent date bootstrap.')
    (HERE/'OLD_GATE_DAYCLEAN_20261020.json').write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(equal_station=equal, ci95=out['bootstrap']['ci95_gate_minus_raw'],
                         removed={s:point[s]['removed'] for s in stations}), ensure_ascii=False))

if __name__ == '__main__':
    main()
