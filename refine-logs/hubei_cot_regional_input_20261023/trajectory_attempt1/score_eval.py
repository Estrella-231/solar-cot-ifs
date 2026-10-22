"""FP64 common-pixel scoring; arm choice comes only from internal selection."""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

STATIONS = ['Niushou', 'Wangzhai', 'Mizhuang']
SEEDS = [42, 43, 44]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    assert not a.out.exists()
    selection_path = a.root / 'fits_attempt1/ARM_SELECTION.json'
    selection = json.loads(selection_path.read_text())
    assert selection['state'] == 'SIX_FITS_COMPLETE_ARM_FROZEN_BEFORE_OUTER_VAL'
    runs = selection['runs']
    assert {(r['arm'], r['seed']) for r in runs} == {(arm, s) for arm in ['full', 'repeat_last'] for s in SEEDS}
    means = {arm: np.mean([r['best_select_mse_normalized'] for r in runs if r['arm'] == arm]) for arm in ['full', 'repeat_last']}
    chosen = 'full' if means['full'] < means['repeat_last'] - 1e-12 else 'repeat_last'
    assert chosen == selection['selected_arm']
    rows = list(csv.DictReader(a.baseline.open()))
    row_keys = [(r['init'], r['target'], r['station'], int(r['lead']), int(r['dr']), int(r['dc'])) for r in rows]
    assert len(set(row_keys)) == len(rows)
    expected = {(r['init'], r['target'], int(r['lead'])) for r in rows}
    names = ['persistence', 'transport', 'forecast']
    values = [np.array([float(r[n]) for r in rows], np.float64) for n in names]
    sources = {'baseline': sha(a.baseline), 'arm_selection': sha(selection_path)}
    for arm in ['full', 'repeat_last']:
        for seed in SEEDS:
            name = f'seed{seed}_{arm}'
            path = a.root / 'eval_attempt1' / name / 'predictions.npz'
            receipt = json.loads(path.with_name('COMPLETE.json').read_text())
            spec = next(r for r in runs if r['arm'] == arm and r['seed'] == seed)
            assert receipt['state'] == 'FROZEN_TRAJECTORY_VAL_PREDICTIONS_SAVED'
            assert receipt['checkpoint_sha256'] == spec['best_sha256']
            assert receipt['prediction_sha256'] == sha(path)
            with np.load(path) as z:
                keys = list(zip(z['init'].astype(str), z['target'].astype(str), z['lead'].astype(int)))
                assert len(set(keys)) == len(keys) and set(keys) == expected
                pred = np.expm1(z['pred_log'].astype(np.float64))
                assert pred.shape == (len(keys), 3, 3, 3) and np.isfinite(pred).all()
                index = {k: i for i, k in enumerate(keys)}
                vals = np.array([pred[index[(r['init'], r['target'], int(r['lead']))], STATIONS.index(r['station']), int(r['dr'])+1, int(r['dc'])+1] for r in rows])
            names.append(name); values.append(vals); sources[name] = sha(path)
    pred = np.array(values)
    # CSV writes missing CPP as an empty cell; preserve it as missing, never clear sky.
    ref = np.array([float(r['reference']) if r['reference'].strip() else np.nan for r in rows])
    common = np.array([r['common'] == '1' for r in rows])
    center = np.array([r['dr'] == '0' and r['dc'] == '0' for r in rows])
    assert np.isfinite(pred[:, common]).all() and np.isfinite(ref[common]).all()
    site = np.array([STATIONS.index(r['station']) for r in rows])
    lead = np.array([int(r['lead']) for r in rows])
    dates, date_id = np.unique([r['date'] for r in rows], return_inverse=True)
    err = pred - ref
    records = []

    def metric(mask):
        assert mask.any()
        e = err[:, mask]
        return dict(n=int(mask.sum()), rmse=np.sqrt(np.mean(e*e, axis=1)).tolist(),
                    mae=np.mean(abs(e), axis=1).tolist(), bias=np.mean(e, axis=1).tolist())

    for scope, mask in [('station_pixel', common & center), ('station_3x3', common)]:
        for st in range(3):
            for ll in range(15, 241, 15):
                records.append(dict(scope=scope, station=STATIONS[st], lead=ll, **metric(mask & (site == st) & (lead == ll))))
    primary = [r for r in records if r['scope'] == 'station_pixel']
    main_rmse = np.mean([r['rmse'] for r in primary], axis=0)
    # Every draw resamples target dates jointly for every method/site/lead/seed.
    mask = common & center
    cell = site*16 + lead//15 - 1
    sse = np.zeros((len(names), 48, len(dates)), np.float64)
    count = np.zeros((48, len(dates)), np.int64)
    for c in range(48):
        mm = mask & (cell == c)
        count[c] = np.bincount(date_id[mm], minlength=len(dates))
        for j in range(len(names)):
            sse[j, c] = np.bincount(date_id[mm], weights=err[j, mm]**2, minlength=len(dates))
    w = np.random.default_rng(20261023).multinomial(len(dates), np.ones(len(dates))/len(dates), 5000)
    den = count @ w.T
    # No drop of empty cells on rare date draws: such a draw cannot estimate the declared metric.
    supported = np.all(den > 0, axis=0)
    assert supported.all(), 'A date draw has an unsupported station/lead cell; inspect, do not silently change weighting'
    boot = np.sqrt(np.einsum('mcd,bd->mcb', sse, w)/den[None]).mean(axis=1)
    arm_ids = {arm: [names.index(f'seed{s}_{arm}') for s in SEEDS] for arm in ['full', 'repeat_last']}
    def comparison(ids, other):
        delta = boot[ids].mean(axis=0) - (boot[other].mean(axis=0) if isinstance(other, list) else boot[other])
        return dict(delta_rmse=float(main_rmse[ids].mean() - (main_rmse[other].mean() if isinstance(other, list) else main_rmse[other])),
                    date_block_ci95=np.quantile(delta, [.025, .975]).tolist())
    comparisons = {arm: {base: comparison(ids, names.index(base)) for base in names[:3]} for arm, ids in arm_ids.items()}
    comparisons['full_minus_repeat_last'] = comparison(arm_ids['full'], arm_ids['repeat_last'])
    strata = {}
    for label, condition in [('near_zero', ref < 1), ('thin', (ref >= 1) & (ref < 10)), ('moderate', (ref >= 10) & (ref < 30)), ('thick', ref >= 30)]:
        mm = mask & condition
        strata[label] = metric(mm)
        event = pred[:, mm] >= 30; truth = ref[mm] >= 30
        tp = np.sum(event & truth, axis=1); fp = np.sum(event & ~truth, axis=1); fn = np.sum(~event & truth, axis=1)
        strata[label]['thick_confusion'] = dict(tp=tp.tolist(), fp=fp.tolist(), fn=fn.tolist())
    event = pred[:, mask] >= 30; truth = ref[mask] >= 30
    tp = np.sum(event & truth, axis=1); fp = np.sum(event & ~truth, axis=1); fn = np.sum(~event & truth, axis=1)
    def ratio(num, den):
        return [float(n/d) if d else None for n, d in zip(num, den)]
    detection = dict(tp=tp.tolist(), fp=fp.tolist(), fn=fn.tolist(), POD=ratio(tp,tp+fn), FAR=ratio(fp,tp+fp), CSI=ratio(tp,tp+fp+fn))
    by_site = {st: np.mean([r['rmse'] for r in primary if r['station'] == st], axis=0).tolist() for st in STATIONS}
    hourly = {}
    for h in range(1, 5):
        hourly[str(h)] = dict(exact=metric(mask & (lead == h*60)), interval=metric(mask & (lead > (h-1)*60) & (lead <= h*60)))
    ids = arm_ids[chosen]
    gates = dict(vs_persistence_ci_upper_negative=comparisons[chosen]['persistence']['date_block_ci95'][1] < 0,
                 vs_forecast_ci_upper_negative=comparisons[chosen]['forecast']['date_block_ci95'][1] < 0,
                 every_seed_not_worse_persistence=bool(np.all(main_rmse[ids] <= main_rmse[0])),
                 every_site_seed_mean_not_worse_persistence=all(np.mean(np.array(v)[ids]) <= v[0] for v in by_site.values()),
                 thick_seed_mean_rmse_not_worse_persistence=bool(np.mean(np.array(strata['thick']['rmse'])[ids]) <= strata['thick']['rmse'][0]))
    report = dict(state='FROZEN_TRAJECTORY_COMMON_VALIDATION_SCORED', methods=names, main_definition='equal station and equal 15-min lead RMSE; seeds scored separately, then averaged',
                  main_rmse=dict(zip(names, main_rmse.tolist())), selected_arm=chosen, comparisons=comparisons,
                  by_site=by_site, hourly=hourly, strata=strata, thick_detection=detection, gates=gates,
                  may_advance_paired_GHI=all(gates.values()), common_center_rows=int(mask.sum()), common_3x3_rows=int(common.sum()),
                  dates=dates.tolist(), bootstrap_draws=5000, sources=sources, CPP='retrieval reference, not independent truth',
                  outer_validation='previously exposed development validation, not independent test', test_payloads_read=0, GHI_gain_proven=False)
    a.out.mkdir(parents=True)
    (a.out/'SCORES.json').write_text(json.dumps(report, indent=2))
    (a.out/'SITE_LEAD.json').write_text(json.dumps(records, indent=2))
    np.savez(a.out/'DATE_ERROR_SUMS.npz', sse=sse, count=count, methods=names, dates=dates)
    print(json.dumps(dict(selected_arm=chosen, main_rmse=report['main_rmse'], gates=gates, may_advance_paired_GHI=all(gates.values())), indent=2))


if __name__ == '__main__':
    main()
