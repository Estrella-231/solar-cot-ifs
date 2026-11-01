"""Apply the user-confirmed invalid GHI station day to saved paired predictions."""
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from station_day_quality import POLICY, ghi_day_keep_mask


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def summarize(pred, truth, station, dates, keep, arms):
    days = np.unique(dates[keep])
    rng = np.random.default_rng(20261019)
    weights = rng.multinomial(len(days), np.full(len(days), 1/len(days)), 2000)
    point, boot, sites = [], [], {}
    for s in ('sili', 'zhujia'):
        use = keep & (station == s)
        e = (pred[:, :, use] - truth[use])**2
        v = np.sqrt(e.mean(axis=2))
        point.append(v)
        sites[s] = dict(n=int(use.sum()), rmse=dict(zip(arms, v.mean(axis=0).tolist())))
        di = np.searchsorted(days, dates[use])
        counts = np.bincount(di, minlength=len(days))
        sums = np.stack([e[:, :, di == i].sum(axis=2) for i in range(len(days))], axis=2)
        den = weights @ counts
        assert (den > 0).all()
        boot.append(np.sqrt(np.einsum('bd,sad->bsa', weights, sums)/den[:, None, None]))
    values = np.stack(point).mean(axis=(0, 1))
    b = np.stack(boot).mean(axis=(0, 2))
    contrasts = {}
    for i, arm in enumerate(arms[1:], 1):
        contrasts[arm+' minus '+arms[0]] = dict(delta=float(values[i]-values[0]),
            CI95=np.quantile(b[:, i]-b[:, 0], [.025,.975]).tolist(),
            improvement_pct=float(100*(values[0]-values[i])/values[0]))
    return dict(n=int(keep.sum()), dates=len(days), station_seed_equal_RMSE=dict(zip(arms,values.tolist())),
                by_station=sites, contrasts=contrasts)


def main():
    with (ROOT/'results/interval_alignment_20261015/ghi_rows_source.csv').open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        rows, fields = list(reader), reader.fieldnames
    assert set(r['split'] for r in rows) <= {'train','val'}
    keep = ghi_day_keep_mask([r['station'] for r in rows], target_times_utc=[r['target_time_utc'] for r in rows])
    excluded = [r for r, k in zip(rows, keep) if not k]
    assert len(excluded) == 39 and all(float(r['ghi']) == 0 and r['split'] == 'val' for r in excluded)
    with (HERE/'excluded_GHI_label_rows.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(excluded)
    with (HERE/'ghi_rows_source_dayclean_trainval.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(r for r,k in zip(rows,keep) if k)
    workbook = json.loads((ROOT/'results/cot_interval_history_20261015/audit/workbook_audit.json').read_text(encoding='utf-8'))
    raw = {k:v for k,v in workbook['stations']['sili']['values'].items() if k.startswith('2024-09-05')}
    assert len(raw)==121 and all(v == 0 for v in raw.values())
    out = dict(state='USER_CONFIRMED_INVALID_STATION_DAY_EXCLUDED_AND_FIXED_CHECKPOINTS_RESCORED',
        policy=str(POLICY.relative_to(ROOT)), policy_sha256=sha(POLICY),
        evidence=dict(source_rechecked_5min_cells=len(raw), source_zero_cells=len(raw),
                      source_time_range=[min(raw),max(raw)], excluded_label_rows=len(excluded),
                      excluded_labels_zero=len(excluded), affected_splits=dict(Counter(r['split'] for r in excluded))),
        limits='Saved checkpoints unchanged; selection originally used old validation. No retraining or independent-test claim. Other station/dates and other suspect-zero records retained.',
        test_payloads_read=0, experiments={})
    tasks = [
        ('current_three_frame_geometry', 'experiments/r_geometry_downstream_ghi_20261016/results/paired_three_arm_v1/analysis/paired_validation_scoring.npz', ['A_history','AC_old_R','AC_corrected_R']),
        ('continuous_history_original_R', 'results/cot_interval_history_20261015/paired_validation_scoring.npz', ['A_history','AC_history']),
        ('high_transmittance_loss', 'experiments/high_kt_loss_20261016/analysis/paired_validation_scoring.npz', ['A_uniform','A_bright2x','AC_uniform','AC_bright2x']),
        ('single_frame_A_AC_AP_AL', 'experiments/monthly_R_station16_20261013/results/r_al_formal_t18_20261014_v1/fp64_audit_v3/paired_predictions_fp64.npz', ['A','AC','AP','AL'])]
    for name, rel, arms in tasks:
        path=ROOT/rel
        with np.load(path, allow_pickle=False) as z:
            arrays={k:z[k] for k in z.files}
        st=arrays['station'].astype(str)
        if 'prediction' in arrays:
            pred, truth, dates = arrays['prediction'], arrays['truth'], arrays['dates'].astype(str)
        else:
            assert set(arrays['split']) == {'val'}
            pred=np.array([[arrays[f'{arm}_seed{seed}_ghi_wm2'] for arm in arms] for seed in (42,43,44)])
            truth=arrays['ghi_truth_wm2']
            dates=np.array([(datetime.fromisoformat(str(t))+timedelta(hours=8)).date().isoformat() for t in arrays['target_time_utc']])
        valid=ghi_day_keep_mask(st,bjt_dates=dates)
        assert np.all(truth[~valid]==0)
        # Keep the other station on the same BJT date, and every other date.
        assert valid[st=='zhujia'].all()
        before=summarize(pred,truth,st,dates,np.ones(len(st),bool),arms)
        after=summarize(pred,truth,st,dates,valid,arms)
        clean={k:(v[...,valid] if v.ndim and v.shape[-1]==len(st) else v) for k,v in arrays.items()}
        np.savez_compressed(HERE/f'{name}_dayclean.npz',**clean)
        np.savez_compressed(HERE/f'{name}_keep_mask.npz',keep=valid)
        out['experiments'][name]=dict(source=rel,source_sha256=sha(path),removed=int((~valid).sum()),before=before,after=after)
    (HERE/'RESULTS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    for name, record in out['experiments'].items():
        print(name,'removed',record['removed'],'after',record['after']['station_seed_equal_RMSE'])
        print('COT contrasts',record['after']['contrasts'])


if __name__=='__main__':
    main()
