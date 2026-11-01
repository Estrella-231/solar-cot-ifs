"""Describe error concentration in rapidly varying 15-minute validation labels."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'results/cot_interval_history_20261015/paired_validation_scoring.npz'


def main():
    with np.load(SOURCE, allow_pickle=False) as z:
        pred, truth = z['prediction'], z['truth']
        station, dates = z['station'], z['dates']
        rapid = z['interval_spread'] > 100
        zero = z['zero_flag']
    err = (pred - truth[None, None, :]) ** 2
    out = dict(scope='Observed three-frame AGRI synchronous validation; original frozen R; not forecast',
               criterion='Range of three source 5-minute GHI interval values >100 W/m2',
               limitation='Rapid variation is a proxy, not proof of CPP-clear/interval-cloud mismatch or a causal error fraction',
               source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               test_payloads_read=0, groups={})
    for name, use in [('all', np.ones(len(truth), dtype=bool)), ('sili', station == 'sili'),
                      ('zhujia', station == 'zhujia'), ('exclude_suspect_zero_sensitivity', ~zero)]:
        high, other = use & rapid, use & ~rapid
        scores = {}
        for arm, label in enumerate(('A', 'AC')):
            e = err[:, arm]
            scores[label] = dict(
                rapid_SSE_share_pct=float(100 * e[:, high].sum() / e[:, use].sum()),
                rapid_pooled_RMSE=float(np.sqrt(e[:, high].mean())),
                other_pooled_RMSE=float(np.sqrt(e[:, other].mean())),
                all_pooled_RMSE=float(np.sqrt(e[:, use].mean())))
        out['groups'][name] = dict(n=int(use.sum()), rapid_n=int(high.sum()),
            rapid_pct=float(100 * high.sum() / use.sum()), rapid_dates=int(len(np.unique(dates[high]))),
            scores=scores)
    # Reproduce the existing station/seed-equal subgroup result as a contract check.
    for label, use in [('all_validation', np.ones(len(truth), dtype=bool)), ('high_interval_variation_gt100', rapid)]:
        value = np.stack([np.sqrt(err[:, :, use & (station == s)].mean(axis=2))
                          for s in ('sili', 'zhujia')]).mean(axis=(0, 1))
        expected = json.loads((ROOT / 'results/cot_interval_history_20261015/independent_score.json').read_text())['groups'][label]
        assert np.allclose(value, [expected['A_RMSE'], expected['AC_RMSE']], atol=1e-10, rtol=0)
    (HERE / 'interval_variation_impact.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
