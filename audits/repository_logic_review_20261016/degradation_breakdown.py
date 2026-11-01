"""Describe where adding COT worsens current paired synchronous predictions."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'experiments/r_geometry_downstream_ghi_20261016/results/paired_three_arm_v1/analysis/paired_validation_scoring.npz'


def main():
    with np.load(SOURCE, allow_pickle=False) as z:
        p, y, st, dates = z['prediction'], z['truth'], z['station'], z['dates']
        indices = z['pack_index']
    a, c = p[:, 0], p[:, 2]
    with (ROOT / 'experiments/cot_underestimate_forensics_20261016/results/row_diagnostics.csv').open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(y)
    assert np.array_equal(indices, [int(r['pack_index']) for r in rows])
    assert np.allclose(y, [float(r['ghi']) for r in rows], atol=1e-8)
    assert np.allclose(a, np.array([[float(r[f'A_{seed}']) for r in rows] for seed in (42, 43, 44)]), atol=1e-8)
    assert np.allclose(c, np.array([[float(r[f'AC_{seed}']) for r in rows] for seed in (42, 43, 44)]), atol=1e-8)
    error_a, error_c = a-y, c-y
    delta = c-a
    dse = error_c**2-error_a**2
    worse = dse > 0
    kinds = {'underestimate_further_down': (error_a < 0) & (delta < 0),
             'overestimate_further_up': (error_a > 0) & (delta > 0),
             'right_direction_overshoot': ((error_a*delta) < 0) & worse}
    out = dict(scope='Current corrected-R three-frame observed AGRI paired validation, three seeds; descriptive post-hoc analysis; not forecast',
               source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(), test_payloads_read=0,
               counting='Seed-sample records are repeated outcomes, not independent observations; positive SSE shares are accounting, not causal effects',
               error_types={}, strata=[])
    positive = np.maximum(dse, 0)
    for name, m in kinds.items():
        out['error_types'][name] = dict(seed_sample_n=int(m.sum()),
            fraction_all_pct=float(100*m.mean()),
            fraction_worsened_pct=float(100*m.sum()/worse.sum()),
            positive_delta_SSE_share_pct=float(100*positive[m].sum()/positive.sum()),
            ghi_lt800_seed_sample_n=int((m & (y[None, :] < 800)).sum()))
    assert np.array_equal(np.logical_or.reduce(list(kinds.values())), worse)
    clear = np.array([float(r['clear']) for r in rows])
    kt = np.array([float(r['kt']) for r in rows])
    cot = np.array([float(r['cot_center_current']) for r in rows])
    groups = [('all', np.ones(len(y), dtype=bool)),
              ('GHI_0_200', (y>=0)&(y<200)), ('GHI_200_400', (y>=200)&(y<400)),
              ('GHI_400_600', (y>=400)&(y<600)), ('GHI_600_800', (y>=600)&(y<800)),
              ('GHI_ge800', y>=800), ('nearzero_R_COT_lt0.1', cot<.1),
              ('low_kt', (clear>=100)&(kt<.3)), ('middle_kt', (clear>=100)&(kt>=.3)&(kt<.9))]
    for name, m in groups:
        per = []
        by_station = {}
        for s in ('sili', 'zhujia'):
            use = m & (st == s)
            v = np.stack([np.sqrt(np.mean(e[:,use]**2, axis=1)) for e in (error_a,error_c)]).mean(axis=1)
            per.append(v)
            by_station[s] = dict(n=int(use.sum()), A=float(v[0]), AC=float(v[1]), delta=float(v[1]-v[0]))
        values = np.mean(per, axis=0)
        out['strata'].append(dict(group=name,n=int(m.sum()), dates=int(len(np.unique(dates[m]))),
            A=float(values[0]),AC=float(values[1]),delta=float(values[1]-values[0]),
            worse_seed_sample_pct=float(100*worse[:,m].mean()), by_station=by_station))
    (HERE / 'degradation_breakdown.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
