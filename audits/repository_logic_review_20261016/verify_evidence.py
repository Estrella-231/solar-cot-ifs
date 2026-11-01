"""Read-only numeric witnesses for the repository/PPT review; no model execution."""
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT / 'audits/cpp_reference_join_20261016/sources'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    with (SRC / 'rows.csv').open(encoding='utf-8') as f:
        sup = [r for r in csv.DictReader(f) if r['state'] == 'SUPERVISED']
    with np.load(SRC / 'validation_predictions.npz', allow_pickle=False) as z:
        ids = z['sample_indices'].astype(int)
        pred = np.expm1(z['pred_log1p_cot'].astype(float))
        ref = z['reference_cot'].astype(float)
        mask = z['mask'].astype(bool)
    stations = np.array([sup[i]['station'] for i in ids])
    times = np.array([sup[i]['target_time_utc'] for i in ids])
    archive = ROOT / 'refine-logs/cot_value_first_20261013/clp_reference_monthly_R_seed42.npz'
    with np.load(archive, allow_pickle=False) as clp:
        binding = {
            'ids_equal': bool(np.array_equal(ids, clp['sample_indices'])),
            'stations_equal': bool(np.array_equal(stations, clp['station'].astype(str))),
            'target_times_equal': bool(np.array_equal(times, clp['target_time_utc'].astype(str))),
            'mask_equal': bool(np.array_equal(mask, clp['common_mask'].astype(bool))),
        }
    assert all(binding.values()), binding
    original = ROOT / 'refine-logs/cot_value_first_20261013/validation_predictions_monthly_R_seed42.npz'
    assert sha(original) == sha(SRC / 'validation_predictions.npz')
    scores = {}
    for station in ('all', 'sili', 'zhujia'):
        use = np.ones(len(ids), dtype=bool) if station == 'all' else stations == station
        x, y = ref[use][mask[use]], pred[use][mask[use]]
        record = dict(n=int(len(x)), rmse=float(np.sqrt(np.mean((y-x)**2))))
        for name, threshold in [('cloud', .1), ('thick', 30)]:
            a, b = y >= threshold, x >= threshold
            tp, fp, fn = (int(v.sum()) for v in (a & b, a & ~b, ~a & b))
            record[name] = dict(tp=tp, fp=fp, fn=fn, POD=tp/(tp+fn),
                                FAR=fp/(tp+fp), CSI=tp/(tp+fp+fn))
        scores[station] = record
    git = lambda *args: subprocess.check_output(['git', *args], cwd=ROOT).decode('utf-8', 'replace').strip()
    state = git('status', '--porcelain').splitlines()
    out = dict(scope='Read-only local review; no SSH, training, test payload or source PPT write',
               git_head=git('rev-parse', 'HEAD'), git_commit_count=int(git('rev-list', '--count', 'HEAD')),
               modified_tracked=sum(not r.startswith('??') for r in state),
               untracked_entries=sum(r.startswith('??') for r in state),
               source_sha256=sha(SRC / 'validation_predictions.npz'),
               direct_R_CLP_binding=binding, direct_R_CPP_threshold_scores=scores)
    (HERE / 'numeric_witness.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
