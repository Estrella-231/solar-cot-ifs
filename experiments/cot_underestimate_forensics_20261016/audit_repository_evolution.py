"""Join archived residual outputs to current failures; no fitting or model changes."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    repo = Path(__file__).resolve().parents[2]
    here = Path(__file__).resolve().parent
    old_path = repo / 'experiments/monthly_R_station16_20261013_v1/results/signed_residual_oof_v3_unified_analysis_20261014_v1/validation_predictions_and_keys.npz'
    current_dir = repo / 'experiments/r_geometry_downstream_ghi_20261016/results/paired_three_arm_v1'
    current_path = current_dir / 'analysis/paired_validation_scoring.npz'
    rows_path = current_dir / 'validation_rows.csv'
    cases_path = here / 'results/selected_cases.json'
    with np.load(old_path, allow_pickle=False) as data:
        old = {key: data[key] for key in data.files}
    with np.load(current_path, allow_pickle=False) as data:
        current = {key: data[key] for key in data.files}
    assert not old['test_payload_opened'].item()
    rows = list(csv.DictReader(rows_path.open(encoding='utf-8', newline='')))
    assert len(rows) == len(current['truth'])
    lookup = {}
    for i in range(len(old['seed'])):
        key = (str(old['station'][i]), str(old['target_time_utc'][i]), int(old['seed'][i]))
        assert key not in lookup
        lookup[key] = i
    old_keys = ['pred_A_wm2', 'pred_AGRI_wm2', 'pred_AGRI_COT_wm2']
    records = []
    reconstructed = np.empty_like(current['prediction'])
    for j, row in enumerate(rows):
        assert row['station'] == current['station'][j] and row['split'] == 'val'
        for s, seed in enumerate((42, 43, 44)):
            i = lookup[(row['station'], row['target_time_utc'], seed)]
            assert float(old['ghi_truth_wm2'][i]) == current['truth'][j]
            assert str(old['bjt_date'][i]) == current['dates'][j]
            for a, key in enumerate(old_keys):
                reconstructed[s, a, j] = old[key][i]
            records.append(dict(station=row['station'], target_time_utc=row['target_time_utc'],
                                seed=seed, truth=float(current['truth'][j]),
                                archived_A=float(old['pred_A_wm2'][i]),
                                archived_AGRI_residual=float(old['pred_AGRI_wm2'][i]),
                                archived_AGRI_COT_residual=float(old['pred_AGRI_COT_wm2'][i]),
                                current_A=float(current['prediction'][s, 0, j]),
                                current_AC_corrected_R=float(current['prediction'][s, 2, j])))
    old_scores = {}
    for key in old_keys:
        old_scores[key] = float(np.mean([
            np.sqrt(np.mean((old[key][(old['station'] == station) & (old['seed'] == seed)].astype('f8') -
                             old['ghi_truth_wm2'][(old['station'] == station) & (old['seed'] == seed)]) ** 2))
            for station in ('sili', 'zhujia') for seed in (42, 43, 44)]))
    cases = []
    selected = json.loads(cases_path.read_text(encoding='utf-8'))
    for case in selected['cases']:
        if case['role'] != 'severe_under_drop':
            continue
        j = case['validation_index']
        pred = reconstructed[:, :, j]
        y = float(current['truth'][j])
        cases.append(dict(case_index=case['case_index'], station=case['station'], bjt=case['bjt'], truth=y,
                          archived_A=float(pred[:, 0].mean()), archived_AGRI_residual=float(pred[:, 1].mean()),
                          archived_AGRI_COT_residual=float(pred[:, 2].mean()),
                          current_A=case['A_mean'], current_AC_corrected_R=case['AC_mean'],
                          archived_AGRI_upward_seed_count=int((pred[:, 1] > pred[:, 0]).sum()),
                          archived_AGRI_smaller_SSE_seed_count=int(((pred[:, 1] - y)**2 < (pred[:, 0] - y)**2).sum())))
    severe = np.array([bool(int(row['severe_under_drop_3seed'])) for row in
                      csv.DictReader((here / 'results/row_diagnostics.csv').open(encoding='utf-8', newline=''))])
    assert severe.sum() == 254
    conditional = {}
    for a, name in enumerate(old_keys[1:], start=1):
        p = reconstructed[:, a, severe]
        base = reconstructed[:, 0, severe]
        y = current['truth'][severe][None, :]
        conditional[name] = dict(upward_fraction=float((p > base).mean()),
                                 smaller_SSE_fraction=float(((p-y)**2 < (base-y)**2).mean()),
                                 mean_bias_wm2=float((p-y).mean()),
                                 archived_A_mean_bias_wm2=float((base-y).mean()))
    out = here / 'results'
    for name, data in [('evolution_common_support.csv', records), ('evolution_six_failures.csv', cases)]:
        with (out / name).open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    summary = dict(state='PASS_EXACT_STATION_TIME_SEED_TRUTH_JOIN',
                   archived_validation_rows=len(old['seed']) // 3,
                   joined_current_validation_rows=len(rows), joined_seed_rows=len(records),
                   archived_equal_station_seed_RMSE_wm2=old_scores,
                   current_selected_failure_stratum_n=int(severe.sum()),
                   archived_residual_on_current_selected_failures=conditional,
                   selected_case_count=len(cases),
                   archived_AGRI_improves_all_three_seeds_case_count=sum(c['archived_AGRI_smaller_SSE_seed_count'] == 3 for c in cases),
                   sources=[dict(path=str(p.relative_to(repo)).replace('\\', '/'), sha256=sha(p))
                            for p in (old_path, current_path, rows_path, cases_path)],
                   interpretation='Descriptive bridge between different training/input contracts. Not a controlled method comparison. Failure stratum selected on current outputs; conditional scores do not estimate population skill.',
                   training_run=False, test_payloads_read=0)
    (out / 'evolution_bridge_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(cases, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
