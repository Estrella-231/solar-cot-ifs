"""Read-only event audit from archived V predictions and two offline references."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

ARMS = ('cot_last', 'cot_simvp_r', 'cot_p_c', 'cot_p_xc')
STATIONS = ('sili', 'zhujia')
STEP = timedelta(minutes=15)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read_csv(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def csvout(p, rr):
    if not rr:
        p.write_text('no_eligible_rows\n', encoding='utf-8')
        return
    fields = list(dict.fromkeys(k for r in rr for k in r))
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rr)


def runs(times, values, valid):
    """All contiguous high runs, including singletons and explicit censor flags."""
    out = []
    high = np.asarray(valid, bool) & (np.asarray(values) >= 30)
    i = 0
    while i < len(times):
        if not high[i]:
            i += 1; continue
        start = i
        while i + 1 < len(times) and times[i + 1] - times[i] == STEP and high[i + 1]:
            i += 1
        end = i
        left = start > 0 and times[start] - times[start - 1] == STEP and valid[start - 1] and values[start - 1] < 30
        right = end + 1 < len(times) and times[end + 1] - times[end] == STEP and valid[end + 1] and values[end + 1] < 30
        out.append({'start': times[start], 'end': times[end], 'n': end - start + 1,
                    'left_censored': not bool(left), 'right_censored': not bool(right),
                    'indices': list(range(start, end + 1))})
        i += 1
    return out


def select_detection(reference, prediction_runs):
    candidates = []
    for r in prediction_runs:
        if r['n'] < 2:
            continue
        overlap = int((min(reference['end'], r['end']) - max(reference['start'], r['start'])) / STEP) + 1
        if overlap > 0:
            candidates.append((-overlap, abs((r['start'] - reference['start']).total_seconds()), r['start'], r))
    return min(candidates, key=lambda x: x[:3])[-1] if candidates else None


def self_test():
    t0 = datetime(2025, 1, 1)
    tt = [t0 + i * STEP for i in range(7)]
    a = runs(tt[:4], [0, 31, 40, 0], [True] * 4)
    assert len(a) == 1 and a[0]['n'] == 2 and not a[0]['left_censored'] and not a[0]['right_censored']
    b = runs(tt[:5], [0, 31, 40, 50, 0], [True, True, False, True, True])
    assert [r['n'] for r in b] == [1, 1] and b[0]['right_censored'] and b[1]['left_censored']
    c = runs([tt[0], tt[1], tt[3], tt[4]], [31, 40, 50, 60], [True] * 4)
    assert [r['n'] for r in c] == [2, 2]
    d = runs(tt, [0, 0, 31, 40, 0, 0, 0], [True] * 7)
    assert select_detection(a[0], d)['start'] == tt[2]
    assert select_detection(a[0], runs(tt, [0, 0, 0, 0, 31, 40, 0], [True] * 7)) is None
    e = runs(tt[:4], [31, 40, 0, 0], [True] * 4)
    assert e[0]['left_censored'] and select_detection(a[0], e)['left_censored']
    print('PASS: continuity, gap/mask splitting, singleton exclusion, overlap, miss and censor fixtures.')


def peak_metrics(p, y):
    pi, yi = int(np.argmax(p)), int(np.argmax(y))
    pt, yt = int(np.sum(p == p[pi])), int(np.sum(y == y[yi]))
    return {'peak_pred': float(p[pi]), 'peak_ref': float(y[yi]),
            'peak_error': float(p[pi] - y[yi]), 'pred_peak_ties': pt, 'ref_peak_ties': yt,
            'peak_time_error_min': 15 * (pi - yi) if pt == yt == 1 else None}


def aggregate(rr):
    out = {}
    for key in ('peak_error', 'peak_time_error_min', 'patch3_peak_error'):
        x = np.array([r[key] for r in rr if r.get(key) is not None], np.float64)
        out[key] = {'n': len(x), 'mean': float(x.mean()) if len(x) else None,
                    'mae': float(np.abs(x).mean()) if len(x) else None,
                    'median': float(np.median(x)) if len(x) else None}
    tp, fp, fn = [sum(r[k] for r in rr) for k in ('tp', 'fp', 'fn')]
    out.update({'windows': len(rr), 'tp': tp, 'fp': fp, 'fn': fn,
                'pod': tp / (tp + fn) if tp + fn else None,
                'far': fp / (tp + fp) if tp + fp else None,
                'csi': tp / (tp + fp + fn) if tp + fp + fn else None})
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--base', type=Path); ap.add_argument('--output', type=Path)
    ap.add_argument('--self-test', action='store_true'); a = ap.parse_args()
    self_test()
    if a.self_test:
        return
    base, out = a.base, a.output
    if out.exists():
        raise FileExistsError(out)
    prev = base / 'results/teacher_forecast_error_decomposition_v1_20261021'
    receipt = json.loads((prev / 'artifact_audit.json').read_text())
    for name in ('observed_agri_r_u_cot.npy', 'corrected_common_reference_mask.npy', 'reference_rows.csv', 'matched_forecast_keys.csv', 'summary.json'):
        assert sha(prev / name) == receipt['files'][name], name
    refs = read_csv(prev / 'reference_rows.csv'); items = read_csv(prev / 'matched_forecast_keys.csv')
    assert len(refs) == 1398 and all(r['split'] == 'val' and r['state'] == 'SUPERVISED' for r in refs)
    imroot = base / 'data/cpp_score_inputs_v1/inference'
    manifest = json.loads((imroot / 'inference_manifest.json').read_text())
    summary = json.loads((prev / 'summary.json').read_text())
    assert sha(imroot / 'inference_manifest.json') == summary['source']['inference_manifest_sha256']
    ca = json.loads((base / 'data/cpp_score_inputs_v1/cpp_pack/pack_audit.json').read_text())
    assert sha(base / 'data/cpp_score_inputs_v1/cpp_pack/pack_audit.json') == summary['source']['cpp_audit_sha256']
    cp = base / 'data/cpp_score_inputs_v1/cpp_pack/cot.npy'
    assert sha(cp) == ca['arrays']['cot']['sha256']
    assert not manifest['test_payload_opened'] and not manifest['buffer_payload_opened']
    assert not manifest['future_observed_agri_read'] and not manifest['cpp_read']
    idx = np.array([int(r['array_index']) for r in refs])
    cpp = np.asarray(np.load(cp, mmap_mode='r')[idx, 0], np.float32)
    observed = np.load(prev / 'observed_agri_r_u_cot.npy')[:, 0]
    mask = np.load(prev / 'corrected_common_reference_mask.npy')[:, 0]
    forecasts = {}
    for arm in ARMS:
        path = imroot / 'V' / (arm + '.npy')
        assert sha(path) == manifest['artifact_sha256']['V/' + arm + '.npy']
        forecasts[arm] = np.load(path, mmap_mode='r')
    bywindow = defaultdict(list)
    for r in items:
        bywindow[(r['station'], int(r['sequence_index']))].append(r)
    windows = []
    for (st, sq), rr in sorted(bywindow.items()):
        rr = sorted(rr, key=lambda r: int(r['lead_index']))
        if len(rr) != 16 or [int(r['lead_index']) for r in rr] != list(range(16)):
            continue
        ri = np.array([int(r['reference_index']) for r in rr])
        tt = [datetime.fromisoformat(r['target_time_utc']) for r in rr]
        assert all(tt[j + 1] - tt[j] == STEP for j in range(15))
        if not mask[ri, 8, 8].all():
            continue
        windows.append({'station': st, 'sq': sq, 'init': datetime.fromisoformat(rr[0]['init_time_utc']),
                        'ri': ri, 'times': tt, 'block': rr[0]['block_id'], 'id': f'{st}_{sq}'})
    daily = {}
    for w in windows:
        dates = {(t + timedelta(hours=8)).date().isoformat() for t in w['times']}
        if len(dates) != 1:
            continue
        key = (w['station'], next(iter(dates)))
        if key not in daily or w['init'] < daily[key]['init']:
            daily[key] = w
    refmaps = {'CPP': cpp, 'observed_R': observed}
    eventrows, eventpred, dailyrows, coverage = [], [], [], []
    windowlist = []
    for (st, date), w in sorted(daily.items()):
        windowlist.append({'station': st, 'date_bjt': date, 'window_id': w['id'],
                           'sequence_index': w['sq'], 'init_time_utc': w['init'].isoformat(' '),
                           'first_target_bjt': (w['times'][0] + timedelta(hours=8)).isoformat(' '),
                           'last_target_bjt': (w['times'][-1] + timedelta(hours=8)).isoformat(' '),
                           'patch3_all_valid': bool(mask[w['ri'], 7:10, 7:10].all())})
        for refname, ref in refmaps.items():
            y = ref[w['ri'], 8, 8]
            for arm in ARMS:
                predmap = np.asarray(forecasts[arm][w['sq'], STATIONS.index(st), :, 0])
                p = predmap[:, 8, 8]
                assert np.isfinite(p).all()
                hi, ph = y >= 30, p >= 30
                patchok = bool(mask[w['ri'], 7:10, 7:10].all())
                dailyrows.append({'reference': refname, 'station': st, 'date_bjt': date,
                    'window_id': w['id'], 'arm': arm, 'tp': int((hi & ph).sum()),
                    'fp': int((~hi & ph).sum()), 'fn': int((hi & ~ph).sum()),
                    'reference_thick_frames': int(hi.sum()), 'prediction_thick_frames': int(ph.sum()),
                    **peak_metrics(p, y), 'patch3_peak_error': float(predmap[:, 7:10, 7:10].max() - ref[w['ri'], 7:10, 7:10].max()) if patchok else None})
    matchedri = set(int(r['reference_index']) for r in items)
    for st in STATIONS:
        ri = sorted([i for i in matchedri if refs[i]['station'] == st], key=lambda i: refs[i]['target_time_utc'])
        tt = [datetime.fromisoformat(refs[i]['target_time_utc']) for i in ri]
        for refname, ref in refmaps.items():
            es = runs(tt, ref[ri, 8, 8], mask[ri, 8, 8])
            coverage.append({'station': st, 'reference': refname, 'single_frame_runs': sum(e['n'] == 1 for e in es),
                             'two_frame_events': sum(e['n'] >= 2 for e in es)})
            for e in es:
                if e['n'] < 2:
                    continue
                eid = f'{st}_{refname}_{e["start"]:%Y%m%d%H%M}'
                candidates = [w for w in windows if w['station'] == st and w['times'][0] <= e['start']
                              and w['times'][-1] >= e['end'] + STEP] if not e['left_censored'] and not e['right_censored'] else []
                chosen = min(candidates, key=lambda w: w['init']) if candidates else None
                eventrows.append({'event_id': eid, 'station': st, 'reference': refname,
                    'start_time_bjt': (e['start'] + timedelta(hours=8)).isoformat(' '),
                    'end_time_bjt': (e['end'] + timedelta(hours=8)).isoformat(' '), 'thick_frames': e['n'],
                    'left_censored': e['left_censored'], 'right_censored': e['right_censored'],
                    'canonical_window': chosen['id'] if chosen else '',
                    'status': 'EVALUABLE' if chosen else ('CENSORED_REFERENCE' if e['left_censored'] or e['right_censored'] else 'NO_COMPLETE_WINDOW')})
                if chosen is None:
                    continue
                w = chosen
                for arm in ARMS:
                    p = np.asarray(forecasts[arm][w['sq'], STATIONS.index(st), :, 0, 8, 8])
                    pruns = runs(w['times'], p, np.ones(16, bool)); detected = select_detection(e, pruns)
                    eventpred.append({'event_id': eid, 'reference': refname, 'station': st,
                        'window_id': w['id'], 'arm': arm, 'start_lead_min': int((e['start'] - w['init']).total_seconds()/60),
                        'detected': detected is not None,
                        'predicted_left_censored': detected['left_censored'] if detected else False,
                        'arrival_error_min': int((detected['start'] - e['start']).total_seconds()/60) if detected and not detected['left_censored'] else None,
                        'predicted_events_in_window': sum(r['n'] >= 2 for r in pruns)})
    out.mkdir(parents=True)
    csvout(out / 'unique_reference_events.csv', eventrows)
    csvout(out / 'canonical_event_predictions.csv', eventpred)
    csvout(out / 'fixed_daily_windows.csv', windowlist)
    csvout(out / 'daily_window_metrics.csv', dailyrows)
    byref = {}
    for refname in refmaps:
        byref[refname] = {}
        for st in STATIONS:
            ee = [r for r in eventrows if r['reference'] == refname and r['station'] == st]
            entry = {'unique_events': len(ee), 'evaluable_events': sum(r['status'] == 'EVALUABLE' for r in ee),
                'censored_reference': sum(r['status'] == 'CENSORED_REFERENCE' for r in ee),
                'no_complete_window': sum(r['status'] == 'NO_COMPLETE_WINDOW' for r in ee), 'arms': {}}
            for arm in ARMS:
                er = [r for r in eventpred if r['reference'] == refname and r['station'] == st and r['arm'] == arm]
                arrivals = np.array([r['arrival_error_min'] for r in er if r['arrival_error_min'] is not None], float)
                entry['arms'][arm] = {'events': len(er), 'detected': sum(r['detected'] for r in er),
                    'missed': sum(not r['detected'] for r in er),
                    'predicted_left_censored': sum(r['predicted_left_censored'] for r in er),
                    'arrival_n': len(arrivals), 'arrival_mae_min': float(np.abs(arrivals).mean()) if len(arrivals) else None,
                    'arrival_bias_min': float(arrivals.mean()) if len(arrivals) else None,
                    'daily_window': aggregate([r for r in dailyrows if r['reference'] == refname and r['station'] == st and r['arm'] == arm])}
            byref[refname][st] = entry
    use = np.array(sorted(matchedri)); good = mask[use, 8, 8]
    yh, rh = cpp[use, 8, 8] >= 30, observed[use, 8, 8] >= 30
    disagreement = {'unique_valid_center_keys': int(good.sum()),
        'thick_class_disagreements': int(((yh != rh) & good).sum()),
        'CPP_thick_R_not': int((yh & ~rh & good).sum()), 'R_thick_CPP_not': int((rh & ~yh & good).sum())}
    # Save deterministic post-hoc examples, plus all candidate daily-window scores.
    selected = []
    for st in STATIONS:
        for refname in refmaps:
            rr = [r for r in dailyrows if r['station'] == st and r['reference'] == refname and r['arm'] == 'cot_p_xc']
            if rr:
                selected.append(max(rr, key=lambda r: abs(r['peak_error'])))
    csvout(out / 'posthoc_timeline_selection.csv', selected)
    timeline = []
    wbyid = {w['id']: w for w in daily.values()}
    for wid in sorted({r['window_id'] for r in selected}):
        w = wbyid[wid]
        for j in range(16):
            timeline.append({'window_id': wid, 'station': w['station'], 'lead_minutes': 15*(j+1),
                'init_time_bjt': (w['init']+timedelta(hours=8)).isoformat(' '),
                'target_time_bjt': (w['times'][j]+timedelta(hours=8)).isoformat(' '),
                'CPP': float(cpp[w['ri'][j], 8, 8]), 'observed_R': float(observed[w['ri'][j], 8, 8]),
                **{arm: float(forecasts[arm][w['sq'], STATIONS.index(w['station']), j, 0, 8, 8]) for arm in ARMS}})
    csvout(out / 'selected_actual_timelines.csv', timeline)
    result = {'state': 'COMPLETE_COT_EVENT_TIMING_DIAGNOSTIC', 'scope': 'seen V, sparse CPP-supported offline explanation; no GHI scoring',
        'complete_center_windows': len(windows), 'fixed_daily_windows': len(daily),
        'fixed_daily_windows_by_station': {st: sum(w['station'] == st for w in daily.values()) for st in STATIONS},
        'reference_disagreement': disagreement, 'run_coverage': coverage, 'by_reference_station': byref,
        'test_payload_opened': False, 'buffer_payload_opened': False, 'training_started': False,
        'future_observations_used_by_deployable_inference': False,
        'source': {'script_sha256': sha(__file__), 'plan_sha256': sha(base / 'COT_EVENT_TIMING_DIAGNOSTIC_PLAN_20261021.md'),
            'previous_summary_sha256': sha(prev / 'summary.json'), 'forecast_manifest_sha256': sha(imroot / 'inference_manifest.json')}}
    (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
