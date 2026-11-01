"""Metadata-only monthly split; never select blocks using weather or errors."""
import argparse
import calendar
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

STEP = timedelta(minutes=15)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(source, output, seed=42):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    frames = {}
    with open(source, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            paths = row['data_relpaths'].split('|')
            times = [datetime.strptime(Path(p).stem, '%Y%m%d%H%M') + timedelta(hours=8) for p in paths]
            assert len(times) == 24 and all(b-a == STEP for a,b in zip(times,times[1:]))
            assert str(times[0]) == row['BJT_start']
            for t,p in zip(times,paths):
                if t in frames and frames[t] != p:
                    raise ValueError('duplicate timestamp with conflicting path')
                frames[t] = p
    day_frames = Counter(t.date() for t in frames)
    month_days = defaultdict(list)
    for d in sorted(day_frames):
        month_days[d.strftime('%Y-%m')].append(d)
    assignment = {d: 'train' for d in day_frames}
    blocks = []
    for month, days in sorted(month_days.items()):
        # Metadata availability only: >=48 frames/day. Sparse months stay
        # training-only, rather than spending most of their days on holdouts.
        eligible = [d for d in days if day_frames[d] >= 48]
        candidates = [d for d in eligible if d+timedelta(days=1) in eligible]
        pairs = [(a,b) for a in candidates for b in candidates
                 if abs((a-b).days) >= 6]
        if len(eligible) < 20 or not pairs:
            blocks.append({'month':month,'status':'sparse_month_train_only',
                           'available_days':len(days),'eligible_days':len(eligible)})
            continue
        month_seed = int(hashlib.sha256(f'{seed}:{month}'.encode()).hexdigest()[:16],16)
        val_start, test_start = random.Random(month_seed).choice(pairs)
        for split, start in [('val',val_start),('test',test_start)]:
            for k in (0,1):
                assignment[start+timedelta(days=k)] = split
            for k in (-1,2):
                d = start+timedelta(days=k)
                # Buffer may cross a calendar-month boundary.
                if d in assignment:
                    if assignment[d] not in ('train','buffer'):
                        raise ValueError('holdout/buffer collision across months')
                    assignment[d] = 'buffer'
            blocks.append({'month':month,'status':'selected','split':split,
                           'start_BJT':str(start),'end_BJT':str(start+timedelta(days=1)),
                           'buffer_before_BJT':str(start-timedelta(days=1)),
                           'buffer_after_BJT':str(start+timedelta(days=2))})
    output.mkdir(parents=True)
    used = {k:set() for k in ('train','val','test')}
    counts = Counter()
    monthly = defaultdict(Counter)
    dropped = Counter()
    with (output/'hunan_8to16.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=['split','seq_id','BJT_start','data_relpaths'])
        writer.writeheader()
        for t in sorted(frames):
            ts = [t+k*STEP for k in range(24)]
            if any(u not in frames for u in ts):
                dropped['missing_frame'] += 1
                continue
            labels = {assignment[u.date()] for u in ts}
            if len(labels) != 1 or 'buffer' in labels:
                dropped['boundary_or_buffer'] += 1
                continue
            split = labels.pop()
            paths = [frames[u] for u in ts]
            writer.writerow({'split':split,'seq_id':'hunan_'+(t-timedelta(hours=8)).strftime('%Y%m%d%H%M'),
                             'BJT_start':str(t),'data_relpaths':'|'.join(paths)})
            used[split].update(paths)
            counts[split] += 1
            monthly[(t+7*STEP).strftime('%Y-%m')][split] += 1
    assert all(counts[k] > 0 for k in used)
    assert all(not (used[a]&used[b]) for a,b in [('train','val'),('train','test'),('val','test')])
    # Every held-out frame must have no training frame within its neighboring
    # calendar buffer day; verify against assignments, not just disjoint IDs.
    for block in blocks:
        if block['status'] != 'selected':
            continue
        for field in ('buffer_before_BJT','buffer_after_BJT'):
            d = datetime.strptime(block[field],'%Y-%m-%d').date()
            assert d not in assignment or assignment[d] == 'buffer'
    for split,paths in used.items():
        (output/f'{split}_frames.txt').write_text('\n'.join(sorted(paths))+'\n',encoding='utf-8')
    with (output/'day_assignment.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['date_BJT','split','available_frames'])
        for d in sorted(assignment): w.writerow([d,assignment[d],day_frames[d]])
    with (output/'monthly_counts.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['month','train','val','test'])
        for m in sorted(month_days):w.writerow([m,*(monthly[m][k] for k in used)])
    report={'split_id':'hunan_monthly_block_2d_buffer1d_seed42_v1','seed':seed,
            'source_manifest_sha256':sha(source),'manifest_sha256':sha(output/'hunan_8to16.csv'),
            'source':'union of timestamp/path metadata in previously audited manifest; includes former test dates',
            'selection':'fixed per-month seeded availability-only selection, no weather/CPP/GHI/errors used',
            'frame_range_BJT':[str(min(frames)),str(max(frames))],
            'source_unique_frames':len(frames),'sequence_counts':dict(counts),
            'unique_frames':{k:len(v) for k,v in used.items()},
            'available_day_counts':dict(Counter(assignment.values())),
            'monthly_sequence_counts':{k:dict(v) for k,v in monthly.items()},
            'blocks':blocks,'dropped_candidate_windows':dict(dropped),
            'frame_disjoint_splits':True,'full_24_frame_split_membership':True,
            'frame_list_sha256':{k:sha(output/f'{k}_frames.txt') for k in used},
            'buffer_calendar_days':1,'history_frames':8,'target_frames':16,
            'stride_minutes':15,'lead_minutes':list(range(15,241,15)),
            'test_payloads_read':0,'test_metrics_computed':False,
            'old_models_eligible_for_new_test':False,
            'scientific_scope':'season-representative held-out time blocks; not forward-year extrapolation',
            'sparse_month_policy':'<20 eligible days or no separated pair: train-only, explicitly reported'}
    (output/'manifest_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('manifest_sha256','sequence_counts','unique_frames','available_day_counts')}))
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-manifest',required=True)
    p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();build(a.source_manifest,a.output,a.seed)
