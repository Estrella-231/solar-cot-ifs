"""Reassign an existing candidate COT manifest to the paper's BJT blocks.

Does not certify original CPP quality filters or expand their coverage. The
output remains a candidate until raw-product quality and source paths pass.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

BJT = timezone(timedelta(hours=8))
BOUNDARIES = tuple(datetime.fromisoformat(x).replace(tzinfo=BJT) for x in
                   ('2024-04-02', '2025-07-01', '2025-10-01', '2026-07-01'))


def paper_split(utc):
    dt = datetime.fromisoformat(utc.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError('timestamp_utc must have explicit UTC offset')
    if dt.utcoffset() != timedelta(0):
        raise ValueError('timestamp_utc is not UTC')
    bjt = dt.astimezone(BJT)
    for name, lo, hi in zip(('train', 'validation', 'test'), BOUNDARIES, BOUNDARIES[1:]):
        if lo <= bjt < hi:
            return name, bjt.isoformat()
    return None, bjt.isoformat()


def build(source, output):
    output.mkdir(parents=True, exist_ok=False)
    counts, transitions, excluded = Counter(), Counter(), Counter()
    seen, ranges = set(), {}
    with source.open(newline='', encoding='utf-8-sig') as f, (output/'cot_candidate_split.csv').open('w', newline='') as out:
        reader = csv.DictReader(f)
        names = list(reader.fieldnames) + ['legacy_split', 'timestamp_bjt']
        writer = csv.DictWriter(out, fieldnames=names)
        writer.writeheader()
        for row in reader:
            split, bjt = paper_split(row['timestamp_utc'])
            key = row['station_id'], row['timestamp_utc']
            if key in seen:
                raise ValueError('duplicate station timestamp: ' + repr(key))
            seen.add(key)
            if split is None:
                excluded['outside_paper_dates'] += 1
                continue
            old = row['split']
            transitions[f'{old}->{split}'] += 1
            row.update(split=split, legacy_split=old, timestamp_bjt=bjt)
            writer.writerow(row)
            counts[f'{split}/{row["station_id"]}'] += 1
            if split not in ranges:
                ranges[split] = [bjt, bjt]
            else:
                ranges[split] = [min(ranges[split][0], bjt), max(ranges[split][1], bjt)]
    result = {'status': 'CANDIDATE_TIME_SPLIT_ONLY',
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'output_sha256': hashlib.sha256((output/'cot_candidate_split.csv').read_bytes()).hexdigest(),
        'input_rows': len(seen), 'counts': dict(counts), 'transitions': dict(transitions),
        'excluded': dict(excluded), 'ranges_bjt': ranges,
        'remaining_gates': ['official CPP quality provenance', 'source availability',
            'rebuild normalization on new train only', 'train fresh COT; old selected weights not reusable'],
        'note': 'Reuses a previously filtered candidate list; not a complete new inventory. No GHI read.'}
    (output/'split_audit.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output), indent=2))
