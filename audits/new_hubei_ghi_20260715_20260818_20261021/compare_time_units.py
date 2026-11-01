"""Read-only new/old export audit; old radiation decoded on TRAIN dates only."""
from __future__ import annotations
import collections
import csv
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
from build_hubei_ghi15_trainval import worksheet_paths, read_header_and_times, stream_cells, col_index
from attach_hubei_daylight_mask import solar_cosine_noaa, STATIONS as COORDS

NAMES = {'王寨供电所': 'Wangzhai', '米庄供电所': 'Mizhuang', '牛首供电所': 'Niushou'}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def quantiles(values):
    if not values:
        return {'n': 0}
    v = sorted(values)
    def q(p):
        pos = (len(v)-1)*p
        lo = int(pos); hi = math.ceil(pos)
        return v[lo] + (v[hi]-v[lo])*(pos-lo)
    return {'n': len(v), 'p10': q(.1), 'median': q(.5), 'p90': q(.9)}

def phase(records, station):
    result = {}
    for name, offset in [('as_BJT', 0), ('as_UTC_then_plus8', 8)]:
        dark = dark_positive = 0
        amplitude_dark = amplitude_all = 0.
        for t, ghi, _ in records:
            cos = solar_cosine_noaa(t+timedelta(hours=offset), *COORDS[station])
            isdark = cos < math.sin(math.radians(-6))
            dark += isdark
            dark_positive += isdark and ghi > 20
            amplitude_all += max(ghi, 0.)
            amplitude_dark += max(ghi, 0.) if isdark else 0.
        result[name] = {'rows': len(records), 'deep_night_rows': dark,
                        'deep_night_ghi_gt20_rows': dark_positive,
                        'positive_reading_amplitude_fraction_at_deep_night': amplitude_dark/amplitude_all if amplitude_all else None}
    return result

def hours(records, source, station):
    byhour = collections.defaultdict(list)
    for row in records:
        byhour[row[0].replace(minute=0, second=0, microsecond=0)].append(row)
    rows=[]
    for hour, rr in sorted(byhour.items()):
        rr.sort()
        gaps = [(b[0]-a[0]).total_seconds() for a,b in zip(rr,rr[1:])]
        integral = sum((a[1]+b[1])*.5*dt for a,b,dt in zip(rr,rr[1:],gaps))/1e6
        dk = rr[-1][2]-rr[0][2]
        decreases=sum(b[2]<a[2] for a,b in zip(rr,rr[1:]))
        reasons=[]
        if len(rr)<50: reasons.append('fewer_than_50_records')
        if (rr[-1][0]-rr[0][0]).total_seconds()<3000: reasons.append('span_below_50min')
        if max(gaps,default=0)>120: reasons.append('gap_above_120sec')
        if decreases: reasons.append('cumulative_decrease')
        if integral<.1: reasons.append('integral_below_0.1MJ')
        if dk<=0: reasons.append('nonpositive_cumulative_increment')
        rows.append({'source':source,'station':station,'hour_unzoned':str(hour),'n':len(rr),
                     'integral_F_if_Wm2_to_MJm2':integral,'delta_K':dk,
                     'K_increment_over_F_MJ':dk/integral if not reasons else None,
                     'cumulative_decrease_count':decreases,'excluded_reasons':'|'.join(reasons)})
    return rows

def main():
    out=HERE/'time_unit_comparison_v1'
    if out.exists(): raise FileExistsError(out)
    initial=json.loads((HERE/'results_v1/summary.json').read_text(encoding='utf-8'))
    sourcecsv=HERE/'results_v1/source_records_audit_only.csv'
    assert sha(sourcecsv)==initial['artifacts'][sourcecsv.name]
    fresh={name:[] for name in NAMES.values()}
    with sourcecsv.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            station=NAMES[r['source_sheet']]
            fresh[station].append((datetime.fromisoformat(r['timestamp_unzoned']),float(r['GHI_F']),float(r['daily_cumulative_K'])))
    oldroot=REPO/'refine-logs/hubei_cot_independent_20261014'
    invpath=oldroot/'audit_receipts/test_truth_source_inventory_20261018/source_inventory.json'
    inventory=json.loads(invpath.read_text(encoding='utf-8'))
    source=inventory['inputs']['jul7_14_workbook']
    oldfile=Path(source['path']); assert sha(oldfile)==source['sha256']
    assignment=oldroot/'frozen_split_20261015_v1/day_assignment.csv'
    with assignment.open(encoding='utf-8-sig',newline='') as f:
        split={r['date_BJT']:r['split'] for r in csv.DictReader(f)}
    old={name:[] for name in NAMES.values()}; old_meta={}
    with zipfile.ZipFile(oldfile) as z:
        sheets=worksheet_paths(z)
        assert set(sheets)==set(NAMES)
        for sheet,station in NAMES.items():
            headers,times=read_header_and_times(z,sheets[sheet])
            assert headers['总辐射1瞬时值']==6 and headers['时间']==17 and headers['总辐射1日累计']==11
            allowed={row for row,t in times.items() if t is not None and split.get(t.date().isoformat())=='train'}
            cells=collections.defaultdict(dict)
            for row,ref,kind,val in stream_cells(z,sheets[sheet],lambda row,col,ref:row in allowed and col in (6,11)):
                assert kind!='s', 'unexpected shared numeric cell'
                cells[row][col_index(ref)]=float(val)
            for row,values in sorted(cells.items()):
                assert split[times[row].date().isoformat()]=='train'
                if set(values)=={6,11} and all(math.isfinite(v) for v in values.values()):
                    old[station].append((times[row],values[6],values[11]))
            old_meta[station]={'captured_train_rows':len(old[station]),'train_dates':sorted({str(r[0].date()) for r in old[station]}),
                               'column_headers':headers,'timestamp_rows_read_metadata_only':len(times)}
    headerpath=oldroot/'aggregate_header_audit_20261015_v1/aggregate_headers.json'
    headers=json.loads(headerpath.read_text(encoding='utf-8'))
    original_headers=[]
    for book in headers['workbooks']:
        for sheet in book['sheets']:
            values=initial['sheet_schema'][sheet['name']]['headers_by_column']
            current=[values[chr(65+i)] for i in range(17)]
            original_headers.append({'old_file':book['file'],'sheet':sheet['name'],'same_17_headers_in_order':current==sheet['header_row_1']})
    hourly=[];summary={}
    for label,groups in [('new',fresh),('old_train_only',old)]:
        summary[label]={}
        for station,records in groups.items():
            hh=hours(records,label,station);hourly.extend(hh)
            ratios=[r['K_increment_over_F_MJ'] for r in hh if r['K_increment_over_F_MJ'] is not None]
            gaps=[(b[0]-a[0]).total_seconds() for a,b in zip(sorted(records),sorted(records)[1:])]
            summary[label][station]={'rows':len(records),'median_sampling_seconds':statistics.median(gaps),
                'phase_hypotheses':phase(records,station),'hour_unit_ratio':quantiles(ratios),
                'hour_exclusion_reasons':dict(collections.Counter(reason for r in hh for reason in r['excluded_reasons'].split('|') if reason)),
                'ghi_max':max(r[1] for r in records)}
    report={'state':'COMPLETE_READONLY_NEW_OLD_TIME_UNIT_COMPARISON', 'summary':summary,
        'header_comparisons':original_headers,'old_train_rows':old_meta,
        'inputs':{'new_workbook_sha256':initial['source_sha256'],'new_records_sha256':sha(sourcecsv),
                  'old_workbook':str(oldfile),'old_workbook_sha256':sha(oldfile),'frozen_days_sha256':sha(assignment),
                  'old_source_inventory_sha256':sha(invpath),'old_header_audit_sha256':sha(headerpath),
                  'solar_code_sha256':sha(REPO/'scripts/attach_hubei_daylight_mask.py'),
                  'previous_label_builder_sha256':sha(REPO/'scripts/build_hubei_ghi15_trainval.py'),
                  'previous_aggregation_code_sha256':sha(REPO/'scripts/hubei_ghi15_contract.py')},
        'script_sha256':sha(Path(__file__)),'plan_sha256':sha(HERE/'TIME_UNIT_COMPARISON_PLAN.md'),
        'old_test_buffer_val_numeric_cells_decoded':0,'new_model_scores_computed':False,
        'sensor_vs_platform_timestamp_established':False,'instrument_units_independently_certified':False,
        'formal_new_confirmation_labels_built':False}
    out.mkdir()
    with (out/'hourly_cumulative_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(hourly[0]));w.writeheader();w.writerows(hourly)
    report['hourly_csv_sha256']=sha(out/'hourly_cumulative_comparison.csv')
    (out/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'state':report['state'],'summary':summary,'headers_all_match':all(r['same_17_headers_in_order'] for r in original_headers)},ensure_ascii=False))

if __name__=='__main__': main()
