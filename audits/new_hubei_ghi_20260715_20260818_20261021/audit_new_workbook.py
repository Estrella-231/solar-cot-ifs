"""Audit user-supplied new Hubei observations, without model fitting/scoring."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment

EXPECTED_SHA = '6fc4659bf382c21f5db87e27b17762acb8588929e4580502b7a3283c4e5ee300'
STATIONS = {'王寨供电所':'wangzhai','米庄供电所':'mizhuang','牛首供电所':'niushou'}
FIELDS = {'GHI_F':5, 'DHI_G':6, 'DNI_H':7, 'daily_cumulative_K':10}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def numeric(v):
    try:
        a = float(v)
        return a if math.isfinite(a) else None
    except (TypeError, ValueError):
        return None


def csvout(p, rows):
    if not rows:
        p.write_text('no_rows\n', encoding='utf-8'); return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields);w.writeheader();w.writerows(rows)


def nums(values):
    vv = [v for v in values if v is not None]
    return {'valid':len(vv),'missing_nonfinite':len(values)-len(vv),
            'zero':sum(v==0 for v in vv),'positive':sum(v>0 for v in vv),'negative':sum(v<0 for v in vv),
            'min':min(vv) if vv else None,'max':max(vv) if vv else None,
            'median':statistics.median(vv) if vv else None}


def longest_clock_zero_run(rr):
    last=None;start=None;max_s=0;count=0;max_count=0
    for r in rr:
        t,g=r['dt'],r['GHI_F']
        inside=10<=t.hour<14
        if inside and g==0:
            if last is not None and 0<(t-last).total_seconds()<=120:
                count+=1
            else:
                start=t;count=1
            max_s=max(max_s,(t-start).total_seconds());max_count=max(max_count,count);last=t
        else:
            last=None;start=None;count=0
    return max_s,max_count


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--frozen-days',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();assert sha(a.source)==EXPECTED_SHA
    if a.output.exists():raise FileExistsError(a.output)
    with a.frozen_days.open(encoding='utf-8-sig') as f:old={r['date_BJT']:r['split'] for r in csv.DictReader(f)}
    a.output.mkdir(parents=True);source_dir=a.output/'source';source_dir.mkdir()
    copied=source_dir/'user_provided_weather.xlsx';shutil.copy2(a.source,copied);assert sha(copied)==EXPECTED_SHA
    wb=openpyxl.load_workbook(copied,read_only=True,data_only=False)
    assert set(wb.sheetnames)==set(STATIONS)
    summaryrows=[];dayrows=[];gaps=[];duplicates=[];metarows=[];records=[];source_sheets={}
    for ws in wb:
        it=ws.iter_rows(values_only=True);headers=list(next(it));nonempty=[h for h in headers if h is not None]
        assert headers[5]=='总辐射1瞬时值' and headers[15]=='采集点' and headers[16]=='时间'
        rr=[];formula_count=0
        for rownum,row in enumerate(it,2):
            if all(v is None for v in row):continue
            formula_count+=sum(isinstance(v,str) and v.startswith('=') for v in row)
            assert row[15]==ws.title,(ws.title,rownum,row[15])
            t=row[16] if isinstance(row[16],datetime) else datetime.fromisoformat(str(row[16]))
            assert t.date().isoformat() not in old, 'new source overlaps frozen old date: '+str(t)
            rec={'station':STATIONS[ws.title],'source_row':rownum,'dt':t,**{k:numeric(row[j]) for k,j in FIELDS.items()}}
            rr.append(rec)
        assert formula_count==0
        order_violations=sum(rr[i]['dt']<rr[i-1]['dt'] for i in range(1,len(rr)))
        rr.sort(key=lambda r:(r['dt'],r['source_row']))
        times=[r['dt'] for r in rr];bytime=defaultdict(list);byday=defaultdict(list)
        for r in rr:
            bytime[r['dt']].append(r);byday[r['dt'].date().isoformat()].append(r)
            records.append({'station':r['station'],'source_sheet':ws.title,'source_row':r['source_row'],
                            'timestamp_unzoned':r['dt'].isoformat(' '),'GHI_F':r['GHI_F'],
                            'DHI_G':r['DHI_G'],'DNI_H':r['DNI_H'],'daily_cumulative_K':r['daily_cumulative_K']})
        for t,rows in bytime.items():
            if len(rows)>1:
                vv={r['GHI_F'] for r in rows}
                duplicates.append({'station':STATIONS[ws.title],'timestamp_unzoned':str(t),
                    'count':len(rows),'GHI_conflict':len(vv)>1,'source_rows':'|'.join(str(r['source_row']) for r in rows)})
        intervals=[(times[i]-times[i-1]).total_seconds() for i in range(1,len(times))]
        for i,dt in enumerate(intervals,1):
            if dt>120:
                gaps.append({'station':STATIONS[ws.title],'previous_time':str(times[i-1]),'next_time':str(times[i]),'gap_seconds':dt})
        source_sheets[ws.title]={'headers_by_column':{openpyxl.utils.get_column_letter(j+1):h for j,h in enumerate(headers) if h is not None},
            'duplicate_header_names':{k:v for k,v in Counter(nonempty).items() if v>1},'formula_count':formula_count,
            'fields':{k:nums([r[k] for r in rr]) for k in FIELDS}}
        ss={'station':STATIONS[ws.title],'sheet':ws.title,'rows':len(rr),'days':len(byday),
            'first_timestamp':str(times[0]),'last_timestamp':str(times[-1]),
            'unique_timestamps':len(bytime),'extra_duplicate_rows':len(rr)-len(bytime),
            'conflicting_duplicate_timestamps':sum(len({r['GHI_F'] for r in v})>1 for v in bytime.values()),
            'source_order_decreases':order_violations,
            'median_interval_seconds':statistics.median(intervals),'most_common_interval_seconds':Counter(intervals).most_common(1)[0][0],
            'gaps_gt120_seconds':sum(v>120 for v in intervals),'max_gap_seconds':max(intervals),
            'nonzero_timestamp_seconds_rows':sum(t.second!=0 for t in times),
            'GHI_missing':sum(r['GHI_F'] is None for r in rr),'GHI_negative':sum(r['GHI_F'] is not None and r['GHI_F']<0 for r in rr),
            'GHI_gt1500_flag_only':sum(r['GHI_F'] is not None and r['GHI_F']>1500 for r in rr),
            'GHI_max':max(r['GHI_F'] for r in rr if r['GHI_F'] is not None),
            'DHI_nonzero':sum(r['DHI_G'] not in (None,0) for r in rr),
            'DNI_nonzero':sum(r['DNI_H'] not in (None,0) for r in rr),
            'overlap_frozen_date_count':len(set(byday)&set(old))}
        summaryrows.append(ss)
        for date,dr in sorted(byday.items()):
            g=nums([r['GHI_F'] for r in dr]);clock=[r for r in dr if 10<=r['dt'].hour<14]
            max_s,max_count=longest_clock_zero_run(dr)
            dayrows.append({'station':STATIONS[ws.title],'date_unzoned':date,'rows':len(dr),
                'first_time':str(dr[0]['dt']),'last_time':str(dr[-1]['dt']),
                'unique_clock_minutes':len({r['dt'].replace(second=0,microsecond=0) for r in dr}),
                'GHI_valid':g['valid'],'GHI_zero':g['zero'],'GHI_positive':g['positive'],
                'GHI_negative':g['negative'],'GHI_max':g['max'],'GHI_missing_nonfinite':g['missing_nonfinite'],
                'clock10to14_rows':len(clock),'clock10to14_zero':sum(r['GHI_F']==0 for r in clock),
                'clock10to14_longest_zero_span_s':max_s,'clock10to14_longest_zero_records':max_count,
                'all_valid_GHI_zero':g['valid']>0 and g['zero']==g['valid'],
                'clock_proxy_is_not_solar_day_mask':True})
        metarows.append({'sheet':ws.title,'station':STATIONS[ws.title],'row_count':len(rr),'data_type':'instantaneous total radiation F; units/time semantics unresolved',
                        'role':'NEW_OBSERVATION_CANDIDATE_NOT_RELEASED','region':'Hubei','source_sha256':EXPECTED_SHA})
        print(json.dumps(ss,ensure_ascii=False),flush=True)
    wb.close()
    for name,rs in [('station_summary.csv',summaryrows),('daily_quality.csv',dayrows),('time_gaps_gt120s.csv',gaps),
                    ('duplicate_timestamps.csv',duplicates),('sheet_provenance.csv',metarows),('source_records_audit_only.csv',records)]:
        csvout(a.output/name,rs)
    book=openpyxl.Workbook();book.remove(book.active)
    for title,rows in [('Station audit',summaryrows),('Daily audit',dayrows),('Provenance',metarows),('Time gaps',gaps),('Duplicates',duplicates)]:
        ws=book.create_sheet(title)
        if not rows:ws.append(['No rows']);continue
        fields=list(rows[0]);ws.append(fields)
        for row in rows:ws.append([row.get(k) for k in fields])
        ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
        for c in ws[1]:c.font=Font(name='Arial',bold=True,color='FFFFFF');c.fill=PatternFill('solid',fgColor='334155');c.alignment=Alignment(wrap_text=True)
        ws.row_dimensions[1].height=42
        for j,key in enumerate(fields,1):ws.column_dimensions[openpyxl.utils.get_column_letter(j)].width=min(38,max(16,len(key)*.8))
    note=book.create_sheet('Read me')
    for text in ['Read-only observational quality audit; no model prediction, fitting or evaluation.',
                 'All timestamps are unzoned as supplied. Timezone, units, and measurement-vs-receipt timing await confirmation.',
                 'GHI instantaneous readings are not released as 15-minute mean labels.',
                 'Counts and statistics are imported program audit observations; this workbook is not a forecasting model.',
                 'No automatic station/day removal; raw source preserved. GHI >1500 is a flag only.',
                 'A/C duplicate headers retained. Missing DHI/DNI information must not be treated as clear sky.',
                 'New Hubei data cannot independently confirm Hunan models. Old test/buffer values remain closed.',
                 'Source SHA256: '+EXPECTED_SHA]:note.append([text])
    note.column_dimensions['A'].width=130
    book.save(a.output/'新观测质量核查.xlsx')
    check=openpyxl.load_workbook(a.output/'新观测质量核查.xlsx',read_only=True,data_only=False)
    assert len(check.sheetnames)==6
    assert not any(c.data_type in ('e','f') for ws in check for row in ws for c in row)
    check.close()
    result={'state':'COMPLETE_NEW_HUBEI_OBSERVATION_SOURCE_QC_NOT_RELEASED', 'source_sha256':EXPECTED_SHA,
        'source_original_path':str(a.source.resolve()),'source_copy_sha256':sha(copied),
        'frozen_days_sha256':sha(a.frozen_days),'frozen_old_date_min':min(old),'frozen_old_date_max':max(old),
        'stations':summaryrows,'sheet_schema':source_sheets,'date_count_union':len({r['date_unzoned'] for r in dayrows}),
        'time_timezone_confirmed':False,'radiation_units_confirmed':False,'sampling_vs_receipt_time_confirmed':False,
        'formal_15minute_labels_built':False,'models_loaded':False,'training_started':False,'prediction_scores_computed':False,
        'old_test_payload_opened':False,'old_buffer_payload_opened':False,'historical_noncontact_audit_complete':False,
        'script_sha256':sha(Path(__file__)),'artifacts':{p.name:sha(p) for p in a.output.iterdir() if p.is_file()}}
    (a.output/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print('COMPLETE: all three sheets audited; model fitting/scoring remains closed.',flush=True)


if __name__=='__main__':main()
