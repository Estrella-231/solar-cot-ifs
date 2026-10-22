"""New-source labels only; no model scoring, old test or buffer payloads."""
import argparse, csv, hashlib, json, math, sys, zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
from build_hubei_ghi15_trainval import (worksheet_paths, read_header_and_times,
    stream_cells, col_index, ceil_target, STATIONS)
from hubei_ghi15_contract import aggregate_window

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def writecsv(p, rows):
    with p.open('w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def independent(rr, target):
    start=target-timedelta(minutes=15);seen={};bins=[[] for _ in range(15)]
    for t,v in rr:
        if not start<t<=target: continue
        if not math.isfinite(v) or v<0: return None
        if t in seen and seen[t]!=v: return None
        seen[t]=v
    for t,v in seen.items():
        # Second implementation: explicit minute-interval comparisons.
        hit=[i for i in range(15) if start+timedelta(minutes=i)<t<=start+timedelta(minutes=i+1)]
        assert len(hit)==1;bins[hit[0]].append(v)
    if any(not b for b in bins): return None
    return math.fsum(math.fsum(b)/len(b) for b in bins)/15

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    source=REPO/'audits/new_hubei_ghi_20260715_20260818_20261021/results_v1/source/user_provided_weather.xlsx'
    assert sha(source)=='6fc4659bf382c21f5db87e27b17762acb8588929e4580502b7a3283c4e5ee300'
    a.out.mkdir(parents=True,exist_ok=False)
    frozen=REPO/'refine-logs/hubei_cot_independent_20261014/frozen_split_20261015_v1/day_assignment.csv'
    old={r['date_BJT'] for r in csv.DictReader(frozen.open(encoding='utf-8-sig'))}
    labels=[];summary={};difference=0.;nchecked=0
    with zipfile.ZipFile(source) as z:
        sheets=worksheet_paths(z);assert set(sheets)==set(STATIONS)
        for name,station in STATIONS.items():
            headers,times=read_header_and_times(z,sheets[name]);assert headers['总辐射1瞬时值']==6
            assert all(t is not None and t.date().isoformat() not in old for t in times.values())
            assert all(datetime(2026,7,15)<=t<datetime(2026,8,19) for t in times.values())
            vals={}
            for row,ref,typ,value in stream_cells(z,sheets[name],lambda row,col,ref:row>1 and col==6):
                assert typ!='s'; vals[row]=float(value)
            assert set(vals)==set(times)
            grouped=defaultdict(list)
            for row,t in times.items():grouped[ceil_target(t)].append((t,vals[row]))
            reasons=Counter();daily=defaultdict(Counter);count=0
            for daynum in range(35):
                day=datetime(2026,7,15)+timedelta(days=daynum)
                for minute in range(0,1440,15):
                    t=day+timedelta(minutes=minute);rr=grouped.get(t,[])
                    q=aggregate_window(rr,t);check=independent(rr,t)
                    assert bool(q['valid'])==(check is not None)
                    if check is not None:
                        difference=max(difference,abs(check-q['ghi']));nchecked+=1
                    reasons[q['reason']]+=1;daily[t.date().isoformat()][q['reason']]+=1
                    labels.append(dict(station=station,target_time_BJT=t.isoformat(' '),
                        split='new_period_confirmation_candidate',mask=int(q['valid']),
                        ghi_wm2=q['ghi'] if q['valid'] else '',reason=q['reason'],
                        minute_bins=q.get('minute_bins',0),source_records=len(rr)))
                    count+=q['valid']
            summary[station]=dict(source_rows=len(times),source_start=min(times.values()).isoformat(' '),
                source_end=max(times.values()).isoformat(' '),valid_intervals=count,reasons=dict(reasons),
                daily={k:dict(v) for k,v in daily.items()})
    assert difference<1e-10
    writecsv(a.out/'labels_15min.csv',labels)
    report=dict(state='LABELS_PREPARED_NO_MODEL_SCORING',stations=summary,source_sha256=sha(source),
        source_path=str(source),script_sha256=sha(__file__),aggregation_sha256=sha(REPO/'scripts/hubei_ghi15_contract.py'),
        old_date_manifest_sha256=sha(frozen),labels_sha256=sha(a.out/'labels_15min.csv'),
        independent_aggregation_valid_intervals=nchecked,independent_max_abs_difference=difference,
        time_unit_basis='TIME_UNIT_COMPARISON_RESULTS.md; BJT/Wm2 operational inference',
        old_test_payloads_read=0,old_buffer_payloads_read=0,models_loaded=0,new_period_scores_computed=0,
        note='Labels only; daytime/clear-sky and causal satellite joins pending, not forecast sample counts.')
    (a.out/'LABEL_REPORT.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:dict(valid_intervals=v['valid_intervals'],source_rows=v['source_rows']) for k,v in summary.items()}))

if __name__=='__main__':main()
