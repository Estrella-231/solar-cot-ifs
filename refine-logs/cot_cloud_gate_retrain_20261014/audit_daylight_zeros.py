"""Versioned source-only train/val zero flags, without changing official labels."""
import argparse,csv,json,math,collections
from datetime import datetime,timedelta
from pathlib import Path

COORD={'sili':(25.9478611111,112.347916667),'zhujia':(29.566666667,112.45)}
def elevation(t,st):
    lat,lon=COORD[st];hour=t.hour+t.minute/60+t.second/3600;g=2*math.pi/365*(t.timetuple().tm_yday-1+(hour-12)/24)
    eq=229.18*(.000075+.001868*math.cos(g)-.032077*math.sin(g)-.014615*math.cos(2*g)-.040849*math.sin(2*g))
    d=.006918-.399912*math.cos(g)+.070257*math.sin(g)-.006758*math.cos(2*g)+.000907*math.sin(2*g)-.002697*math.cos(3*g)+.00148*math.sin(3*g)
    ha=math.radians(((hour*60+eq+4*lon-480)%1440)/4-180);la=math.radians(lat)
    return math.degrees(math.asin(max(-1,min(1,math.sin(la)*math.sin(d)+math.cos(la)*math.cos(d)*math.cos(ha)))))
def main():
    p=argparse.ArgumentParser();p.add_argument('--labels',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader(a.labels.open(encoding='utf-8')));native={};splits=collections.defaultdict(set)
    for r in rows:
        assert r['new_split'] in ('train','val')
        ts=r['label_record_times_bjt'].split('|');vs=r['raw_ghi_5min_wm2'].split('|');assert len(ts)==len(vs)==3
        for t,v in zip(ts,vs):
            try:v=float(v)
            except ValueError:v=None
            key=(r['station'],t)
            if key in native:assert native[key]==v or (native[key] is not None and v is not None and math.isnan(native[key]) and math.isnan(v))
            native[key]=v;splits[key].add(r['new_split'])
    dayzero={};groups=collections.defaultdict(list);summary={};requests=[]
    for key,v in native.items():
        st,t=key;el=elevation(datetime.fromisoformat(t)-timedelta(minutes=2.5),st)
        if v==0 and el>=10:dayzero[key]=el;groups[st].append(t)
    persistent=set();runs=[]
    for st,times in groups.items():
        times.sort();current=[]
        def finish(cur):
            if not cur:return
            record=dict(station=st,start_bjt=cur[0],end_bjt=cur[-1],zero_5min_records=len(cur),duration_minutes=len(cur)*5,split=sorted(set.union(*(splits[(st,t)] for t in cur))))
            runs.append(record)
            if len(cur)>=12:persistent.update((st,t) for t in cur)
        for t in times:
            if current and datetime.fromisoformat(t)-datetime.fromisoformat(current[-1])!=timedelta(minutes=5):finish(current);current=[]
            current.append(t)
        finish(current)
    for st in ('sili','zhujia'):
        rr=[r for r in runs if r['station']==st];summary[st]=dict(native_requested=len([k for k in native if k[0]==st]),day_zero_records=sum(k[0]==st for k in dayzero),persistent_zero_records=sum(k[0]==st for k in persistent),persistent_runs=[r for r in rr if r['zero_5min_records']>=12],zero_by_split={sp:sum(k[0]==st and sp in splits[k] for k in dayzero) for sp in ('train','val')},max_run_records=max((r['zero_5min_records'] for r in rr),default=0))
    for (st,t),el in sorted(dayzero.items()):requests.append(dict(station=st,time_bjt=t,expected_value=0,elevation_deg=el,persistent_60min=int((st,t) in persistent),split='|'.join(sorted(splits[(st,t)]))))
    def write(name,records):
        with (a.output/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    write('source_zero_requests.csv',requests);flags=[];counter=collections.Counter()
    for r in rows:
        st=r['station'];keys=[(st,t) for t in r['label_record_times_bjt'].split('|')];pe=any(k in persistent for k in keys);iz=any(k in dayzero for k in keys)
        flag='persistent_day_zero_60min' if pe else ('other_day_zero' if iz else 'none')
        valid=r['main_mask']=='True';counter[(r['new_split'],st,flag,'main' if valid else 'nonmain')]+=1
        flags.append(dict(split=r['new_split'],station=st,target_time_utc=r['target_time_utc'],target_time_bjt=r['target_time_bjt'],original_main_mask=int(valid),quality_flag=flag,candidate_qc1_main_mask=int(valid and not pe),source_values_unchanged=1))
    write('label_quality_flags.csv',flags)
    result=dict(state='COMPLETE_SOURCE_ONLY_ZERO_FLAGS',rule='5min interval midpoint NOAA elevation >=10deg, exactly 0, >=12 contiguous 5min records; missing times break runs',rule_identity='candidate_QC1_suspect_not_confirmed_fault',stations=summary,counts={'/'.join(k):v for k,v in sorted(counter.items())},test_payloads_read=0,official_labels_changed=False)
    (a.output/'zero_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
