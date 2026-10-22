"""Join label availability and causal history coverage, before model scoring."""
import csv,hashlib,json,sys
from collections import Counter
from datetime import datetime,timedelta
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1];sys.path.insert(0,str(REPO/'scripts'))
from attach_hubei_daylight_mask import solar_cosine_noaa,STATIONS
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    out=HERE/'input_selection_attempt1';out.mkdir(exist_ok=False)
    labelpath=HERE/'labels_attempt1/labels_15min.csv';satpath=HERE/'satellite_inventory_attempt1/history_input_support.csv'
    labels={(r['station'],r['target_time_BJT']):r for r in csv.DictReader(labelpath.open())}
    needed=set();keys=[];used=[];counts=Counter()
    for r in csv.DictReader(satpath.open()):
        init=datetime.fromisoformat(r['init_time_BJT']);n=0
        for station,coords in STATIONS.items():
            for lead in range(15,241,15):
                t=init+timedelta(minutes=lead);label=labels.get((station,t.isoformat(' ')))
                if label is None or label['mask']!='1':continue
                if solar_cosine_noaa(t,*coords)<=0:continue
                counts[(station,'available_daylight_labels_before_satellite')]+=1
                if r['eight_history_frames_available']!='1':
                    counts[(station,'missing_history_window')]+=1;continue
                keys.append(dict(station=station,init_time_BJT=init.isoformat(' '),target_time_BJT=t.isoformat(' '),
                    lead_minutes=lead,analysis='primary_shared_dates' if t<datetime(2026,8,10) else 'two_station_extension'))
                n+=1;counts[(station,'candidate_forecast_rows')]+=1
        if n:
            used.append(init.isoformat(' '))
            for offset in range(105,-1,-15):needed.add((init-timedelta(minutes=offset)).isoformat(' '))
    assert len({tuple(r[k] for k in ['station','init_time_BJT','target_time_BJT']) for r in keys})==len(keys)
    with (out/'candidate_keys.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(keys[0]));w.writeheader();w.writerows(keys)
    report=dict(state='INPUT_SUPPORT_ONLY_NO_SCORES',initializations_BJT=used,history_times_BJT=sorted(needed),
        initializations=len(used),history_frames=len(needed),rows=len(keys),counts={'/'.join(k):v for k,v in counts.items()},
        labels_sha256=sha(labelpath),satellite_support_sha256=sha(satpath),candidate_keys_sha256=sha(out/'candidate_keys.csv'),
        note='clear-sky positive check pending; same frozen noon-independent daylight criterion; no GHI magnitude or model errors used to select',
        old_test_payloads_read=0,old_buffer_payloads_read=0,new_period_scores_computed=0)
    (out/'INPUT_SELECTION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['initializations_BJT','history_times_BJT']},indent=2))
if __name__=='__main__':main()
