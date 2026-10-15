"""Reuse already extracted scan metadata without opening additional satellite files."""
import argparse,csv,json
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np

ap=argparse.ArgumentParser();ap.add_argument('--pack',type=Path,required=True);ap.add_argument('--metadata',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
rows=list(csv.DictReader((a.pack/'rows.csv').open()));offsets=json.loads((a.metadata/'scan_offsets.json').read_text());records=[]
for i,r in enumerate(rows):
    t=datetime.fromisoformat(r['target_time_utc']);times=[(t-timedelta(minutes=k)).isoformat(sep=' ') for k in (30,15,0)]
    si=int(r['station_index']);known=all(tt in offsets for tt in times)
    if known:
        actual=np.array([-1800.,-900.,0.])+np.array([offsets[tt][si] for tt in times]);brackets=bool(np.all(np.diff(actual)>0) and actual[0]<=-900 and actual[-1]>=0)
        assert brackets, (i,actual)
    else:actual=[np.nan]*3;brackets=False
    records.append(dict(pack_index=i,station=r['station'],split=r['split'],target_time_utc=r['target_time_utc'],scan_times_known=int(known),label_interval_bracketed=int(brackets),scan_seconds_relative_target='|'.join(str(x) for x in actual)))
with a.output.with_suffix('.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
audit=dict(state='COMPLETE_REUSED_SCAN_SUPPORT_AUDIT',rows=len(rows),known=sum(r['scan_times_known'] for r in records),unknown=sum(not r['scan_times_known'] for r in records),
    unknown_by_station_split={f'{st}_{sp}':sum(r['station']==st and r['split']==sp and not r['scan_times_known'] for r in records) for st in ('sili','zhujia') for sp in ('train','val')},metadata_source=str(a.metadata),test_payloads_read=0)
a.output.write_text(json.dumps(audit,indent=2));print(json.dumps(audit))
