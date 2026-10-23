"""Freeze date-only nested partitions, without any prediction/label scoring."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
P=Path(__file__).resolve().parent;Q=P.parent/'hubei_cot_conditional_information_20261023'
mapping=json.loads((Q/'probe_attempt1/DATE_FOLDS.json').read_text())['date_to_fold']
def exclude(days):return {(datetime.strptime(day,'%Y%m%d')+timedelta(days=o)).strftime('%Y%m%d') for day in days for o in [-1,0,1]}
out=[]
for outer in range(3):
    outer_test=sorted(day for day,fold in mapping.items() if fold==outer)
    allowed=sorted(set(mapping)-exclude(outer_test));inner={}
    for mi,month in enumerate(sorted({day[:6] for day in allowed})):
        blocks=[]
        for day in [day for day in allowed if day[:6]==month]:
            if not blocks or (datetime.strptime(day,'%Y%m%d')-datetime.strptime(blocks[-1][-1],'%Y%m%d')).days>1:blocks.append([])
            blocks[-1].append(day)
        for bi,block in enumerate(blocks):
            for day in block:inner[day]=(mi+bi)%3
    for fold in range(3):
        held=sorted(day for day,v in inner.items() if v==fold);train=sorted(set(allowed)-exclude(held))
        assert held and len(train)>=8 and set(train).isdisjoint(exclude(outer_test))
        sites=[]
        for st in ['Niushou','Wangzhai','Mizhuang']:
            with np.load(Q/'inputs_attempt2'/f'{st}.npz') as z:dates=np.array([str(t)[:8] for t in z['target_time']]);av=z['available']
            tr=np.isin(dates,train)&av;va=np.isin(dates,held)
            assert tr.sum()>1000 and va.sum()>100
            sites.append(dict(station=st,train_rows=int(tr.sum()),validation_rows=int(va.sum()),train_dates=len(set(dates[tr])),validation_dates=len(set(dates[va]))))
        out.append(dict(outer_fold=outer,inner_fold=fold,outer_score_dates=outer_test,train_dates=train,inner_score_dates=held,stations=sites))
target=P/'INNER_PARTITIONS.json';assert not target.exists()
target.write_text(json.dumps(dict(state='DATE_ONLY_NESTED_PARTITIONS_FROZEN',partitions=out,label_scores_read=False,
    source_date_map_sha256=hashlib.sha256((Q/'probe_attempt1/DATE_FOLDS.json').read_bytes()).hexdigest(),model_fits_started=0),indent=2))
print('NESTED_PARTITIONS_READY',len(out),min(s['train_rows'] for p in out for s in p['stations']))
