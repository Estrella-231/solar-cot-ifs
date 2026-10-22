"""Source train metadata only; internal excluded days are not outer buffer."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;H=P.parent/'hubei_forecast_cot_increment_20261022/prepared_attempt1';out={}
for st in ['Niushou','Wangzhai','Mizhuang']:
    p=H/st;rows=list(csv.DictReader((p/'rows.csv').open()));roles=np.load(p/'roles.npy');assert len(rows)==len(roles)
    fdays={r['target_time_bjt'][:8] for r,v in zip(rows,roles) if v in ['F_fit','F_select']}
    cdays={r['target_time_bjt'][:8] for r,v in zip(rows,roles) if v=='C_select'}
    extra=np.array([v=='purged' and r['split']=='train' and r['target_time_bjt'][:8] not in fdays|cdays and r['init_time_bjt'][:8]==r['target_time_bjt'][:8] for r,v in zip(rows,roles)])
    small=roles=='C_fit';wide=small|extra
    assert all(r['split']=='train' for r,v in zip(rows,wide) if v)
    dates=lambda m:sorted({r['target_time_bjt'][:8] for r,v in zip(rows,m) if v})
    assert set(dates(wide)).isdisjoint(fdays|cdays)
    out[st]=dict(small_rows=int(small.sum()),small_dates=dates(small),additional_rows=int(extra.sum()),additional_dates=dates(extra),wide_rows=int(wide.sum()),wide_dates=dates(wide),select_rows=int((roles=='C_select').sum()),select_dates=sorted(cdays),row_metadata_sha256=hashlib.sha256((p/'rows.csv').read_bytes()).hexdigest(),outer_test_buffer_payloads_read=0)
result=P/'METADATA_AUDIT.json';assert not result.exists();result.write_text(json.dumps(out,indent=2));print(json.dumps({k:{n:v[n] if 'dates' not in n else len(v[n]) for n in ['small_rows','small_dates','additional_rows','additional_dates','wide_rows','wide_dates','select_rows','select_dates']} for k,v in out.items()},indent=2))
