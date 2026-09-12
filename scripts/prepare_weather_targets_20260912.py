"""Freeze all existing train/validation target identities; never select by outcomes."""
import csv, datetime, hashlib, json
from pathlib import Path
import numpy as np

ROOT=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907')
pack=ROOT/'data/head_pack_trainval_20260908_v1'
out=ROOT/'data/weather_clp_trainval_20260912_v1'
out.mkdir(exist_ok=True)
assert not list(out.iterdir()), 'refuse overwrite of existing artifacts'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
audit=json.loads((pack/'audit.json').read_text())
assert audit['state']=='COMPLETE' and audit['test_used'] is False
assert sha(pack/'rows.csv')==audit['rows_csv_sha256']
split=np.load(pack/'split.npy'); assert sha(pack/'split.npy')==audit['arrays']['split']['sha256']
targets={}; mapping=[]
with (pack/'rows.csv').open() as f:
    for i,r in enumerate(csv.DictReader(f)):
        assert r['split'] in ('train','validation')
        assert int(split[i])=={'train':0,'validation':1}[r['split']]
        target=datetime.datetime.fromisoformat(r['target_time_utc'])
        init=datetime.datetime.fromisoformat(r['init_time_utc'])
        assert target-init==datetime.timedelta(minutes=int(r['lead_minutes']))
        key=(r['station'],r['target_time_utc'])
        if key not in targets:
            targets[key]=dict(target_id=len(targets),station=key[0],target_time_utc=key[1],split=int(split[i]))
        assert targets[key]['split']==int(split[i]),'target leakage'
        mapping.append(targets[key]['target_id'])
assert len(mapping)==len(split)==637902
(out/'targets.json').write_text(json.dumps(list(targets.values()),indent=2))
np.save(out/'row_target_id.npy',np.array(mapping,dtype=np.int64))
meta=dict(state='TARGETS_FROZEN',test_used=False,pack_audit_sha256=sha(pack/'audit.json'),rows_sha256=audit['rows_csv_sha256'],targets_sha256=sha(out/'targets.json'),mapping_sha256=sha(out/'row_target_id.npy'),counts={str(s):sum(r['split']==s for r in targets.values()) for s in (0,1)})
(out/'selection.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta))
