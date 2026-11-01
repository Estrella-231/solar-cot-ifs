"""Read only legacy source-NC lineage metadata for actual training rows."""
import json, re
from collections import Counter
from datetime import datetime
from pathlib import Path
p=Path('/home/Data_Pool_3/wangyc/irradiance_paper/solar_cot_ifs_20260905/data/cpp_aligned_20260905_v2/inventory_status.jsonl')
out={'ledger_path':str(p),'exists':p.is_file(),'test_buffer_payloads_read':False}
expected={(s,t):h for s,t,h in EXPECTED}
found={}; offsets=Counter(); bad=[]
if p.is_file():
    with p.open() as f:
        for line in f:
            try:r=json.loads(line)
            except ValueError:continue
            key=(r.get('station_id'),r.get('timestamp_utc'))
            if key in expected:found[key]=r
    for key,r in found.items():
        match=re.search(r'FY4B_AGRI_(\d{14})',r.get('source_nc',''))
        if not match:
            bad.append({'key':key,'reason':'missing_source_nc'});continue
        u=datetime.fromisoformat(key[1].replace('Z','+00:00')).replace(tzinfo=None)
        c=datetime.strptime(match.group(1),'%Y%m%d%H%M%S')
        offsets[str((c-u).total_seconds()/3600)]+=1
        if c!=u or expected[key]!=r.get('sidecar_sha256'):bad.append({'key':key,'reason':'time_or_hash_mismatch'})
out.update(expected_rows=len(expected),found_rows=len(found),cpp_minus_agri_utc_hours=dict(offsets),mismatches=len(bad),examples=bad[:4])
print(json.dumps(out,indent=2))
