import csv,json,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]);out=root/'evaluation_attempt1';assert not out.exists();out.mkdir()
shards=[root/f'evaluation_shards1/rank{i}' for i in range(8)]
receipts=[json.loads((s/'completion.json').read_text()) for s in shards]
assert all(r['status']=='PASS' and r['test_payloads_read']==0 for r in receipts)
assert sum(r['windows'] for r in receipts)==3140
assert all(r['head_sha256']==receipts[0]['head_sha256'] for r in receipts)
for name in ('cot_rows.csv','ghi_rows.csv'):
 rows=[]
 for s in shards:
  with (s/name).open() as f:rows.extend(csv.DictReader(f))
 assert len({(r['window'],r['station'],r['lead_minutes']) for r in rows})==len(rows)
 with (out/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for name in ('cot_predictions.npz','ghi_predictions.npz'):
 values={}
 for s in shards:
  with np.load(s/name) as z:
   if values:assert set(z.files)==set(values)
   for k in z.files:values.setdefault(k,[]).append(z[k].copy())
 np.savez_compressed(out/name,**{k:np.concatenate(v) for k,v in values.items()})
excluded=[]
for s in shards:excluded.extend(json.loads((s/'excluded_ghi_rows.json').read_text()))
(out/'excluded_ghi_rows.json').write_text(json.dumps(excluded))
r={**receipts[0],'windows':3140,'COT_rows':sum(v['COT_rows'] for v in receipts),'GHI_rows':sum(v['GHI_rows'] for v in receipts),'scope':'all 3140 frozen validation windows; fixed Hubei DyDiff and Hunan SimVP, R and paired GHI heads; no retraining','shard_receipts':[str(s/'completion.json') for s in shards]}
(out/'completion.json').write_text(json.dumps(r,indent=2))
print('MERGE_FULL_VALIDATION_PASS',r['GHI_rows'])
