import csv,json,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]);old=root.parent/'cot_ghi_diagnostic/attempt1'
with (old/'ghi_rows.csv').open() as f:rr=list(csv.DictReader(f))
lookup={(r['station'],r['window'],r['lead_minutes']):i for i,r in enumerate(rr)}
with np.load(old/'ghi_predictions.npz') as z:original={k:z[k].copy() for k in z.files}
maximum=0.;count=0
for rank in range(8):
 shard=root/f'gpu_smoke1/rank{rank}'
 assert json.loads((shard/'completion.json').read_text())['status']=='PASS'
 with (shard/'ghi_rows.csv').open() as f:rows=list(csv.DictReader(f))
 ix=np.array([lookup[(r['station'],r['window'],r['lead_minutes'])] for r in rows]);count+=len(rows)
 with np.load(shard/'ghi_predictions.npz') as z:
  for k in z.files:
   a=z[k];b=original[k][ix];assert np.array_equal(np.isfinite(a),np.isfinite(b))
   finite=np.isfinite(a)
   if finite.any():maximum=max(maximum,float(np.max(abs(a[finite]-b[finite]))))
assert count==432 and maximum<.1,(count,maximum)
(root/'GPU_SMOKE_PASS.json').write_text(json.dumps({'status':'PASS','rows':count,'max_abs_Wm2':maximum,'tolerance_Wm2':.1,'test_payloads_read':0}))
print('GPU_FROZEN_PIPELINE_REPLAY_PASS',maximum)
