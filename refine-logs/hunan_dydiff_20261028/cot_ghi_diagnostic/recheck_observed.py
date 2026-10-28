"""Compare reconstructed real-input branch against original saved predictions."""
import csv,json
from pathlib import Path
import numpy as np
BASE=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/HunanDyDiff_20261028/cot_ghi_diagnostic/attempt1')
HEAD=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/cloud_probability_input_20261020/results/four_arm_paired_v1_attempt1')
rows=list(csv.DictReader((BASE/'ghi_rows.csv').open()));stations=np.array([r['station'] for r in rows]);records=[]
with np.load(BASE/'ghi_predictions.npz') as p:
 for s in ('sili','zhujia'):
  ids=np.flatnonzero(stations==s)
  for seed in (42,43,44):
   for arm in ('M00','M10'):
    with np.load(HEAD/f'seed{seed}'/s/arm/'validation_predictions.npz') as z:
     saved={int(k):(float(y),float(t)) for k,y,t in zip(z['pack_index'],z['prediction'],z['truth'])}
    original=np.array([saved[int(rows[i]['pack_index'])][0] for i in ids]);truth=np.array([saved[int(rows[i]['pack_index'])][1] for i in ids])
    assert np.max(abs(truth-p['truth'][ids]))<1e-3
    new=p[f'{s}_seed{seed}_{arm}_observed'][ids];delta=float(abs(new-original).max())
    records.append({'station':s,'seed':seed,'arm':arm,'max_abs_Wm2':delta,'mean_abs_Wm2':float(abs(new-original).mean())})
result={'status':'PASS' if max(r['max_abs_Wm2'] for r in records)<.1 else 'FAIL_INPUT_RECONSTRUCTION','records':records,'test_payloads_read':0}
(BASE/'observed_input_reconstruction.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
assert result['status']=='PASS'
