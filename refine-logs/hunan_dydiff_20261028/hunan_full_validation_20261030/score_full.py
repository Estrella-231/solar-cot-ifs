"""Paired full-validation GHI scoring and fixed lead groups; no training."""
import csv,json,subprocess,sys
from pathlib import Path
import numpy as np
from score_saved import scores,macro_boot

root=Path(sys.argv[1])
subprocess.run([sys.executable,str(Path(__file__).with_name('score_saved.py')),str(root)],check=True)
with (root/'ghi_rows.csv').open() as f:rows=list(csv.DictReader(f))
with np.load(root/'ghi_predictions.npz') as z:pred={k:z[k].copy() for k in z.files}
leads=np.array([int(r['lead_minutes']) for r in rows]);stations=np.array([r['station'] for r in rows]);dates=np.array([r['target_date_bjt'] for r in rows])
records=[];subgroups={}
for label,allowed in [('all_15_240',leads>=15),('all_forecast_slots_45_240',leads>=45)]:
    groups=[];pairs=[]
    for station in ('sili','zhujia'):
        ids=np.flatnonzero(allowed&(stations==station))
        for seed in (42,43,44):
            groups.append(ids)
            pairs.append((pred[f'{station}_seed{seed}_M00_dydiff'][ids],pred[f'{station}_seed{seed}_M10_dydiff'][ids],pred['truth'][ids]))
    rec=macro_boot(dates,groups,pairs)
    b=np.mean([scores(p[0],p[2])['rmse'] for p in pairs]);a=np.mean([scores(p[1],p[2])['rmse'] for p in pairs])
    rec.update(rows=int(allowed.sum()),before_macro_rmse_Wm2=float(b),after_macro_rmse_Wm2=float(a),relative_rmse_reduction_pct=float(100*(b-a)/b))
    subgroups[label]=rec
for station in ('sili','zhujia'):
    for lead in range(15,241,15):
        ids=np.flatnonzero((stations==station)&(leads==lead))
        if not len(ids):continue
        for seed in (42,43,44):
            for method in ('dydiff','simvp','observed'):
                for arm in ('M00','M10'):
                    records.append(dict(station=station,seed=seed,lead_minutes=lead,method=method,arm=arm,**scores(pred[f'{station}_seed{seed}_{arm}_{method}'][ids],pred['truth'][ids])))
with (root/'ghi_lead_metrics_fp64.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
(root/'lead_subgroups_fp64.json').write_text(json.dumps(subgroups,indent=2))
# Compare labels and keys to the completed full-validation replay, not its scores.
old=root.parents[1]/'full_validation_20261029/evaluation_attempt1'
previous_models=json.loads((old/'completion.json').read_text())
current_models=json.loads((root/'completion.json').read_text())
assert current_models['head_sha256']==previous_models['head_sha256']
def key(r):return (r['station'],r['init_time_utc'],r['target_time_utc'])
with (old/'ghi_rows.csv').open() as f:prior={key(r):r for r in csv.DictReader(f)}
assert len(prior)==len(rows)==63480 and len({key(r) for r in rows})==len(rows)
for r in rows:
    p=prior[key(r)]
    assert all(r[k]==p[k] for k in ('ghi','clear','lead_minutes','pack_index','target_date_bjt'))
audit={'status':'PASS','common_rows':len(rows),'labels_clear_keys_identical_to_previous_full_validation':True,'same_12_GHI_checkpoints_as_previous':True,'test_payloads_read':0}
(root/'COMMON_KEYS_CHECK.json').write_text(json.dumps(audit,indent=2))
(root/'FULL_EVALUATION_COMPLETE.json').write_text(json.dumps(dict(audit,scope='Hunan epoch150 EMA, fixed corrected R and same paired GHI heads, full validation',windows=3140),indent=2))
subprocess.run([sys.executable,str(Path(__file__).with_name('write_summary.py')),str(root)],check=True)
print('HUNAN_FULL_PAIRED_EVALUATION_COMPLETE')
