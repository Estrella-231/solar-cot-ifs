"""Train-internal holdout only: separate displacement from unchanged thickness."""
import csv,json,sys,time
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parent))
from g4_motion import advect
out=P/'internal_motion_diagnosis_attempt1';assert not out.exists()
pack=Path(json.loads((P/'PAIR_PACK_COMPLETE.json').read_text())['pack'])
assert json.loads((pack/'REPORT.json').read_text())['state']=='TRAIN_ONLY_TRAJECTORY_PAIR_PACK_COMPLETE_NOT_TRAINED'
rows=list(csv.DictReader((pack/'pairs.csv').open()));rows=[r for r in rows if r['role']=='select']
log=np.load(pack/'history_log.npy',mmap_mode='r');valid=np.load(pack/'history_valid.npy',mmap_mode='r')
cot=np.load(pack/'target_cot.npy',mmap_mode='r');mask=np.load(pack/'target_valid.npy',mmap_mode='r')
sites=['Niushou','Wangzhai','Mizhuang'];yy=np.array([111,112,110]);xx=np.array([124,126,128]);records=[];start=time.monotonic()
for r in rows:
    hi=int(r['history_indices'].split('|')[-1]);ti=int(r['target_index']);lead=int(r['lead'])
    dy=float(r['motion_dy']);dx=float(r['motion_dx']);coords=np.array([yy-lead/15*dy,xx-lead/15*dx])
    moved,good=advect(log[hi],valid[hi],coords)
    for j,st in enumerate(sites):
        target_good=bool(mask[ti,0,yy[j],xx[j]]);p_good=bool(valid[hi,yy[j],xx[j]])
        if target_good and p_good and good[j]:
            records.append(dict(station=st,init=r['init'],target=r['target'],lead=lead,
                reference=float(cot[ti,0,yy[j],xx[j]]),persistence=float(np.expm1(float(log[hi,yy[j],xx[j]]))),
                transport=float(np.expm1(float(moved[j]))),motion_pixels_per_frame=float(np.hypot(dy,dx)),
                projected_distance_pixels=float(np.hypot(dy,dx)*lead/15)))
ref=np.array([r['reference'] for r in records]);p=np.array([r['persistence'] for r in records]);t=np.array([r['transport'] for r in records])
def score(m):
    return dict(n=int(m.sum()),persistence_rmse=float(np.sqrt(np.mean((p[m]-ref[m])**2))),
        transport_rmse=float(np.sqrt(np.mean((t[m]-ref[m])**2))),transport_bias=float(np.mean(t[m]-ref[m])),
        transport_better_fraction=float(np.mean(abs(t[m]-ref[m])<abs(p[m]-ref[m]))))
summary={'all':score(np.ones(len(ref),bool))}
for label,m in [('near_zero',ref<1),('thin',(ref>=1)&(ref<10)),('moderate',(ref>=10)&(ref<30)),('thick',ref>=30)]:
    if m.any():summary[label]=score(m)
for st in sites:summary[st]=score(np.array([r['station']==st for r in records]))
report=dict(state='TRAIN_INTERNAL_FIXED_MOTION_DIAGNOSIS_COMPLETE',select_pairs=len(rows),common_station_records=len(records),
    score_definition='pooled station records, descriptive internal analysis; not outer validation main metric',summary=summary,
    motion_quantiles=np.quantile([r['motion_pixels_per_frame'] for r in records],[0,.25,.5,.75,1]).tolist(),
    target_dates=sorted({r['target'][:8] for r in records}),seconds=time.monotonic()-start,
    outer_validation_payloads_read=0,test_payloads_read=0,GHI_labels_read=0,new_training_runs=0)
out.mkdir();(out/'SUMMARY.json').write_text(json.dumps(report,indent=2))
with (out/'pixels.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
print(json.dumps(report,indent=2))
