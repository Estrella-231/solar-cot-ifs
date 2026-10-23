import csv,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;H=P.parent/'hubei_forecast_cot_increment_20261022';G=P.parent/'hubei_cot_calibration_coverage_20261023';A=P.parent/'hubei_ghi_dual_domain_20261023';R=P.parent/'hubei_matched_R_dynamics_20261023'
cov=json.loads((G/'METADATA_AUDIT.json').read_text());out=[]
with np.load(R/'PAIRED_TRAIN_REFERENCES.npz') as z:ref={k:z[k] for k in z.files if k not in ['ghi','clear']}
lookup={(str(st),str(i),str(t),int(l),str(role)):j for j,(st,i,t,l,role) in enumerate(zip(ref['station'],ref['init_time'],ref['target_time'],ref['lead'],ref['source_role']))}
for st in ['Niushou','Wangzhai','Mizhuang']:
    p=H/'prepared_attempt1'/st;rows=list(csv.DictReader((p/'rows.csv').open()));meta=json.loads((p/'meta.json').read_text());wide=set(cov[st]['wide_dates']);take=np.array([i for i,r in enumerate(rows) if r['split']=='train' and r['target_time_bjt'][:8] in wide]);rr=[rows[i] for i in take]
    f=np.load(p/'future.npy',mmap_mode='r')[take];ff=np.load(p/'fusion_features.npy',mmap_mode='r')[take];pm=(f[:,13].reshape(-1,9)>np.cos(np.deg2rad(80)))&ff[:,297:306].astype(bool)
    with np.load(A/'prepared_attempt1'/st/'actual_training_views.npz') as z:aux={k:z[k] for k in ['cot_features','prepared_index','available']}
    am=aux['cot_features'][:,18:27].astype(bool);different=pm!=am
    ids=np.array([lookup[st,r['init_time_bjt'],r['target_time_bjt'],int(r['lead_minutes']),r['source_role']] for r in rr]);actual=ref['observed_R'][ids];pred=ref['forecast'][ids]
    positions=np.argwhere(different);examples=[]
    for i,j in positions[:12]:examples.append(dict(init=rr[i]['init_time_bjt'],target=rr[i]['target_time_bjt'],pixel=int(j),predicted_valid=bool(pm[i,j]),actual_valid=bool(am[i,j]),predicted_COT=float(pred[i,j]),actual_COT=float(actual[i,j]),cosSOZ=float(f[i,13].reshape(9)[j])))
    out.append(dict(station=st,indices_equal=bool(np.array_equal(take,aux['prepared_index'])),different_pixels=int(different.sum()),different_all9_available_rows=int((pm.all(1)!=am.all(1)).sum()),actual_above100=int((actual>100).sum()),pred_above100=int((pred>100).sum()),examples=examples))
(P/'SUPPORT_DIAGNOSTIC.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
