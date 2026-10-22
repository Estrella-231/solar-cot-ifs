"""Same checkpoint is not proof of identical saved AGRI: compare actual arrays."""
import csv,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;BASE=P.parent;ROOT=BASE.parent
out=P/'existing_source_audit_attempt1/INPUT_BINDING.json';assert not out.exists()
pack=ROOT/'hubei_solar_geometry_repair_20261021/GHI_pack_attempt12'
rows=list(csv.DictReader((pack/'rows.csv').open()))
lookup={(r['station'],r['init_time_bjt'],r['target_time_bjt'],int(r['lead_minutes'])):i for i,r in enumerate(rows) if r['split']=='val'}
cases=list(csv.DictReader((P/'existing_source_audit_attempt1/pixels.csv').open()))
agri=np.load(pack/'agri_raw.npy',mmap_mode='r');geo=np.load(pack/'geometry.npy',mmap_mode='r')
selection=json.loads((BASE/'G4_SELECTION.json').read_text());inits=selection['initializations']
features={}
for shard in range(4):
    with np.load(BASE/f'g4_attempt1/forecast{shard}/forecast.npz') as z:
        for index,feature in zip(z['ids'],z['future_features']):features[inits[int(index)]]=feature
differences=[];gd=[]
for r in cases:
    st=r['station'];i=lookup[st,r['init'],r['target'],int(r['lead'])];j=int(r['lead'])//15-1;s=['Niushou','Wangzhai','Mizhuang'].index(st)
    x=features[r['init']][j,:,s]
    assert agri[i].shape==x[:13].shape
    differences.append(np.asarray(agri[i],np.float64)-x[:13])
    gd.append(np.asarray(geo[i],np.float64)-x[13:15])
d=np.array(differences);g=np.array(gd)
report=dict(state='ACTUAL_FORECAST_INPUT_ARRAYS_COMPARED',rows=len(cases),old_agri_shape=list(agri.shape),old_geometry_shape=list(geo.shape),
    agri_max_abs_by_channel=np.max(abs(d),axis=(0,2,3)).tolist(),agri_rmse_by_channel=np.sqrt(np.mean(d*d,axis=(0,2,3))).tolist(),
    agri_exact_equal=bool(np.array_equal(d,np.zeros_like(d))),geometry_max_abs=float(np.max(abs(g))),geometry_exact_equal=bool(np.array_equal(g,np.zeros_like(g))),
    note='Any nonzero input difference prevents calling prior COT score difference an R-only intervention',test_buffer_payloads_read=0)
out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
