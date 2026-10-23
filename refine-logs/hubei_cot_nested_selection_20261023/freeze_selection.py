"""Freeze ALL inner-only selections before any outer workers start."""
import hashlib,json
from pathlib import Path
import numpy as np
from model import load_npz
P=Path(__file__).resolve().parent;steps=[0,50,100,200,400,600];selections=[];inputs={}
assert not (P/'outer_attempt1').exists() and not (P/'SELECTION.json').exists()
for station in ['Niushou','Wangzhai','Mizhuang']:
    for outer in range(3):
        root=P/'inner_attempt1'/f'{station}_outer{outer}'
        assert len(json.loads((root/'COMPLETE.json').read_text())['records'])==27
        for seed in [42,43,44]:
            for arm in ['physical','task','direct_AGRI']:
                sums=np.zeros(6);count=0;seen=[]
                for inner in range(3):
                    file=root/f'seed{seed}_inner{inner}_{arm}_predictions.npz';d=load_npz(file)
                    assert d['steps'].tolist()==steps
                    sums+=((d['prediction']-d['truth'][None])**2).sum(1);count+=len(d['truth']);seen.extend(d['index'].tolist())
                    inputs[str(file.relative_to(P))]=hashlib.sha256(file.read_bytes()).hexdigest()
                assert len(seen)==len(set(seen))
                mse=sums/count;chosen=int(np.argmin(mse))
                selections.append(dict(station=station,seed=seed,outer_fold=outer,arm=arm,step=steps[chosen],inner_mse=mse.tolist(),inner_rows=count))
(P/'SELECTION.json').write_text(json.dumps(dict(state='ALL_INNER_SELECTIONS_FROZEN_BEFORE_OUTER_SCORING',selections=selections,
    inner_prediction_sha256=inputs,test_payloads_read=0),indent=2))
print('SELECTION_FROZEN',len(selections),flush=True)
