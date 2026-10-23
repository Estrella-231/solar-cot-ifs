"""Recombine outer-held-out rows only after choices were frozen."""
import json
from pathlib import Path
import numpy as np
from model import load_npz
P=Path(__file__).resolve().parent;Q=P.parent/'hubei_cot_conditional_information_20261023';O=P/'assembled_attempt1';O.mkdir(exist_ok=False)
for st in ['Niushou','Wangzhai','Mizhuang']:
    d=load_npz(Q/'inputs_attempt2'/f'{st}.npz');out=O/st;out.mkdir()
    for sj,seed in enumerate([42,43,44]):
        prediction=np.full((4,len(d['truth'])),np.nan);prediction[0]=d['bases'][sj];cot=np.full((2,len(d['truth']),9),np.nan);folds=np.full(len(d['truth']),-1)
        for outer in range(3):
            folder=P/'outer_attempt1'/f'{st}_outer{outer}'
            assert len(json.loads((folder/'COMPLETE.json').read_text())['records'])==9
            for ai,arm in enumerate(['physical','task','direct_AGRI'],1):
                v=load_npz(folder/f'seed{seed}_{arm}_selected.npz');index=v['index']
                for key in ['truth','init_time','target_time','lead','available','source_role']:assert np.array_equal(v[key],d[key][index])
                prediction[ai,index]=v['prediction'];folds[index]=outer
                if ai<3:cot[ai-1,index]=v['cot_prediction']
        assert np.isfinite(prediction).all() and np.isfinite(cot).all() and (folds>=0).all()
        np.savez_compressed(out/f'seed{seed}_OOF.npz',prediction=prediction,cot_prediction=cot,truth=d['truth'],date=np.array([str(t)[:8] for t in d['target_time']]),
            arms=['G1','physical','task','direct_AGRI'],init_time=d['init_time'],target_time=d['target_time'],lead=d['lead'],available=d['available'],source_role=d['source_role'],
            cot_raw=np.expm1(d['features'][:,279:288]),cot_reference=np.expm1(d['observed_COT_features'][:,:9]),fold=folds)
(O/'COMPLETE.json').write_text(json.dumps(dict(state='NESTED_OUTER_OOF_ASSEMBLED',C_select_scored_rows=0,test_payloads_read=0),indent=2))
print('OUTER_OOF_ASSEMBLED',flush=True)
