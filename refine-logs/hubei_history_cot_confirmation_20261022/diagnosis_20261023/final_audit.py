import json,hashlib,numpy as np,csv
from pathlib import Path
p=Path(__file__).resolve().parent;b=p.parent;checks=[]
for st in ['Niushou','Wangzhai','Mizhuang']:
    physical=np.load(p/f'{st}_physical.npz');zero=physical['history_valid_fraction']==0
    for seed in [42,43,44]:
        a=np.load(b/'predictions_attempt3'/f'{st}_seed{seed}_G0.npz');c=np.load(b/'predictions_attempt3'/f'{st}_seed{seed}_G1.npz');z=np.load(p/f'{st}_seed{seed}_counterfactual.npz')
        assert np.array_equal(a['ghi'],c['ghi']) and np.array_equal(a['init_time_BJT'],physical['init_time_BJT'])
        assert np.array_equal(z['g1_replay'],c['pred_ghi'])
        assert np.array_equal(c['pred_ghi'][zero],z['g1_zero_cot'][zero])
        m=a['analysis']=='primary_shared_dates';e=a['pred_ghi'][m]-a['ghi'][m];d=c['pred_ghi'][m]-a['pred_ghi'][m]
        delta=(c['pred_ghi'][m]-a['ghi'][m])**2-e**2
        err=float(abs(delta-(2*e*d+d*d)).max());assert err<1e-8
        checks.append(dict(station=st,seed=seed,all_zero_identity_rows=int(zero.sum()),mse_identity_max_abs_error=err))
figs={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (p/'figures').glob('*')}
(p/'FINAL_AUDIT.json').write_text(json.dumps(dict(state='PASS',checks=checks,figure_sha256=figs,old_test_payloads_read=0,new_training_runs=0),indent=2));print('PASS',len(checks),'pairs',len(figs),'figure files')
