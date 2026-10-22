"""Independent math.fsum scoring of saved predictions, without training imports."""
import hashlib,json,math
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
report=json.loads((P/'scores_attempt1/SCREEN_RESULT.json').read_text())
arms=list(report['arms']);scores={k:[] for k in arms};hashes={}
for st in ['Niushou','Wangzhai','Mizhuang']:
    for seed in [42,43,44]:
        f=P/'replay'/st/f'seed{seed}_Cselect.npz'
        with np.load(f) as z:
            assert z['arms'].tolist()==arms
            for name,pred in zip(arms,z['prediction']):
                errors=[float(x)-float(y) for x,y in zip(pred,z['truth'])]
                scores[name].append(math.sqrt(math.fsum(x*x for x in errors)/len(errors)))
        hashes[str(f.relative_to(P))]=hashlib.sha256(f.read_bytes()).hexdigest()
        c=json.loads((f.parent/'COMPLETE.json').read_text())
        rec=next(v for v in c['receipts'] if v['seed']==seed)
        assert len({v['initialization_sha256'] for v in rec['arms']})==1
        assert len({v['order_sha256'] for v in rec['arms']})==1
        assert len({v['updates'] for v in rec['arms']})==1
        assert all(v['parameters']==9922 for v in rec['arms'])
        for arm in rec['arms']:
            log=json.loads((f.parent/f"seed{seed}_{arm['arm']}_epochs.json").read_text())
            assert len(log)==31 and log[0]['epoch']==0 and log[0]['updates']==0
            assert arm['select_rmse']==min(v['select_rmse'] for v in log)
points={k:math.fsum(v)/len(v) for k,v in scores.items()}
err=max(abs(points[k]-report['arms'][k]) for k in arms); assert err<1e-10
out=dict(state='PASS_SAVED_PREDICTION_FP64_INDEPENDENT_RECALC',arms=points,max_abs_difference=err,files=hashes,
         limitation='Point scores independently recalculated; date-block intervals are produced by score.py and preserved with daily sums. No claim of new independent bootstrap implementation.',
         official_val_test_new_period_payloads_read=0)
(P/'INDEPENDENT_SCORE.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
