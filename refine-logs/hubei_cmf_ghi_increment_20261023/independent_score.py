"""Independent math.fsum scoring of saved predictions, without training imports."""
import hashlib,json,math
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
report=json.loads((P/'scores_attempt2/SCREEN_RESULT.json').read_text())
arms=list(report['arms']);scores={k:[] for k in arms};hashes={}
for st in ['Niushou','Wangzhai','Mizhuang']:
    for seed in [42,43,44]:
        f=P/'screen_attempt2'/st/f'seed{seed}_Cselect.npz'
        with np.load(f) as z:
            assert z['arms'].tolist()==arms
            for name,pred in zip(arms,z['prediction']):
                errors=[float(x)-float(y) for x,y in zip(pred,z['truth'])]
                scores[name].append(math.sqrt(math.fsum(x*x for x in errors)/len(errors)))
        hashes[str(f.relative_to(P))]=hashlib.sha256(f.read_bytes()).hexdigest()
        c=json.loads((f.parent/f'seed{seed}_coefficients.json').read_text())
        assert c['old_prediction_maxdiff']==0
points={k:math.fsum(v)/len(v) for k,v in scores.items()}
err=max(abs(points[k]-report['arms'][k]) for k in arms); assert err<1e-10
out=dict(state='PASS_SAVED_PREDICTION_FP64_INDEPENDENT_RECALC',arms=points,max_abs_difference=err,files=hashes,
         limitation='Point scores independently recalculated; date-block intervals are produced by score.py and preserved with daily sums. No claim of new independent bootstrap implementation.',
         official_val_test_new_period_payloads_read=0)
(P/'INDEPENDENT_SCORE.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
