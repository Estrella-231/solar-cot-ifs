"""Recompute saved point scores using Python fsum, without analyze.py."""
import json,math
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
z=np.load(P/'PAIRED_TRAIN_REFERENCES.npz');expected=json.loads((P/'ANALYSIS.json').read_text())
out={};maxdiff=0.
for scope in ['C_fit','C_select']:
    out[scope]={}
    for name in ['persistence','forecast']:
        site_scores=[]
        for st in ['Niushou','Wangzhai','Mizhuang']:
            m=(z['station']==st)&(z['scope']==scope)&z['valid'][:,4]
            errors=z[name][m,4]-z['observed_R'][m,4]
            site_scores.append(math.sqrt(math.fsum(float(e)*float(e) for e in errors)/len(errors)))
        point=math.fsum(site_scores)/3;out[scope][name]=point
        maxdiff=max(maxdiff,abs(point-expected['summary'][scope]['all'][name+'_rmse']))
assert maxdiff<1e-10,maxdiff
(P/'INDEPENDENT_POINT_CHECK.json').write_text(json.dumps(dict(state='PASS_MATH_FSUM_POINT_SCORES',scores=out,max_abs_difference=maxdiff,note='Date-cluster CI uses analyze.py; not independently reimplemented here.'),indent=2))
print(json.dumps(out),maxdiff)
