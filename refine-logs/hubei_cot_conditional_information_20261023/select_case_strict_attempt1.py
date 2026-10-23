import json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;candidates=None
for st in ['Niushou','Wangzhai','Mizhuang']:
    with np.load(P/'inputs_attempt2'/f'{st}.npz') as z:init=z['init_time'];lead=z['lead']
    complete={str(t) for t in set(init) if set(lead[init==t])==set(range(15,241,15))}
    candidates=complete if candidates is None else candidates&complete
assert candidates,'No common complete initialization'
day=min(t[:8] for t in candidates);same_day=[t for t in candidates if t.startswith(day)]
chosen=min(same_day,key=lambda t:(abs(int(t[8:10])*60+int(t[10:12])-720),t))
scope=dict(init=chosen,rule='Earliest C_fit date with complete 16 leads in all three stations; initialization closest to 12:00 BJT, earlier tie; no prediction errors used',common_complete_initializations=len(candidates),target_leads=list(range(15,241,15)),usage='OOF explanatory case, not frequency estimate or independent test')
(P/'CASE_SCOPE.json').write_text(json.dumps(scope,indent=2));print(scope)
