import csv,json
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent/'hubei_matched_R_dynamics_20261023';candidates=None;roles={}
for st in ['Niushou','Wangzhai','Mizhuang']:
    with np.load(P/'inputs_attempt2'/f'{st}.npz') as z:init=z['init_time'];lead=z['lead'];source=z['source_role']
    complete={str(t) for t in set(init)}
    for t in complete:
        rr=set(source[init==t]);assert len(rr)==1
        if t in roles:assert roles[t]==next(iter(rr))
        roles[t]=next(iter(rr))
    candidates=complete if candidates is None else candidates&complete
times={r['bjt'] for r in csv.DictReader((R/'raw_attempt1/frames.csv').open())};role_times={(r['role'],r['bjt']) for r in csv.DictReader((R/'raw_attempt1/role_frames.csv').open())}
def covered(t):
    init=datetime.strptime(t,'%Y%m%d%H%M');needed=[(init+timedelta(minutes=l)).strftime('%Y%m%d%H%M') for l in range(0,241,15)]
    return all(x in times and (roles[t],x) in role_times for x in needed)
candidates={t for t in candidates if covered(t)}
assert candidates,'No common initialization with complete satellite material'
day=min(t[:8] for t in candidates);same_day=[t for t in candidates if t.startswith(day)]
chosen=min(same_day,key=lambda t:(abs(int(t[8:10])*60+int(t[10:12])-720),t))
scope=dict(init=chosen,rule='Earliest C_fit date with scored rows at all three stations and all17 satellite/R frames; initialization closest to 12:00 BJT, earlier tie; no errors used',strict_first_attempt='No common initialization had all16 GHI labels in all3 stations; no data fabricated. Preserve partial GHI labels as gaps.',common_satellite_complete_initializations=len(candidates),target_leads=list(range(15,241,15)),usage='OOF explanatory case, not frequency estimate or independent test')
(P/'CASE_SCOPE.json').write_text(json.dumps(scope,indent=2));print(scope)
