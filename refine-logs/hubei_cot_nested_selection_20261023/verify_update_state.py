import hashlib,json
from pathlib import Path
import torch
torch.set_num_threads(1);P=Path(__file__).resolve().parent;records=[]
for stage,expected in [('inner_attempt1',243),('outer_attempt1',81)]:
    files=sorted((P/stage).glob('*/*_steps.pt'));assert len(files)==expected
    for file in files:
        c=torch.load(file,map_location='cpu');counts=[int(s['step']) for s in c['optimizer_last']['state'].values()]
        assert c['updates']==600 and c['scheduler_last']['last_epoch']==600 and min(counts)==max(counts)==600
        assert sorted(c['models'])==[0,50,100,200,400,600]
        records.append(dict(path=str(file.relative_to(P)),parameter_states=len(counts),minimum_update=min(counts),maximum_update=max(counts)))
(P/'ACTUAL_UPDATE_AUDIT.json').write_text(json.dumps(dict(state='PASS',checkpoints=len(records),records=records),indent=2))
print('ALL_UPDATE_STATES_PASS',len(records))
