import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;records=[]
for stage,expected in [('inner_attempt1',243),('outer_attempt1',81)]:
    files=sorted((P/stage).glob('*/*_steps.pt'));assert len(files)==expected
    for f in files:records.append(dict(path=str(f.relative_to(P)),size=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
(P/'CHECKPOINT_INDEX.json').write_text(json.dumps(dict(checkpoints=len(records),records=records),indent=2))
print('CHECKPOINT_INDEX',len(records))
