"""Record delivered audit artifacts, including local arrays omitted from Git."""
from pathlib import Path
import hashlib,json
B=Path(__file__).parent
skip={'ARTIFACT_MANIFEST.json','ghi_pack_rows.csv','ghi_pack_manifest.json','score_console.json','numeric_audit_console.json'}
items=[]
for p in sorted(B.rglob('*')):
 if not p.is_file() or p.name in skip or '__pycache__' in p.parts:continue
 items.append({'path':p.relative_to(B).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'git_included':p.suffix not in ('.npz','.npy','.pt','.pth','.pyc')})
(B/'ARTIFACT_MANIFEST.json').write_text(json.dumps({'scope':'selected Hunan validation diagnostic and independent5percent audit; no test payload','files':items},indent=2),encoding='utf-8')
