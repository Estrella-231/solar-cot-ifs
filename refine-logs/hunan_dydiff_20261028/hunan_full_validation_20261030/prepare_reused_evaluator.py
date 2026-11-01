"""Adapt the completed four-GPU evaluator without changing R, heads or metrics."""
from pathlib import Path
import ast
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'full_validation_20261029'
for name in ('evaluate_full_gpu.py','merge_shards.py','score_saved.py'):
    text=(OLD/name).read_text(encoding='utf-8')
    if name=='evaluate_full_gpu.py':
        text=text.replace('CPU only. All input times','GPU inference. All input times')
        needle="selection=json.loads((PRED/'selection.json').read_text())"
        assert text.count(needle)==1
        text=text.replace(needle,needle+"\n source=json.loads((PRED/'completion.json').read_text());assert source['status']=='PASS' and source['checkpoint_epoch']==150 and source['test_payloads_read']==0")
        needle="'test_payloads_read':0,'script_sha256':sha(__file__)}"
        assert text.count(needle)==1
        text=text.replace(needle,"'test_payloads_read':0,'backbone_source':source,'script_sha256':sha(__file__)}")
    elif name=='merge_shards.py':
        text=text.replace('evaluation_shards2','evaluation_shards1')
        text=text.replace('fixed Hubei DyDiff and Hunan SimVP','independently trained Hunan epoch150 EMA DyDiff and Hunan SimVP')
    ast.parse(text)
    (HERE/name).write_text(text,encoding='utf-8')
records={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
(HERE/'CODE_SHA256.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps({'syntax':'PASS','files':list(records)},indent=2))
