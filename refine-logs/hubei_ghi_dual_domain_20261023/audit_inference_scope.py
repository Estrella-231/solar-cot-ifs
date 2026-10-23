"""Check auxiliary tensors are training-only and absent from inference closure."""
import ast,hashlib,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;source=(P/'train_station.py').read_text(encoding='utf-8');tree=ast.parse(source)
predict=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='predict')
names=sorted({n.id for n in ast.walk(predict) if isinstance(n,ast.Name)})
assert not any('actual' in n or 'aux' in n for n in names)
assert "features_arm=zero if 'no_COT' in arm else features" in source
assert "features=torch.cat([tensors['fusion_features'],base[:,None]],1).detach()" in source
assert "actual_ff=ff.copy()" in source and "actual_future=data['future'].copy()" in source
receipt=json.loads((P/'prepared_attempt1/COMPLETE.json').read_text());checks=[]
for st in ['Niushou','Wangzhai','Mizhuang']:
    meta=next(r for r in receipt['stations'] if r['station']==st);file=P/'prepared_attempt1'/st/'actual_training_views.npz'
    assert hashlib.sha256(file.read_bytes()).hexdigest()==meta['view_sha256'] and meta['C_select_views']==0
    with np.load(file) as z:d={k:z[k] for k in z.files}
    assert len(d['future'])==meta['fit_rows'] and len(set(d['prepared_index']))==meta['fit_rows']
    checks.append(dict(station=st,actual_training_rows=meta['fit_rows'],actual_Cselect_views=0,view_hash_matches=True))
(P/'INFERENCE_SCOPE_AUDIT.json').write_text(json.dumps(dict(state='PASS_SOURCE_AND_PREPARED_SCOPE',checks=checks,
    predict_names=names,trainer_sha256=hashlib.sha256((P/'train_station.py').read_bytes()).hexdigest(),
    limitation='Static closure and prepared-view provenance check; physical forecast tensors are separately checked by cold checkpoint replay',test_payloads_read=0),indent=2));print('PASS')
