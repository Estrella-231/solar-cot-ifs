import hashlib,json
from pathlib import Path
import numpy as np
root=Path('/home/Data_Pool_3/chenyi/Auxiliary_data/HuBei_AGRI_Preprocessed/data')
items=json.loads(Path('/tmp/missing_r_sources_attempt6.json').read_text())
records=[]
for i,item in enumerate(items):
    p=root/item['relative']
    if not p.is_file(): raise FileNotFoundError(str(p))
    a=np.load(p,mmap_mode='r',allow_pickle=False)
    if a.shape!=(20,256,256): raise ValueError((str(p),a.shape))
    if not np.isfinite(a).all(): raise ValueError('nonfinite '+str(p))
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    records.append({'bjt':item['bjt'],'utc':item['utc'],'split':item['split'],'relative':item['relative'],'shape':list(a.shape),'dtype':str(a.dtype),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
    if i%32==0: print('AUDITED',i+1,len(items),flush=True)
report={'state':'PASS_SOURCE_R_ONLY_TRAIN_VAL','count':len(records),'split_counts':{s:sum(x['split']==s for x in records) for s in ('train','val')},'records':records,'test_payloads_read':0,'buffer_payloads_read':0}
Path('/tmp/MISSING_R_SOURCE_AUDIT_attempt6.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
