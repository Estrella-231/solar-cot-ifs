import json,time,numpy as np
from pathlib import Path
P=Path(__file__).resolve().parent;z=np.load(P/'old_CPP_reference.npz');root=Path('/tmp/hubei_cot_regional_input_20261023_attempt1/payload');records=[]
for k,s,co,mask in zip(z['time_bjt'],z['split'],z['cot'],z['mask']):
 f=root/str(s)/(str(k)+'.npz')
 if not f.exists() or time.time()-f.stat().st_mtime<120:continue
 with np.load(f) as n:
  ref=n['cot'][103:119,118:134];nm=n['mask'][103:119,118:134];common=mask[0]&nm
 if common.any():records.append(dict(time_bjt=str(k),split=str(s),pixels=int(common.sum()),max_abs=float(abs(ref[common]-co[0][common]).max()),changed_gt1e4=int((abs(ref[common]-co[0][common])>1e-4).sum())))
 if len(records)>=100:break
r=dict(state='PARTIAL_COMMON_REFERENCE_CHECK_NOT_FINAL',frames=len(records),valid_pixels=sum(x['pixels'] for x in records),max_abs=max([x['max_abs'] for x in records],default=None),changed_gt1e4=sum(x['changed_gt1e4'] for x in records),records=records,test_payloads_read=0,ghi_labels_read=0)
(Path('/home/Data_Pool/zjnu')/'hubei_cpp_partial_reference_check_20261023.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='records'}))
