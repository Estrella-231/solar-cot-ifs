#!/usr/bin/env python3
"""Select deterministic paired R validation examples and package real AGRI."""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--paired-r',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    pack=a.paired_r/'data/paired_common_v1'; pred=a.paired_r/'results/paired_v1'
    rows=list(csv.DictReader((pack/'rows_supervised.csv').open(newline='',encoding='utf-8')))
    old=np.load(pred/'old/validation_predictions.npz',allow_pickle=False)
    new=np.load(pred/'corrected/validation_predictions.npz',allow_pickle=False)
    if not np.array_equal(old['array_indices'],new['array_indices']): raise ValueError('R validation row keys differ')
    ids=old['array_indices'].astype(int)
    if len(ids)!=1398 or any(rows[i]['split']!='val' for i in ids): raise ValueError('unexpected R val support')
    if not np.array_equal(old['reference_cot'],new['reference_cot']) or not np.array_equal(old['mask'],new['mask']): raise ValueError('R pair labels/support differ')
    y=np.asarray(old['reference_cot'],np.float64)[:,0]
    po=np.expm1(np.asarray(old['pred_log1p_cot'],np.float64)[:,0])
    pn=np.expm1(np.asarray(new['pred_log1p_cot'],np.float64)[:,0])
    m=np.asarray(old['mask'],bool)[:,0]
    loc=(8,8); choices=[]
    for station in ('sili','zhujia'):
        q=[j for j,i in enumerate(ids) if rows[i]['station']==station and m[(j,)+loc] and np.isfinite(y[(j,)+loc])]
        if not q: raise ValueError('empty station validation support: '+station)
        clear=[j for j in q if y[(j,)+loc]<0.1]
        thick=[j for j in q if y[(j,)+loc]>=30]
        chosen=[]
        for label,cands,score in [
            ('clear_reference_min',clear or q,lambda j:y[(j,)+loc]),
            ('thick_reference_max',thick or q,lambda j:-y[(j,)+loc]),
            ('largest_R_disagreement',q,lambda j:-abs(po[(j,)+loc]-pn[(j,)+loc]))]:
            remain=[j for j in cands if j not in chosen]
            j=min(remain or cands,key=score); chosen.append(j)
            choices.append((station,label,j))
    selected=[x[2] for x in choices]
    xraw=np.load(pack/'x_old_geometry.npy',mmap_mode='r',allow_pickle=False)
    payload=dict(rgb_c01_c02_c03=np.asarray(xraw[ids[selected],:3],np.float32),
                 cpp_reference=np.asarray(y[selected],np.float32),
                 old_r=np.asarray(po[selected],np.float32),corrected_r=np.asarray(pn[selected],np.float32),
                 common_mask=np.asarray(m[selected],np.uint8),array_indices=np.asarray(ids[selected],np.int64))
    a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**payload)
    meta=[]
    for (station,label,j),idx in zip(choices,ids[selected]):
        r=rows[int(idx)]
        meta.append(dict(station=station,selection_rule=label,target_time_utc=r['target_time_utc'],bjt=r['bjt'],
                         array_index=int(idx),reference_center=float(y[(j,)+loc]),old_R_center=float(po[(j,)+loc]),
                         corrected_R_center=float(pn[(j,)+loc]),absolute_center_difference=float(abs(po[(j,)+loc]-pn[(j,)+loc]))))
    side=dict(state='PASS_REAL_AGRI_PAIRED_R_VAL_CASES',selection='per station: minimum near-zero CPP center, maximum thick CPP center, and largest old-vs-corrected R center difference; deterministic, validation only',
              cases=meta,rgb_mapping='display RGB=[AGRI C03,C02,C01]',rgb_channels_in_pack='first three physical AGRI channels are C01,C02,C03',
              prediction_transformation='expm1(saved log1p(COT))',test_payloads_read=0,
              source_hashes={str(z):sha(z) for z in [pack/'rows_supervised.csv',pack/'x_old_geometry.npy',pred/'old/validation_predictions.npz',pred/'corrected/validation_predictions.npz']},
              payload_sha256=sha(a.output))
    a.output.with_suffix('.json').write_text(json.dumps(side,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(side,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
