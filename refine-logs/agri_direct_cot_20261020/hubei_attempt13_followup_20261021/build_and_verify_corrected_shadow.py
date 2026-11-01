"""Build an isolated corrected-geometry CPP sidecar; never activate it."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime,timedelta
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import numpy as np

BUILDER_SHA='1998efda2beaa86d72fe529de38db63eca3712e8e7f3a7b584d06986d284141f'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def stamp(s):
    q=re.sub(r'\D','',s)
    if len(q)!=12:raise ValueError(s)
    datetime.strptime(q,'%Y%m%d%H%M')
    return q

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--attempt',type=Path,required=True)
    ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--builder',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    assert sha(a.builder)==BUILDER_SHA
    evidence=json.loads(a.audit.read_text());meta_path=a.attempt/'prepared_inputs.json'
    meta=json.loads(meta_path.read_text());prepared=Path(meta['prepared_dir']);pack=prepared/'R_packs/full'
    active=Path(meta['sidecar']);active_manifest=json.loads((active/'manifest.json').read_text())
    corrected_status=json.loads((pack/'pack_status.json').read_text())
    assert corrected_status['solar_geometry_corrected'] and corrected_status['corrected_SZA80_mask']
    assert corrected_status['test_payloads_read']==corrected_status['buffer_payloads_read']==0
    assert str(pack)==evidence['source_pack_paths']['corrected_solar']
    assert sha(meta_path)==evidence['sources_sha256']['prepared_inputs']
    assert sha(Path(meta['index']))==evidence['sources_sha256']['index']
    assert Path(meta['index'])==prepared/'indexes/full.csv'
    for name,field in [('rows.csv','new_rows'),('target.npy','new_target'),('mask.npy','new_mask'),
                       ('x_norm.npy','new_x_norm'),('norm.json','new_norm'),('pack_status.json','new_status')]:
        assert sha(pack/name)==evidence['sources_sha256'][field],name
    watched={'prepared_inputs':meta_path,'index':Path(meta['index']),'side_manifest':active/'manifest.json'}
    for split in ('train','val'):
        for suffix in ('keys.csv','target_physical.npy','mask.npy'):
            watched['side_'+split+'_'+suffix]=active/(split+'_'+suffix)
    before={k:sha(v) for k,v in watched.items()}
    assert all(before[k]==evidence['sources_sha256'][k] for k in before)
    proc=subprocess.run([sys.executable,str(a.builder),'--prepared',str(prepared),'--r-pack',str(pack),
                         '--output',str(a.output)],capture_output=True,text=True)
    if proc.returncode:raise RuntimeError(proc.stdout+'\n'+proc.stderr)
    generated=json.loads((a.output/'manifest.json').read_text())
    assert generated['R_pack_sha256']==sha(pack/'pack_status.json')
    assert generated['R_mask_sha256']==sha(pack/'mask.npy')
    rrows=rows(pack/'rows.csv');bytime={stamp(r['time_bjt']):i for i,r in enumerate(rrows)}
    assert len(bytime)==len(rrows)==3479
    refmask=np.load(pack/'mask.npy',mmap_mode='r',allow_pickle=False)
    reftarget=np.load(pack/'target.npy',mmap_mode='r',allow_pickle=False)
    xn=np.load(pack/'x_norm.npy',mmap_mode='r',allow_pickle=False);norm=json.loads((pack/'norm.json').read_text())
    geometry=np.asarray(xn[:,13:16],np.float32)*np.asarray(norm['std'][13:16],np.float32)[None,:,None,None]+np.asarray(norm['mean'][13:16],np.float32)[None,:,None,None]
    assert refmask.dtype==np.bool_ and refmask.shape==(3479,1,16,16)
    cos=geometry[:,0:1];day=geometry[:,2:3]
    assert not (refmask&((cos<=math.cos(math.radians(80)))|(day<=.5))).any()
    idx=rows(prepared/'indexes/full.csv');assert Counter(r['split'] for r in idx)=={'train':4086,'val':407}
    stats={};detail=[]
    for split,n in [('train',4086),('val',407)]:
        oldkeys=rows(active/(split+'_keys.csv'));newkeys=rows(a.output/(split+'_keys.csv'))
        oldmask=np.load(active/(split+'_mask.npy'),mmap_mode='r',allow_pickle=False)
        newmask=np.load(a.output/(split+'_mask.npy'),mmap_mode='r',allow_pickle=False)
        target=np.load(a.output/(split+'_target_physical.npy'),mmap_mode='r',allow_pickle=False)
        assert newmask.dtype==oldmask.dtype==np.bool_
        assert newmask.shape==target.shape==(n,16,1,16,16)
        assert np.isfinite(target).all() and target.min()>=0 and target.max()<=100.0001
        counter=Counter();matched=0
        sequences=[r for r in idx if r['split']==split]
        assert len(newkeys)==len(oldkeys)==len(sequences)==n
        for si,(ok,nk,seq) in enumerate(zip(oldkeys,newkeys,sequences)):
            for field in ('sample_index','split','init_aux_bjt','target_times_bjt','matched_cpp_frames'):
                assert ok[field]==nk[field],(si,field)
            assert int(nk['sample_index'])==si and nk['split']==split
            dp=seq['data_paths'].split('|');aux=seq['aux_paths'].split('|')
            utc=[datetime.strptime(Path(x).stem,'%Y%m%d%H%M') for x in dp]
            bjt=[datetime.strptime(Path(x).name[:12],'%Y%m%d%H%M') for x in aux]
            assert len(utc)==len(bjt)==24 and [t+timedelta(hours=8) for t in utc]==bjt
            assert all(bjt[k+1]-bjt[k]==timedelta(minutes=15) for k in range(23))
            assert stamp(nk['init_aux_bjt'])==bjt[7].strftime('%Y%m%d%H%M')
            keys=nk['target_times_bjt'].split('|');assert keys==[t.strftime('%Y%m%d%H%M') for t in bjt[8:]]
            for li,t in enumerate(keys):
                ri=bytime.get(t)
                if ri is None:
                    assert not newmask[si,li].any() and not target[si,li].any()
                    continue
                assert rrows[ri]['split']==split
                assert np.array_equal(newmask[si,li],refmask[ri])
                assert np.array_equal(target[si,li],np.asarray(reftarget[ri],np.float32)*np.float32(100))
                matched+=1
        for li in range(16):
            old=np.asarray(oldmask[:,li]);new=np.asarray(newmask[:,li])
            counts={'old_valid':int(old.sum()),'corrected_valid':int(new.sum()),'same_valid':int((old&new).sum()),
                    'old_only':int((old&~new).sum()),'corrected_only':int((new&~old).sum())}
            detail.append({'split':split,'lead_minutes':15*(li+1),**counts});counter.update(counts)
        expected=evidence['repeated_sidecar_target_stats'][split]
        assert matched==expected['matched_forecast_targets']
        assert all(counter[k]==expected[k] for k in counter)
        stats[split]={'matched_frames':matched,**dict(counter),'source_geometry_violations':0}
    after={k:sha(v) for k,v in watched.items()};assert before==after
    with (a.output/'support_changes_by_lead.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(detail[0]));w.writeheader();w.writerows(detail)
    result={'state':'PASS_CORRECTED_GEOMETRY_SHADOW_SIDECAR_NOT_ACTIVATED','stats':stats,
            'source_pack':str(pack),'source_pack_sha256':sha(pack/'pack_status.json'),
            'index_sha256':sha(prepared/'indexes/full.csv'),'shadow_manifest_sha256':sha(a.output/'manifest.json'),
            'source_audit_sha256':sha(a.audit),'builder_sha256':BUILDER_SHA,'verification_script_sha256':sha(Path(__file__)),
            'plan_sha256':sha(Path(__file__).with_name('SHADOW_SUPPORT_REPAIR_PLAN.md')),
            'active_files_before_after_sha256_equal':before==after,'active_file_hashes':after,
            'future_target_keys_equal_old':True,'all_targets_match_corrected_pack':True,
            'geometry_violations':0,'activation_allowed':False,'training_started':False,'ghi_scored':False,
            'test_payloads_read':0,'buffer_payloads_read':0,
            'note':'Old mask also affects lambda calibration and checkpoint selection; do not hot-replace active inputs.'}
    (a.output/'shadow_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'state':result['state'],'stats':stats,'active_files_unchanged':before==after},indent=2))

if __name__=='__main__':main()
