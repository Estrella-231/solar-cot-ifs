#!/usr/bin/env python3
"""Read-only, train/validation-only R patch vs full-scene/data16 audit.

Reads only R rows.csv, the AGRI ``patch`` member of each indexed NPZ, and
canonical AGRI arrays for those same train/validation UTC keys. Never opens
GHI/COT labels from NPZ files and never visits test/buffer keys.
"""
import csv, datetime as dt, json
from pathlib import Path
import numpy as np

ROOT = Path('/home/Data_Pool_3/chenyi/Auxiliary_data/HuBei_AGRI_Preprocessed')
ROWS = Path('/tmp/hubei_R_rows_trainval_attempt2.csv')
OUT = Path('/tmp/hubei_R_full_scene_source_audit_attempt2')
CHANNELS = [0,1,2,3,4,5,8,9,10,11,12,13,14]

def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    with ROWS.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3479 and all(r['split'] in {'train','val'} for r in rows)
    assert len({r['time_bjt'] for r in rows}) == len(rows)
    rec=[]; missing=[]; exact=0; data16_exact=0; full_count=0
    chmax=np.zeros(13,np.float64); chsum=np.zeros(13,np.float64)
    for i,r in enumerate(rows):
        bjt=dt.datetime.strptime(r['time_bjt'],'%Y%m%d%H%M'); utc=bjt-dt.timedelta(hours=8)
        stamp=utc.strftime('%Y%m%d%H%M')
        rel=Path(utc.strftime('%Y/%Y%m/%Y%m%d'))/f'{stamp}.npy'
        full_path=ROOT/'data'/rel; small_path=ROOT/'data16'/rel; patch_path=Path(r['path_agri'])
        if not full_path.is_file() or not patch_path.is_file():
            missing.append(dict(index=i,time_bjt=r['time_bjt'],split=r['split'],full_scene_exists=full_path.is_file(),patch_exists=patch_path.is_file()))
            continue
        with np.load(patch_path,allow_pickle=False) as z:
            patch=np.asarray(z['patch'][CHANNELS],dtype=np.float32)
        scene=np.load(full_path,mmap_mode='r',allow_pickle=False)
        assert scene.shape==(20,256,256) and patch.shape==(13,16,16)
        crop=np.asarray(scene[CHANNELS,103:119,118:134],dtype=np.float32)
        d=np.abs(patch.astype(np.float64)-crop.astype(np.float64)); mx=d.max(axis=(1,2)); rowmax=float(mx.max())
        full_count+=1; exact+=int(rowmax<=1e-7); chmax=np.maximum(chmax,mx); chsum+=d.sum(axis=(1,2))
        small_exact=None
        if small_path.is_file():
            small=np.load(small_path,mmap_mode='r',allow_pickle=False)
            assert small.shape==(20,16,16)
            small_crop=np.asarray(small[CHANNELS],dtype=np.float32)
            small_exact=bool(np.array_equal(small_crop,crop));data16_exact+=int(small_exact)
        rec.append(dict(index=i,time_bjt=r['time_bjt'],time_utc=stamp,split=r['split'],max_abs=rowmax,
                        patch_equals_full_scene_crop=rowmax<=1e-7,data16_exists=small_path.is_file(),
                        data16_equals_full_scene_crop=small_exact,source_npz=str(patch_path),canonical_full=str(full_path)))
        if (i+1)%500==0:
            print(f'AUDITED {i+1}/{len(rows)} full={full_count} exact={exact}',flush=True)
    assert full_count+len(missing)==len(rows)
    summary=dict(state='R_SOURCE_VS_FULL_SCENE_TRAIN_VAL_AUDIT_COMPLETE',rows=len(rows),full_scene_files_found=full_count,
                 missing_files=missing,exact_rows=exact,mismatched_rows=full_count-exact,data16_exact_rows=data16_exact,
                 max_abs_by_channel_order=CHANNELS,max_abs_by_channel=chmax.tolist(),
                 mean_abs_by_channel=(chsum/max(full_count*256,1)).tolist(),
                 largest_mismatches=sorted(rec,key=lambda x:x['max_abs'],reverse=True)[:30],
                 rows_by_split={s:sum(x['split']==s for x in rec) for s in ('train','val')},
                 test_payloads_read=0,buffer_payloads_read=0,ghi_label_members_opened=0)
    (OUT/'R_SOURCE_AUDIT.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (OUT/'per_row.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rec[0]));w.writeheader();w.writerows(rec)
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
