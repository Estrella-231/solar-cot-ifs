from pathlib import Path
import shutil

root=Path(r'F:\CODE_Classify\辐照度预报\solar-cot-ifs\refine-logs\hubei_solar_geometry_repair_20261021')
src=root/'code_pipeline_attempt5'; dst=root/'code_pipeline_attempt6'
if dst.exists(): raise RuntimeError(f'refusing to overwrite {dst}')
shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__'))
for p in dst.glob('*.py'):
    s=p.read_text(encoding='utf-8')
    p.write_text(s.replace('attempt5','attempt6'),encoding='utf-8',newline='\n')
(dst/'audit_prepared_attempt5.py').rename(dst/'audit_prepared_attempt6.py')

b=dst/'build_corrected_data.py'; s=b.read_text(encoding='utf-8')
old="cot=np.stack([cp['cot'][lut[r['time_bjt']]] for r in rr]);valid=np.stack([cp['cot_valid_mask'][lut[r['time_bjt']]] for r in rr]).astype(bool);max_reconstruction=0.;source_audit=[]"
new="""cot=np.stack([cp['cot'][lut[r['time_bjt']]] for r in rr]);valid=np.stack([cp['cot_valid_mask'][lut[r['time_bjt']]] for r in rr]).astype(bool);max_reconstruction=0.;source_audit=[]
 source_receipt=json.loads((a.base/'MISSING_R_SOURCE_AUDIT_attempt6.json').read_text())
 transfer_receipt=json.loads((a.base/'TRANSFER_AUDIT_attempt6.json').read_text())
 assert source_receipt['state']=='PASS_SOURCE_R_ONLY_TRAIN_VAL' and source_receipt['count']==202
 assert transfer_receipt['state']=='PASS_TRANSFERRED_R_ONLY_TRAIN_VAL_SHA256' and transfer_receipt['count']==202 and transfer_receipt['all_hashes_match']
 canonical_lut={x['utc']:x for x in source_receipt['records']};assert len(canonical_lut)==202"""
if old not in s: raise RuntimeError('builder insertion anchor missing')
s=s.replace(old,new)
old="""rel=Path(utc.strftime('%Y/%Y%m/%Y%m%d'))/f'{u}.npy';full_path=a.canonical_data_root/rel
   if not full_path.is_file():raise FileNotFoundError(f'Canonical full AGRI missing for train/val R row {r[\"time_bjt\"]}: {full_path}')
   source_path_used=str(full_path)
   full=np.load(full_path,mmap_mode='r',allow_pickle=False)"""
new="""rel=Path(utc.strftime('%Y/%Y%m/%Y%m%d'))/f'{u}.npy';full_path=a.canonical_data_root/rel
   src=canonical_lut.get(u)
   if src is None or src['bjt']!=r['time_bjt'] or src['split']!=r['split'] or src['relative']!=rel.as_posix():raise ValueError(f'R-only source receipt does not bind row {r[\"time_bjt\"]}: {rel}')
   if not full_path.is_file():raise FileNotFoundError(f'R-only train/val AGRI missing for row {r[\"time_bjt\"]}: {full_path}')
   if sha(full_path)!=src['sha256']:raise ValueError(f'R-only source SHA-256 mismatch for {r[\"time_bjt\"]}: {full_path}')
   source_path_used=str(full_path)
   full=np.load(full_path,mmap_mode='r',allow_pickle=False)"""
if old not in s: raise RuntimeError('builder source fallback anchor missing')
s=s.replace(old,new)
old="""R_input_source_alignment_sha256=sha(out/'R_input_source_alignment.csv'),R_input_source_alignment_rows=len(source_audit),R_raw_corrected_sha256=sha(out/'R_raw_corrected_canonical_attempt6.npy'),"""
new="""R_input_source_alignment_sha256=sha(out/'R_input_source_alignment.csv'),R_input_source_alignment_rows=len(source_audit),R_only_source_audit_sha256=sha(a.base/'MISSING_R_SOURCE_AUDIT_attempt6.json'),R_only_transfer_audit_sha256=sha(a.base/'TRANSFER_AUDIT_attempt6.json'),R_only_source_count=len(canonical_lut),R_raw_corrected_sha256=sha(out/'R_raw_corrected_canonical_attempt6.npy'),"""
if old not in s: raise RuntimeError('builder final manifest anchor missing')
s=s.replace(old,new); b.write_text(s,encoding='utf-8',newline='\n')

a=dst/'audit_prepared_attempt6.py'; s=a.read_text(encoding='utf-8')
old="""assert ready['R_input_source_alignment_rows'] == 3479
    ref = json.loads((a.base / 'NOAA_ALL_TRAIN_VAL_REFERENCE_AUDIT.json').read_text())"""
new="""assert ready['R_input_source_alignment_rows'] == 3479
    source_receipt=json.loads((a.base/'MISSING_R_SOURCE_AUDIT_attempt6.json').read_text())
    transfer_receipt=json.loads((a.base/'TRANSFER_AUDIT_attempt6.json').read_text())
    assert source_receipt['state']=='PASS_SOURCE_R_ONLY_TRAIN_VAL' and source_receipt['count']==202
    assert transfer_receipt['state']=='PASS_TRANSFERRED_R_ONLY_TRAIN_VAL_SHA256' and transfer_receipt['count']==202 and transfer_receipt['all_hashes_match']
    assert ready['R_only_source_audit_sha256']==sha(a.base/'MISSING_R_SOURCE_AUDIT_attempt6.json')
    assert ready['R_only_transfer_audit_sha256']==sha(a.base/'TRANSFER_AUDIT_attempt6.json')
    ref = json.loads((a.base / 'NOAA_ALL_TRAIN_VAL_REFERENCE_AUDIT.json').read_text())"""
if old not in s: raise RuntimeError('audit receipt insertion anchor missing')
s=s.replace(old,new)
old="""assert all(float(r['R_S_max_abs']) == 0.0 for r in source_manifest if r['overlap_with_corrected_S'] == 'True')

    raw ="""
new="""assert all(float(r['R_S_max_abs']) == 0.0 for r in source_manifest if r['overlap_with_corrected_S'] == 'True')
    r_only=[r for r in source_manifest if r['overlap_with_corrected_S']=='False']
    source_lut={x['utc']:x for x in source_receipt['records']}
    assert len(r_only)==len(source_lut)==202
    assert {r['time_utc'] for r in r_only}==set(source_lut)
    for r in r_only:
        src=source_lut[r['time_utc']]
        pth=Path(r['source_path_used'])
        assert src['bjt']==r['time_bjt'] and src['split']==r['split']
        assert pth==(a.base/'r_only_data_attempt6'/src['relative'])
        assert pth.is_file() and sha(pth)==src['sha256']

    raw ="""
if old not in s: raise RuntimeError('audit source alignment anchor missing')
s=s.replace(old,new)
a.write_text(s,encoding='utf-8',newline='\n')

pbs=(root/'prepare_data_attempt5.pbs').read_text(encoding='utf-8-sig').replace('attempt5','attempt6')
pbs= pbs.replace('#PBS -N hb_solar_data6','#PBS -N hb_solar_data6')
pbs= pbs.replace('--canonical-data-root /home/Data_Pool_3/chenyi/Auxiliary_data/HuBei_AGRI_Preprocessed/data',
                 '--canonical-data-root /public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_solar_geometry_repair_20261021/r_only_data_attempt6')
pbs= pbs.replace('/audit_prepared_attempt6.py','/audit_prepared_attempt6.py')
(root/'prepare_data_attempt6.pbs').write_bytes(pbs.replace('\r\n','\n').replace('\r','\n').encode('utf-8'))
print('created',dst,'and prepare_data_attempt6.pbs')
