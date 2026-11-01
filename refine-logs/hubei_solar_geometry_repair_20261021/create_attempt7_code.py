from pathlib import Path
import shutil

root=Path(r'F:\CODE_Classify\辐照度预报\solar-cot-ifs\refine-logs\hubei_solar_geometry_repair_20261021')
src=root/'code_pipeline_attempt6'; dst=root/'code_pipeline_attempt7'
if dst.exists(): raise RuntimeError(f'refusing to overwrite {dst}')
shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__'))
for p in dst.glob('*.py'):
    s=p.read_text(encoding='utf-8').replace('attempt6','attempt7').replace('ATTEMPT6','ATTEMPT7')
    s=s.replace('MISSING_R_SOURCE_AUDIT_attempt7.json','MISSING_R_SOURCE_AUDIT_attempt6.json')
    s=s.replace('TRANSFER_AUDIT_attempt7.json','TRANSFER_AUDIT_attempt6.json')
    s=s.replace('r_only_data_attempt7','r_only_data_attempt6')
    p.write_text(s,encoding='utf-8',newline='\n')
(dst/'audit_prepared_attempt6.py').rename(dst/'audit_prepared_attempt7.py')

b=dst/'build_corrected_data.py'; s=b.read_text(encoding='utf-8')
old="""source_receipt=json.loads((a.base/'MISSING_R_SOURCE_AUDIT_attempt6.json').read_text())
 transfer_receipt=json.loads((a.base/'TRANSFER_AUDIT_attempt6.json').read_text())
 assert source_receipt['state']=='PASS_SOURCE_R_ONLY_TRAIN_VAL' and source_receipt['count']==202
 assert transfer_receipt['state']=='PASS_TRANSFERRED_R_ONLY_TRAIN_VAL_SHA256' and transfer_receipt['count']==202 and transfer_receipt['all_hashes_match'] and transfer_receipt['source_audit_sha256']==sha(a.base/'MISSING_R_SOURCE_AUDIT_attempt6.json')
 canonical_lut={x['utc']:x for x in source_receipt['records']};assert len(canonical_lut)==202"""
new=old+"""
 witness_audit=json.loads((a.base/'RESTORATION_WITNESS_ARCHIVE_AUDIT_attempt7.json').read_text())
 assert witness_audit['state']=='PASS_RESTORATION_WITNESS_ARCHIVE_MATCHES_RESTORED_REPORT' and witness_audit['witness_frames']==255
 assert witness_audit['witness_archive_sha256']==sha(a.base/'restoration_requests_attempt2.tar.gz')"""
if old not in s: raise RuntimeError('attempt7 builder receipt block not found')
s=s.replace(old,new)
old="""R_only_source_audit_sha256=sha(a.base/'MISSING_R_SOURCE_AUDIT_attempt6.json'),R_only_transfer_audit_sha256=sha(a.base/'TRANSFER_AUDIT_attempt6.json'),R_only_source_count=len(canonical_lut),"""
new="""R_only_source_audit_sha256=sha(a.base/'MISSING_R_SOURCE_AUDIT_attempt6.json'),R_only_transfer_audit_sha256=sha(a.base/'TRANSFER_AUDIT_attempt6.json'),R_only_source_count=len(canonical_lut),restoration_witness_archive_audit_sha256=sha(a.base/'RESTORATION_WITNESS_ARCHIVE_AUDIT_attempt7.json'),"""
if old not in s: raise RuntimeError('attempt7 builder final receipt block not found')
s=s.replace(old,new); b.write_text(s,encoding='utf-8',newline='\n')

a=dst/'audit_prepared_attempt7.py'; s=a.read_text(encoding='utf-8')
old="""restore = json.loads((a.base / 'restored_attempt3' / 'RESTORATION_REPORT.json').read_text())
    assert restore['state'] == 'EXACT_UTC_HDF_RADIANCE_RESTORATION_COMPLETE'"""
new="""restore = json.loads((a.base / 'restored_attempt3' / 'RESTORATION_REPORT.json').read_text())
    assert restore['state'] == 'EXACT_UTC_HDF_RADIANCE_RESTORATION_COMPLETE'
    witness_audit=json.loads((a.base/'RESTORATION_WITNESS_ARCHIVE_AUDIT_attempt7.json').read_text())
    assert witness_audit['state']=='PASS_RESTORATION_WITNESS_ARCHIVE_MATCHES_RESTORED_REPORT'
    assert ready['restoration_witness_archive_audit_sha256']==sha(a.base/'RESTORATION_WITNESS_ARCHIVE_AUDIT_attempt7.json')
    source_witness=json.loads((a.preparation/'RESTORATION_REQUEST_WITNESS.json').read_text())
    witness_lut={r['utc']:r for r in source_witness['records']}
    assert source_witness['frames']==restore['frames']==255 and len(witness_lut)==255
    for r in restore['records']:
        assert witness_lut[r['utc']]['augmented_request_sha256']==r['request_sha256']"""
if old not in s: raise RuntimeError('attempt7 independent witness audit insertion point missing')
s=s.replace(old,new); a.write_text(s,encoding='utf-8',newline='\n')

pbs=(root/'prepare_data_attempt6.pbs').read_text(encoding='utf-8-sig').replace('attempt6','attempt7').replace('hb_solar_data6','hb_solar_data7')
pbs= pbs.replace('preparation_attempt7.tar','preparation_attempt6.tar').replace('r_only_sources_attempt7.tar','r_only_sources_attempt6.tar')
pbs= pbs.replace('MISSING_R_SOURCE_AUDIT_attempt7.json','MISSING_R_SOURCE_AUDIT_attempt6.json').replace('TRANSFER_AUDIT_attempt7.json','TRANSFER_AUDIT_attempt6.json')
pbs= pbs.replace('/r_only_data_attempt7','/r_only_data_attempt6')
needle='tar -xf "$B/preparation_attempt6.tar" -C "$PREP"\n'
append='''tar -xf "$B/preparation_attempt6.tar" -C "$PREP"
test "$(sha256sum "$B/restoration_requests_attempt2.tar.gz" | cut -d' ' -f1)" = 59fd880be1b5f1e56a2964f0733ad068115ebda19977ef1686e82f60a6043f8f
tar -xzf "$B/restoration_requests_attempt2.tar.gz" -C "$PREP"
'''
if needle not in pbs: raise RuntimeError('attempt7 PBS archive extraction point missing')
pbs= pbs.replace(needle,append,1)
(root/'prepare_data_attempt7.pbs').write_bytes(pbs.replace('\r\n','\n').replace('\r','\n').encode('utf-8'))
print('created isolated attempt7 source, audit, and PBS files')
