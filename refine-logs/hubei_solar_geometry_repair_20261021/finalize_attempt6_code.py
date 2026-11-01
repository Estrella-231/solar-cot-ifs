from pathlib import Path
root=Path(r'F:\CODE_Classify\辐照度预报\solar-cot-ifs\refine-logs\hubei_solar_geometry_repair_20261021\code_pipeline_attempt6')
for p in root.glob('*.py'):
    s=p.read_text(encoding='utf-8').replace('ATTEMPT5','ATTEMPT6')
    if p.name=='build_corrected_data.py':
        s=s.replace("assert transfer_receipt['state']=='PASS_TRANSFERRED_R_ONLY_TRAIN_VAL_SHA256' and transfer_receipt['count']==202 and transfer_receipt['all_hashes_match']",
                    "assert transfer_receipt['state']=='PASS_TRANSFERRED_R_ONLY_TRAIN_VAL_SHA256' and transfer_receipt['count']==202 and transfer_receipt['all_hashes_match'] and transfer_receipt['source_audit_sha256']==sha(a.base/'MISSING_R_SOURCE_AUDIT_attempt6.json')")
    if p.name=='audit_prepared_attempt6.py':
        s=s.replace("assert transfer_receipt['state']=='PASS_TRANSFERRED_R_ONLY_TRAIN_VAL_SHA256' and transfer_receipt['count']==202 and transfer_receipt['all_hashes_match']",
                    "assert transfer_receipt['state']=='PASS_TRANSFERRED_R_ONLY_TRAIN_VAL_SHA256' and transfer_receipt['count']==202 and transfer_receipt['all_hashes_match'] and transfer_receipt['source_audit_sha256']==sha(a.base/'MISSING_R_SOURCE_AUDIT_attempt6.json')")
    p.write_text(s,encoding='utf-8',newline='\n')
print('finalized attempt6 code')
