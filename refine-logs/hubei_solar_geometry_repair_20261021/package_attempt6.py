from pathlib import Path
import hashlib,json,tarfile
root=Path(r'F:\CODE_Classify\辐照度预报\solar-cot-ifs\refine-logs\hubei_solar_geometry_repair_20261021')
code=root/'code_pipeline_attempt6'; archive=root/'code_pipeline_attempt6.tar.gz'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
with tarfile.open(archive,'w:gz') as tf:
 for p in sorted(code.glob('*.py')): tf.add(p,arcname=f'code_pipeline_attempt6/{p.name}')
files={p.name:sha(p) for p in sorted(code.glob('*.py'))}
receipts={n:sha(root/n) for n in ['prepare_data_attempt6.pbs','MISSING_R_SOURCE_AUDIT_attempt6.json','TRANSFER_AUDIT_attempt6.json']}
manifest={'attempt':6,'code_files_sha256':files,'receipts_sha256':receipts,'code_archive_sha256':sha(archive),'preparation_archive_sha256':'7169be51bff5a20d2cfa53543528cc2914ffa6e2d931833c6b64b5c8742d38e4','r_only_source_archive_sha256':'379b4ab5b13c8af954d3d7ccf84cd9cb695ef07574d3bcb670edf6c0c46f3ed6','test_payloads_read':0,'buffer_payloads_read':0}
(root/'CODE_SHA256_attempt6.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'archive_bytes':archive.stat().st_size,'code_archive_sha256':manifest['code_archive_sha256'],'code_files':len(files),'pbs_sha256':receipts['prepare_data_attempt6.pbs']},indent=2))
