"""Check the original bank manifest field names explicitly."""
import hashlib,json
from pathlib import Path
S=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907')
E=S/'experiments/regional_station_cot_20260925'
O=S/'experiments/history_cot_forecast_ghi_20261015_v1'
names=['sili_cot_dynamics_bank_train512_val512_20260927_v1','regional_ghi_pilot_bank_20260927_v1','regional_ghi_pilot_bank_val512_20260927_v1']
data=[json.loads((E/n/'complete.json').read_text()) for n in names]
keys=['manifest_sha256','backbone_checkpoint_sha256','regional_R_checkpoint_sha256','r_normalization_sha256','base_pack_audit_sha256']
for key in keys:assert len({d[key] for d in data})==1,key
assert all(d['test_used'] is False for d in data)
actual=hashlib.sha256((S/'data/head_pack_trainval_20260908_v1/audit.json').read_bytes()).hexdigest()
assert actual==data[0]['base_pack_audit_sha256']
out=dict(state='PASS_S_R_SPLIT_NORMALIZATION_LABEL_SOURCE_MATCH',bank_names=names,source_hashes={k:data[0][k] for k in keys},test_payload_read=False)
(O/'source_manifests_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
