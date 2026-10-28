"""Independent full-table CPP join replay for saved selected validation patches."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
B=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/HunanDyDiff_20261028/cot_ghi_diagnostic/attempt1')
P=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/r_solar_geometry_corrected_20261015/data/paired_common_v1')
def rows(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
full=rows(P/'rows_supervised.csv');index={(r['station'],r['target_time_utc']):i for i,r in enumerate(full) if r['split']=='val'}
ref=np.load(P/'cot.npy',mmap_mode='r');mask=np.load(P/'common_mask.npy',mmap_mode='r')
assert len(full)==len(ref)==len(mask)
savedrows=rows(B/'cot_rows.csv');out=[]
with np.load(B/'cot_predictions.npz') as z:
 for j,r in enumerate(savedrows):
  key=(r['station'],r['target_time_utc']);i=index.get(key)
  if i is None:
   assert not z['cpp_mask'][j].any() and np.isnan(z['cpp'][j]).all();continue
  assert full[i]['split']=='val' and (full[i]['station'],full[i]['target_time_utc'])==key
  np.testing.assert_array_equal(z['cpp'][j],ref[i,0]);np.testing.assert_array_equal(z['cpp_mask'][j],mask[i,0])
  out.append({'window':r['window'],'station':r['station'],'target_time_utc':r['target_time_utc'],'full_table_array_index':i,'original_array_index_field':full[i]['array_index'],
   'cpp_source_path':full[i]['cpp_path'],'center_valid':bool(mask[i,0,8,8])})
with (B/'CPP_match_receipt.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
result={'status':'PASS_EXACT_CPP_REFERENCE_AND_MASK_REPLAY','matched_patches':len(out),'center_valid':sum(r['center_valid'] for r in out),
 'source_rows_sha256':sha(P/'rows_supervised.csv'),'source_cpp_array_sha256':sha(P/'cot.npy'),'source_mask_sha256':sha(P/'common_mask.npy'),
 'source_full_table_rows':len(full),'index_rule':'full supervised table position joined by station and exact UTC; val only','test_payloads_read':0}
(B/'CPP_JOIN_REPLAY.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
