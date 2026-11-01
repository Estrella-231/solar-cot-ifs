"""Verify full train/val time ledger and weight/array binding, without test data."""
import csv, hashlib, json, re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
MONTH='experiments/monthly_R_station16_20261013_v1'
snap=json.loads((HERE/'zjnu_snapshot.json').read_text(encoding='utf-8-sig'))
art={x['path'].split('SolarCOTIFS_20260907/')[-1]:x for x in snap['artifacts']}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()
def receipt(p): return art[p]['json']
pack=PROJECT/MONTH/'data/pack_trainval_v1'
with (pack/'rows.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
meta=receipt(MONTH+'/results/seed42_v1/run_metadata.json')
checks={}
for name,digest in meta['pack_arrays'].items():
    checks['actual_local_remote_metadata_'+name]=(sha(pack/name)==digest==art[MONTH+'/data/pack_trainval_v1/'+name]['sha256'])
checks['status_binding']=meta['pack_status_sha256']==sha(pack/'pack_status.json')==art[MONTH+'/data/pack_trainval_v1/pack_status.json']['sha256']
checks['R_checkpoint_artifact_binding']=art[MONTH+'/results/seed42_v1/best.pt']['sha256']==receipt(MONTH+'/results/seed42_v1/artifact_sha256.json')['best.pt']
checks['script_binding']=meta['training_script_sha256']==art[MONTH+'/scripts/train_station16_R.py']['sha256']
checks['normalization_binding']=meta['normalization_sha256']==art[MONTH+'/results/seed42_v1/normalization.json']['sha256']
stats=Counter(); offsets=Counter(); errors=[]
for r in rows:
    assert r['split'] in ('train','val')
    utc=datetime.strptime(r['target_time_utc'],'%Y-%m-%d %H:%M:%S')
    bjt=datetime.strptime(r['bjt'],'%Y%m%d%H%M')
    agri=datetime.strptime(Path(r['agri_path']).stem,'%Y%m%d%H%M')
    stats[r['split']+'/'+r['state']]+=1
    if agri!=utc or bjt-utc!=timedelta(hours=8): errors.append(r)
    if r['cpp_path']:
        cpp=datetime.strptime(re.search(r'FY4B_AGRI_(\d{14})',r['cpp_path']).group(1),'%Y%m%d%H%M%S')
        offsets[str((cpp-utc).total_seconds()/3600)]+=1
        if cpp!=utc: errors.append(r)
checks['full_time_ledger_aligned']=not errors
sup=[r for r in rows if r['state']=='SUPERVISED']
checks['contiguous_supervised_indices']=[int(r['array_index']) for r in sup]==list(range(len(sup)))
for s,key in [('train','train'),('val','val')]:
    ids=[int(r['array_index']) for r in sup if r['split']==s]
    checks[s+'_row_selection_hash']=hashlib.sha256('\n'.join(map(str,ids)).encode()).hexdigest()==meta[key+'_rows_sha256']
legacy=art['data/cot_repaired_pack_20260907_v1/rows.csv']['rows']
legacybad=[]
for r in legacy:
    u=datetime.fromisoformat(r['timestamp_utc'].replace('Z','+00:00'))
    b=datetime.fromisoformat(r['timestamp_bjt'])
    if u!=b or Path(r['original_path']).stem!=u.strftime('%Y%m%d%H%M') or Path(r['sidecar_path']).stem!=u.strftime('%Y%m%d%H%M'): legacybad.append(r)
checks['legacy_metadata_utc_bjt_file_stems']=not legacybad
samples=[]
for s in ('train','val'):
    times=sorted(set(r['target_time_utc'] for r in sup if r['split']==s))
    for k in (len(times)//4,3*len(times)//4):
        samples.extend(r for r in sup if r['split']==s and r['target_time_utc']==times[k])
x=np.load(pack/'x_raw.npy',mmap_mode='r'); y=np.load(pack/'cot.npy',mmap_mode='r'); m=np.load(pack/'mask.npy',mmap_mode='r')
payload=[]
for r in samples:
    i=int(r['array_index'])
    payload.append(dict(row=r,x=x[i].tolist(),cot=y[i,0].tolist(),mask=m[i,0].tolist()))
(HERE/'raw_source_sample_request.json').write_text(json.dumps(payload),encoding='utf-8')
future=receipt('refine-logs/cot_aware_simvp_20261017/results/future_simvp_r_bank_v1/bank_manifest.json')
checks['future_bank_R_binding']=future['r_checkpoint_sha256']==art[MONTH+'/results/seed42_v1/best.pt']['sha256']
checks['observed_bank_R_binding']=receipt(MONTH+'/results/new_monthly_R_seed42_cot_bank_v1/bank_manifest.json')['R_checkpoint_sha256']==future['r_checkpoint_sha256']
payload_receipt=json.loads((HERE/'checkpoint_payloads.json').read_text(encoding='utf-8-sig'))
checks['embedded_monthly_checkpoint_metadata']=payload_receipt['checkpoints'][0]['metadata']==meta
geo_receipts=payload_receipt['receipts']
geo_audit=geo_receipts[0]['json']
checks['paired_geometry_pack_retains_CPP_targets']=geo_receipts[1]['sha256']==meta['pack_arrays']['cot.npy']
checks['paired_geometry_old_checkpoint_binding']=payload_receipt['checkpoints'][1]['metadata']['pack_audit_sha256']==geo_receipts[0]['sha256']
checks['paired_geometry_corrected_checkpoint_binding']=payload_receipt['checkpoints'][2]['metadata']['pack_audit_sha256']==geo_receipts[0]['sha256']
checks['paired_geometry_common_mask_binding']=geo_receipts[2]['sha256']==geo_audit['hashes']['common_mask.npy']
legacy_ledger=json.loads((HERE/'legacy_source_ledger.json').read_text(encoding='utf-8-sig'))
checks['legacy_source_nc_times_and_sidecar_hashes']=legacy_ledger['expected_rows']==legacy_ledger['found_rows']==len(legacy) and legacy_ledger['mismatches']==0
checks['legacy_weight_to_pack_manifest']=payload_receipt['checkpoints'][3]['metadata']['pack_sha256']==receipt('data/cot_repaired_pack_20260907_v1/SHA256.json')
raw_witness=json.loads((HERE/'raw_source_witness_detail.json').read_text(encoding='utf-8-sig'))
checks['bounded_raw_AGRI_and_mask_reconstruction']=all(r['raw_x_matches_packed'] and r['raw_mask_matches_packed'] for r in raw_witness['samples'])
raw_drift=[{k:v for k,v in r.items() if k not in ('raw_cot','packed_cot','cpp_variables','cpp_global_attributes')} for r in raw_witness['samples'] if not r['raw_cot_matches_packed']]
out=dict(state='PASS' if all(checks.values()) else 'FAIL',checks=checks,ledger_rows=len(rows),cpp_available_rows=sum(offsets.values()),
         cpp_minus_agri_utc_hours=dict(offsets),supervised_rows=len(sup),states=dict(stats),time_errors=errors[:3],
         legacy_metadata_rows=len(legacy),legacy_metadata_errors=len(legacybad),
         legacy_raw_CPP_lineage=legacy_ledger,
         raw_CPP_source_drift=raw_drift,
         current_R_checkpoint_sha256=future['r_checkpoint_sha256'],samples=[r['row'] for r in payload],
         no_test_buffer_payloads_opened=True)
if out['state']=='PASS' and raw_drift:out['state']='PASS_TIME_AND_WEIGHT_LINEAGE_WITH_RAW_CPP_VERSION_DRIFT'
(HERE/'audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k!='samples'},indent=2))
