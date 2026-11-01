"""Read-only named Hunan R provenance collection. No test/buffer payloads."""
import csv, hashlib, json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907')
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20), b''): h.update(b)
    return h.hexdigest()
def collect(p, content=False):
    p = ROOT/p
    r = dict(path=str(p), exists=p.is_file())
    if r['exists']:
        r.update(sha256=sha(p), bytes=p.stat().st_size)
        if content:
            if p.suffix=='.json': r['json']=json.loads(p.read_text(encoding='utf-8-sig'))
            elif p.suffix=='.csv':
                with p.open(encoding='utf-8-sig',newline='') as f: r['rows']=list(csv.DictReader(f))
    return r
monthly='experiments/monthly_R_station16_20261013_v1'
geom='experiments/r_solar_geometry_corrected_20261015'
paths=[
 (monthly+'/data/pack_trainval_v1/'+n, n.endswith('.json'))
 for n in ['pack_status.json','rows.csv','x_raw.npy','cot.npy','mask.npy']]
paths += [(monthly+'/results/seed42_v1/'+n,n.endswith('.json')) for n in
          ['best.pt','run_metadata.json','normalization.json','artifact_sha256.json','status.json']]
paths += [(monthly+'/scripts/'+n,False) for n in ['train_station16_R.py','pack_station16_cpp_agri.py']]
paths += [(monthly+'/results/new_monthly_R_seed42_cot_bank_v1/bank_manifest.json',True),
 ('refine-logs/cot_aware_simvp_20261017/results/future_simvp_r_bank_v1/bank_manifest.json',True),
 ('experiments/r_geometry_downstream_ghi_20261016/audit/paired_r_cot_bank_manifest.json',True)]
paths += [(geom+'/results/paired_v1/'+arm+'/'+n,n.endswith('.json'))
          for arm in ['old','corrected'] for n in ['best.pt','run_metadata.json','normalization.json','artifact_hashes.json']]
paths += [(geom+'/data/'+n,n.endswith('.json')) for n in ['pack_audit.json','cot.npy','common_mask.npy','rows_supervised.csv']]
paths += [('runs/cot_repaired_seed42_v1/'+n,n.endswith('.json')) for n in ['best.pt','run_metadata.json','artifact_sha256.json']]
paths += [('data/cot_repaired_pack_20260907_v1/'+n,n.endswith('.json') or n=='rows.csv') for n in ['pack_status.json','SHA256.json','rows.csv']]
out=dict(collected_utc=datetime.now(timezone.utc).isoformat(), artifacts=[collect(p,c) for p,c in paths],
         test_buffer_payloads_read=False, raw_source_payloads_read=False)
print(json.dumps(out,ensure_ascii=True,indent=2))
