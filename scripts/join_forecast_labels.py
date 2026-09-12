"""Bind every pre-approved train/val GHI label to a canonical forecast sequence.

Reads sealed CSV only; no test metrics, no CPP or observation-derived inputs.
"""
import argparse,csv,hashlib,json,math
from collections import Counter
from datetime import datetime,timedelta
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    assert sha(a.cohort)=='313153d7279cda3737286bf76000bacb0766e6cc678cef48e925aa5a5c0a1c9b'
    assert sha(a.manifest)=='9b6f5ff452a377a202db3d7dc305cebd74230cb084b6accf21df745b4b47dcf8'
    a.output.mkdir(parents=True,exist_ok=False)
    mapping={};split_count=Counter()
    with a.manifest.open(newline='') as f:
        for r in csv.DictReader(f):
            name={'val':'validation'}.get(r['split'],r['split'])
            start=datetime.fromisoformat(r['BJT_start'])
            init=start+timedelta(minutes=105)
            key=(name,init)
            assert key not in mapping
            mapping[key]=(r['seq_id'],split_count[name]);split_count[name]+=1
    seen=set();counts=Counter();used=set();skipped_test=0
    with a.cohort.open(newline='') as f,(a.output/'labels.csv').open('w',newline='') as out:
        reader=csv.DictReader(f)
        writer=csv.DictWriter(out,fieldnames=list(reader.fieldnames)+['simvp_seq_id','cache_index','station_index','lead_index'])
        writer.writeheader()
        for r in reader:
            if r['split']=='test':skipped_test+=1;continue
            assert r['split'] in ('train','validation')
            init=datetime.fromisoformat(r['init_time_bjt']);target=datetime.fromisoformat(r['target_time_bjt'])
            assert init.utcoffset()==timedelta(hours=8) and target.utcoffset()==timedelta(hours=8)
            assert init==datetime.fromisoformat(r['init_time_utc']) and target==datetime.fromisoformat(r['target_time_utc'])
            lead=int(r['lead_minutes']);assert lead in range(15,241,15) and target-init==timedelta(minutes=lead)
            times=[datetime.fromisoformat(t) for t in r['label_record_times_bjt'].split('|')]
            assert times==[target-timedelta(minutes=k) for k in (10,5,0)]
            key=(r['split'],init.replace(tzinfo=None))
            seq,index=mapping[key]
            unique=(r['station'],init,target);assert unique not in seen;seen.add(unique)
            assert r['station'] in ('sili','zhujia') and r['day_mask_sza_lt_90']=='True'
            ghi=float(r['observed_ghi_15min_wm2']);clear=float(r['clear_sky_ghi_15min_wm2']);kt=float(r['kt'])
            assert all(math.isfinite(v) for v in (ghi,clear,kt)) and clear>0 and ghi>=0 and kt>=0
            assert abs(ghi-clear*kt)<1e-8
            writer.writerow(dict(r,simvp_seq_id=seq,cache_index=index,station_index=('sili','zhujia').index(r['station']),lead_index=lead//15-1))
            counts[r['split']]+=1;used.add(key)
    assert counts=={'train':538053,'validation':99849} and skipped_test==334126
    report=dict(state='PASS_LABEL_TO_SEQUENCE_JOIN',counts=dict(counts),unique_sequences=len(used),
        sequence_counts=dict(split_count),test_rows_not_written=skipped_test,test_metrics_computed=False,
        source_cohort_sha256=sha(a.cohort),manifest_sha256=sha(a.manifest),labels_sha256=sha(a.output/'labels.csv'),
        script_sha256=sha(__file__),feature_cache_payload_validation='PENDING_COMPLETE_SHARDS',
        exclusions_added=0,CPP_filter=False,keys='station/init_time/target_time')
    (a.output/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
