"""Exact scan-time temporal integral weights; metadata only, no training/test reads."""
import argparse,csv,json,concurrent.futures,math
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np

def integral_weights(knots,a=-900.,b=0.):
    knots=np.asarray(knots,dtype=np.float64)
    if not np.all(np.diff(knots)>0) or knots[0]>a or knots[-1]<b:raise ValueError('scans do not bracket label interval')
    w=np.zeros(3)
    for j in range(2):
        lo=max(a,knots[j]);hi=min(b,knots[j+1])
        if hi<=lo:continue
        dt=knots[j+1]-knots[j];mid=(lo+hi)/2;span=(hi-lo)/(b-a)
        w[j]+=span*(knots[j+1]-mid)/dt;w[j+1]+=span*(mid-knots[j])/dt
    if abs(w.sum()-1)>1e-12 or w.min()<-1e-12:raise ValueError(w)
    return w

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--zero-flags',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=0);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    bank=a.root/'results/new_monthly_R_seed42_cot_bank_v1';bf=list(csv.DictReader((bank/'frame_keys.csv').open()));fr={(r['split'],r['target_time_utc']):r for r in bf};assert all(r['split'] in ('train','val') for r in bf)
    rows=list(csv.DictReader((a.root/'data/ghi_head_pack_new_r_v1/rows.csv').open()));flags={(r['station'],r['target_time_utc']):r for r in csv.DictReader(a.zero_flags.open())}
    if a.limit:
        rows=[r for st in ('sili','zhujia') for sp in ('train','val') for r in [q for q in rows if q['station']==st and q['split']==sp][:a.limit]]
    triples=[];excluded=[];needed={}
    for r in rows:
        t=datetime.fromisoformat(r['target_time_utc']);ks=[(r['split'],(t-timedelta(minutes=m)).isoformat(sep=' ')) for m in (30,15,0)]
        if not all(k in fr for k in ks):excluded.append(dict(station=r['station'],utc=r['target_time_utc'],reason='missing_same_split_context'));continue
        fs=[fr[k] for k in ks];triples.append((r,fs))
        for f in fs:needed[f['target_time_utc']]=f
    offsets={};errors=[]
    def read_scan(item):
        utc,f=item;t=datetime.fromisoformat(utc);bjt=t+timedelta(hours=8);folder=a.data_root/'aux_256'/bjt.strftime('%Y%m');stem=bjt.strftime('%Y%m%d%H%M')
        paths=[folder/(stem+'_'+d+'_aux.npz') for d in ('0','1')];paths=[p for p in paths if p.is_file()]
        if len(paths)!=1:raise ValueError(f'{utc}: ambiguous aux files {paths}')
        with np.load(paths[0]) as z:
            arr=z['obs_time_offset_s'];ss=[float(arr[tuple(map(int,f[s+'_grid_rc'].split(',')))]) for s in ('sili','zhujia')]
        if not all(math.isfinite(v) and 0<=v<900 for v in ss):raise ValueError(f'{utc}: invalid scan offsets {ss}')
        return utc,ss
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(read_scan,it):it[0] for it in needed.items()}
        for n,future in enumerate(concurrent.futures.as_completed(futures),1):
            try:k,v=future.result();offsets[k]=v
            except Exception as e:errors.append(dict(utc=futures[future],error=str(e)))
            if n%1000==0 or n==len(needed):
                (a.output/'progress.json').write_text(json.dumps(dict(state='READING_SCAN_METADATA',complete=n,total=len(needed),errors=len(errors),test_payloads_read=0)))
                print(f'SCAN_METADATA {n}/{len(needed)} errors={len(errors)}',flush=True)
    records=[];weights=[]
    for r,fs in triples:
        if any(f['target_time_utc'] not in offsets for f in fs):excluded.append(dict(station=r['station'],utc=r['target_time_utc'],reason='unreadable_scan_metadata'));continue
        si=int(r['station_index']);dt=np.array([-1800,-900,0.])+np.array([offsets[f['target_time_utc']][si] for f in fs]);w=integral_weights(dt);qc=flags[(r['station'],r['target_time_utc'])]
        rec=dict(split=r['split'],station=r['station'],target_time_utc=r['target_time_utc'],bank_indices='|'.join(f['index'] for f in fs),scan_offsets_relative_target_seconds='|'.join(f'{v:.6f}' for v in dt),weights='|'.join(f'{v:.12g}' for v in w),quality_flag=qc['quality_flag'],candidate_qc1_main_mask=qc['candidate_qc1_main_mask'],source='observed_AGRI_offline',test_payloads_read=0)
        records.append(rec);weights.append(w)
    if records:
        with (a.output/'interval_manifest.csv').open('w',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=list(records[0]));wr.writeheader();wr.writerows(records)
    np.save(a.output/'weights.npy',np.array(weights));(a.output/'scan_offsets.json').write_text(json.dumps(offsets));(a.output/'exclusions.json').write_text(json.dumps(excluded+errors,indent=2))
    checks=dict(constant_signal_error=0.,linear_signal_error=0.)
    for k in (np.array([-1616.,-716.,184.]),np.array([-1640.,-740.,160.])):
        w=integral_weights(k);checks['constant_signal_error']=max(checks['constant_signal_error'],abs(w@np.ones(3)-1));checks['linear_signal_error']=max(checks['linear_signal_error'],abs(w@k-(-450.)))
    count={f'{sp}/{st}':sum(r['split']==sp and r['station']==st for r in records) for sp in ('train','val') for st in ('sili','zhujia')}
    done=dict(state='COMPLETE_INTERVAL_METADATA' if not errors else 'INCOMPLETE_SCAN_METADATA',rows=len(records),counts=count,context_missing=len(excluded),scan_errors=len(errors),unique_scan_files=len(offsets),weight_min=np.min(weights,axis=0).tolist() if weights else [],weight_max=np.max(weights,axis=0).tolist() if weights else [],analytic_witness=checks,test_payloads_read=0,mode='metadata_only_observed_same_time_offline',limit_per_station_split=a.limit)
    (a.output/'manifest_audit.json').write_text(json.dumps(done,indent=2)+'\n');(a.output/'progress.json').write_text(json.dumps(done));print(json.dumps(done,indent=2))
if __name__=='__main__':main()
