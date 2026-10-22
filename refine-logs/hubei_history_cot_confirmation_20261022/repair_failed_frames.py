"""Keep successful frames immutable; retry only explicit failed new HDF keys."""
import csv,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import netCDF4 as nc
ROOT=Path('/home/Data_Pool/zjnu');OLD=Path('/tmp/hubei_history_cot_new_period_full_20261022_attempt1')
OUT=Path('/tmp/hubei_history_cot_new_period_ready_20261022_attempt2')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    OUT.mkdir(exist_ok=False);(OUT/'data').mkdir();(OUT/'aux').mkdir()
    prior=json.loads((OLD/'PREPROCESS_REPORT.json').read_text());assert prior['expected']==2025
    good=list(csv.DictReader((OLD/'frame_manifest.csv').open()));audit=[];new=[];failed=[]
    # No old numerical file is changed. New directory links the checked results.
    for r in good:
        for kind in ['data','aux']:
            source=Path(r[kind+'_path']);assert sha(source)==r[kind+'_sha256']
            (OUT/kind/source.name).symlink_to(source)
        new.append(r)
    rows=list(csv.DictReader((ROOT/'hubei_history_cot_confirmation_20261022_inventory_attempt1/hdf_files.csv').open()))
    for number,bad in enumerate(prior['failures']):
        original=next(r for r in rows if r['path']==bad['source']);time_bjt=original['time_BJT']
        candidates=sorted([r for r in rows if r['time_BJT']==time_bjt],key=lambda r:('real_time_data' not in r['path'],r['path']))
        recovered=None;attempts=[]
        for r in candidates:
            for retry in range(3):
                trial=OUT/f'read_{number}_{len(attempts)}';trial.mkdir()
                inventory=trial/'one.csv'
                with inventory.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(r));w.writeheader();w.writerow(r)
                cmd=[sys.executable,ROOT/'preprocess_new_agri.py','--inventory',inventory,
                     '--grid',ROOT/'grid_static256.npz','--existing-processor',ROOT/'preprocess_hubei_full256_trainval_missing.py',
                     '--out',trial/'output','--workers','1']
                with (trial/'run.log').open('w') as f:
                    p=subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,timeout=120)
                rec=dict(path=r['path'],trial=str(trial),exit_code=p.returncode)
                if p.returncode==0:
                    recovered=next(csv.DictReader((trial/'output/frame_manifest.csv').open()));rec['state']='RECOVERED'
                else:
                    rep=json.loads((trial/'output/PREPROCESS_REPORT.json').read_text());rec['failure']=rep['failures']
                attempts.append(rec)
                if recovered:break
                time.sleep(.5*(retry+1))
            if recovered:break
        audit.append(dict(time_BJT=time_bjt,attempts=attempts,recovered=bool(recovered)))
        if recovered:
            new.append(recovered)
            for kind in ['data','aux']:
                source=Path(recovered[kind+'_path']);assert sha(source)==recovered[kind+'_sha256']
                (OUT/kind/source.name).symlink_to(source)
        else:failed.append(dict(time_BJT=time_bjt,reason='all explicit same-UTC source read attempts failed'))
        print(json.dumps(audit[-1]),flush=True)
    new.sort(key=lambda r:r['time_BJT'])
    for r in new:
        for kind in ['data','aux']:r[kind+'_path']=str(OUT/kind/Path(r[kind+'_path']).name)
    with (OUT/'frame_manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(new[0]));w.writeheader();w.writerows(new)
    report=dict(state='READY_WITH_EXPLICIT_SOURCE_GAPS' if failed else 'COMPLETE_ALL_REQUESTED_FRAMES',
        expected=prior['expected'],written=len(new),unreadable=failed,retry_audit=audit,
        prior_report_sha256=sha(OLD/'PREPROCESS_REPORT.json'),frame_manifest_sha256=sha(OUT/'frame_manifest.csv'),
        policy='No channel renaming or interpolation; unresolved HDFs invalidate all history windows needing them, equally for both models; fixed before model scoring',
        old_test_payloads_read=0,old_buffer_payloads_read=0,new_period_scores_computed=0)
    (OUT/'PREPROCESS_REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
