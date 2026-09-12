"""Freeze a transparent suspected-flatline sensitivity view; base pack is immutable."""
import argparse,csv,json,hashlib
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def moments(array,ids,spatial):
    total=np.zeros(array.shape[1],np.float64);square=total.copy();count=0
    for start in range(0,len(ids),2048):
        a=np.asarray(array[ids[start:start+2048]],np.float64)
        axis=(0,2,3) if spatial else 0
        total+=a.sum(axis);square+=(a*a).sum(axis)
        count+=len(a)*(a.shape[-1]*a.shape[-2] if spatial else 1)
    mean=total/count;std=np.sqrt(np.maximum(square/count-mean**2,0))
    assert np.isfinite(mean).all() and np.isfinite(std).all() and (std>1e-12).all()
    return dict(mean=mean.tolist(),std=std.tolist(),count=count)

def affected(row,intervals):
    records=[datetime.fromisoformat(t) for t in row['label_record_times_bjt'].split('|')]
    return any(lo<=t<=hi for lo,hi in intervals.get(row['station'],[]) for t in records)

def main():
    p=argparse.ArgumentParser();p.add_argument('--pack',type=Path,required=True);p.add_argument('--flags',type=Path,required=True)
    p.add_argument('--decision',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    audit=json.loads((a.pack/'audit.json').read_text());assert audit['state']=='COMPLETE'
    assert sha(a.pack/'rows.csv')==audit['rows_csv_sha256']
    report=json.loads(a.flags.read_text());assert report['scope']['test_numeric_values_analyzed'] is False
    intervals={}
    for r in report['flagged_runs']:
        assert r['records']>=12 and r['ghi_wm2']>0
        lo=datetime.fromisoformat(r['first_interval_end_bjt']);hi=datetime.fromisoformat(r['last_interval_end_bjt'])
        assert (hi-lo).total_seconds()/300+1==r['records']
        intervals.setdefault(r['station'],[]).append((lo,hi))
    keep=np.ones(637902,bool);counts={};target_keys=set()
    with (a.pack/'rows.csv').open(newline='') as f,(a.output/'excluded_rows.csv').open('w',newline='') as out:
        reader=csv.DictReader(f);w=csv.DictWriter(out,fieldnames=['pack_row','reason']+reader.fieldnames);w.writeheader()
        total=0
        for i,row in enumerate(reader):
            assert row['split'] in ('train','validation');total+=1
            if affected(row,intervals):
                keep[i]=False;counts[row['split']]=counts.get(row['split'],0)+1
                w.writerow(dict(pack_row=i,reason='suspected_positive_flatline_ge60min',**row))
                target_keys.add((row['station'],row['target_time_bjt']))
        assert total==len(keep)
    assert counts=={'train':448} and len(target_keys)==28
    np.save(a.output/'keep.npy',keep)
    split=np.load(a.pack/'split.npy');train=np.where(keep & (split==0))[0];val=np.where(keep & (split==1))[0]
    assert len(train)==537605 and len(val)==99849
    transforms={}
    for key in ('x','c','g'):
        item=audit['arrays'][key];path=a.pack/item['name'];assert sha(path)==item['sha256']
        array=np.load(path,mmap_mode='r')
        if key=='g':array=array[:,:2]
        transforms[key]=moments(array,train,key=='x')
    station=np.load(a.pack/'station.npy');lead=np.load(a.pack/'lead.npy')
    counts32=np.bincount(station[train]*16+lead[train],minlength=32);assert (counts32>0).all()
    for key in ('station','lead','split','kt','ghi','clear'):
        assert sha(a.pack/(key+'.npy'))==audit['arrays'][key]['sha256']
    kt=np.load(a.pack/'kt.npy')
    view=dict(state='COMPLETE_QC_SENSITIVITY_VIEW',source_pack=str(a.pack),pack_audit_sha256=sha(a.pack/'audit.json'),
        flag_audit_sha256=sha(a.flags),decision_sha256=sha(a.decision),script_sha256=sha(__file__),
        retained_counts=dict(train=len(train),validation=len(val)),excluded_counts=counts,excluded_unique_targets=len(target_keys),
        keep_sha256=sha(a.output/'keep.npy'),excluded_rows_sha256=sha(a.output/'excluded_rows.csv'),
        transforms_in_base_pack_normalized_coordinates=transforms,train_group_counts=counts32.tolist(),
        train_kt_max=float(kt[train].max()),train_kt_gt100=int((kt[train]>100).sum()),test_used=False,
        interpretation='study suspected-flatline QC sensitivity; not certified sensor fault',
        original_cohort_paired_pilot_required=True,loss_changed=False,kt_clipped=False,upstream_retrained=False)
    (a.output/'view.json').write_text(json.dumps(view,indent=2,allow_nan=False))
    print(json.dumps(view),flush=True)

if __name__=='__main__':main()
