"""Bounded Hunan readiness: complete split metadata and six real train/val frames."""
import argparse,csv,hashlib,json,time
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
from solar_geometry_noaa import compute_solar_angles

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--manifests',type=Path,required=True)
    ap.add_argument('--grid',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    assert not a.output.exists(),'Do not overwrite readiness receipt'
    manifest=a.manifests/'hunan_8to16.csv'
    assert sha(manifest)=='f6703ed5e42f1ae00aac6538b108dc4d9361197ad57ad86a0f407767aa2e4cb2'
    with manifest.open() as f: rows=list(csv.DictReader(f))
    with (a.manifests/'day_assignment.csv').open() as f:days={r['date_BJT']:r['split'] for r in csv.DictReader(f)}
    frames={s:set() for s in ['train','val','test']};counts={s:0 for s in frames}
    for row in rows:
        s=row['split'];assert s in frames;paths=row['data_relpaths'].split('|');assert len(paths)==24
        ts=[datetime.strptime(Path(p).stem,'%Y%m%d%H%M') for p in paths]
        assert all(b-c==timedelta(minutes=15) for c,b in zip(ts,ts[1:]))
        assert str(ts[0]+timedelta(hours=8))==row['BJT_start']
        assert all(days[str((t+timedelta(hours=8)).date())]==s for t in ts)
        assert all(not Path(p).is_absolute() and '..' not in Path(p).parts for p in paths)
        frames[s].update(paths);counts[s]+=1
    assert all(not frames[x]&frames[y] for x,y in [('train','val'),('train','test'),('val','test')])
    with np.load(a.grid,allow_pickle=False) as z:grid={k:z[k] for k in ['lat','lon','saa']}
    assert all(v.shape==(256,256) and np.isfinite(v).all() for v in grid.values())
    samples=[];channels=[0,1,2,3,4,5,8,9,10,11,12,13,14]
    for split in ['train','val']:
        names=sorted(frames[split])
        for index in [0,len(names)//2,len(names)-1]:
            rel=names[index];path=a.root/rel
            rec={'split':split,'relative':rel}
            try:
                raw=np.load(path,mmap_mode='r',allow_pickle=False)
                assert raw.shape==(20,256,256) and raw.dtype==np.float32,(raw.shape,str(raw.dtype))
                geoerr=float(np.nanmax(abs((raw[15]-grid['saa']+180)%360-180)))
                assert np.isfinite(geoerr) and geoerr<.02,geoerr
                utc=datetime.strptime(path.stem,'%Y%m%d%H%M');z,az,day=compute_solar_angles(grid['lat'],grid['lon'],utc)
                agri=np.asarray(raw[channels],np.float32)
                good=np.isfinite(agri)&(agri>0)
                good[:6]&=(agri[:6]<=2)&(day[None]>.5)
                good[6:]&=(agri[6:]>=50)&(agri[6:]<=400)
                rec.update(state='PASS',shape=list(raw.shape),dtype=str(raw.dtype),source_bytes=path.stat().st_size,
                    grid_max_abs_deg=geoerr,daylight_fraction=float((day>.5).mean()),
                    supported_fraction_per_channel=good.mean((1,2)).tolist(),old_daymask_difference_fraction=float(((raw[19]>.5)!=(day>.5)).mean()))
            except Exception as e:rec.update(state='FAIL',error=repr(e))
            samples.append(rec)
    result={'state':'SAMPLED_INPUT_PASS_FULL_NUMERIC_AUDIT_PENDING' if all(r['state']=='PASS' for r in samples) else 'SAMPLED_INPUT_FAIL',
        'region':'Hunan','sequence_counts':counts,'unique_frames':{k:len(v) for k,v in frames.items()},
        'estimated_trainval_raw_bytes':(len(frames['train'])+len(frames['val']))*(20*256*256*4+128),
        'manifest_sha256':sha(manifest),'grid_sha256':sha(a.grid),'samples':samples,
        'payload_frames_read':len(samples),'test_payloads_read':0,'buffer_payloads_read':0,
        'full_numeric_audit_done':False,'normalization_refit_done':False,'training_started':False}
    a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
