"""Read exactly the frozen trainval list, with per-file hashes and explicit missingness."""
import argparse, datetime, hashlib, io, json, signal, time
from pathlib import Path
import numpy as np

def timeout(*_):raise TimeoutError('10 second source read deadline')
def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
    root=Path('/home/Data_Pool_3/wangyc/irradiance_paper/solar_cot_ifs_20260905/data/cpp_aligned_20260905_v2/samples')
    rows=json.loads((a.directory/'targets.json').read_text());assert all(r['split'] in (0,1) for r in rows)
    assert root.is_dir();signal.signal(signal.SIGALRM,timeout)
    dest=a.directory/'labels.jsonl';assert not dest.exists()
    started=time.monotonic();counts={}
    with dest.open('x',buffering=1) as f:
        for i,r in enumerate(rows):
            r=dict(r);dt=datetime.datetime.fromisoformat(r['target_time_utc'])
            assert dt.utcoffset()==datetime.timedelta(0)
            path=root/r['station']/dt.strftime('%Y/%Y%m/%Y%m%d/%Y%m%d%H%M.npz')
            r.update(path=str(path),center_clp=-1,patch_class=-1)
            signal.alarm(10)
            try:
                b=path.read_bytes();r['sha256']=hashlib.sha256(b).hexdigest()
                with np.load(io.BytesIO(b),allow_pickle=False) as z:
                    assert list(z['channel_names'])==['COT','CER','CTH','CLP']
                    assert str(z['station_id'])==r['station']
                    assert datetime.datetime.fromisoformat(str(z['timestamp_utc']).replace('Z','+00:00'))==dt
                    if not bool(z['cpp_present']):r['state']='SOURCE_MISSING'
                    else:
                        c=z['cpp'][3];assert c.shape==(16,16)
                        valid=np.isfinite(c)&np.isin(c,[0,1,2]);n=int(valid.sum())
                        counts3=[int(((c==k)&valid).sum()) for k in range(3)]
                        r.update(state='READ',center_clp=int(c[8,8]) if valid[8,8] else -1,valid_pixels=n,class_counts=counts3,
                            patch_class=int(np.argmax(counts3)) if n>=128 and max(counts3)>n/2 else -1)
            except FileNotFoundError:r['state']='FILE_MISSING'
            except Exception as e:r.update(state='READ_ERROR',error=repr(e))
            finally:signal.alarm(0)
            counts[r['state']]=counts.get(r['state'],0)+1;f.write(json.dumps(r)+'\n')
            if (i+1)%1000==0:print(json.dumps(dict(done=i+1,total=len(rows),seconds=time.monotonic()-started,counts=counts)),flush=True)
    meta=dict(state='COMPLETE_CLP_LABEL_READ',test_used=False,targets_sha256=hashlib.sha256((a.directory/'targets.json').read_bytes()).hexdigest(),labels_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),total=len(rows),counts=counts,seconds=time.monotonic()-started)
    (a.directory/'collection.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
if __name__=='__main__':main()
