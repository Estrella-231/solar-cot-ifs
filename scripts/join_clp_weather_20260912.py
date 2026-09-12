"""Fixed validation target CLP/kt cross-tab; descriptive only, no model inputs."""
import argparse,datetime,hashlib,json,signal,time
from pathlib import Path
import numpy as np

def timeout(*_):raise TimeoutError('10s per sidecar deadline')
def main():
    p=argparse.ArgumentParser();p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--sidecars',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    assert a.sidecars.is_dir(), 'sidecar root not readable'
    selection=json.loads(a.selection.read_text());assert len(selection)==256
    signal.signal(signal.SIGALRM,timeout)
    results=[];started=time.monotonic()
    for row in selection:
        r=dict(row)
        if time.monotonic()-started>180:
            r.update(state='NOT_READ_TOTAL_TIME_BUDGET');results.append(r);continue
        dt=datetime.datetime.fromisoformat(row['target_time_utc'])
        assert dt.utcoffset()==datetime.timedelta(0)
        rel=Path(row['station'])/dt.strftime('%Y/%Y%m/%Y%m%d/%Y%m%d%H%M.npz')
        path=a.sidecars/'samples'/rel;r['sidecar']=str(path)
        signal.alarm(10)
        try:
            b=path.read_bytes();r['sha256']=hashlib.sha256(b).hexdigest()
            import io
            with np.load(io.BytesIO(b),allow_pickle=False) as z:
                assert list(z['channel_names'])==['COT','CER','CTH','CLP']
                assert str(z['station_id'])==row['station']
                stamp=datetime.datetime.fromisoformat(str(z['timestamp_utc']).replace('Z','+00:00'))
                assert stamp==dt
                if not bool(z['cpp_present']):r['state']='SOURCE_MISSING'
                else:
                    clp=z['cpp'][3];assert clp.shape==(16,16)
                    valid=np.isfinite(clp)&np.isin(clp,[0,1,2])
                    counts=[int(((clp==k)&valid).sum()) for k in range(3)]
                    r.update(state='READ',valid_pixels=int(valid.sum()),class_counts=counts,
                        center_clp=int(clp[8,8]) if valid[8,8] else None,
                        cloudy_fraction=float((counts[1]+counts[2])/valid.sum()) if valid.any() else None,
                        patch_class=(['clear','water','ice'][int(np.argmax(counts))] if valid.sum()>=128 and max(counts)>valid.sum()/2 else 'mixed_or_low_coverage'))
        except Exception as e:r.update(state='READ_ERROR',error=repr(e))
        finally:signal.alarm(0)
        results.append(r)
    table={}
    for r in results:
        label=r.get('patch_class',r['state']);key=f"kt{r['cluster']}:{label}"
        table[key]=table.get(key,0)+1
    report=dict(state='COMPLETE_BOUNDED_CLP_JOIN',test_used=False,deployment_input=False,
        selection_sha256=hashlib.sha256(a.selection.read_bytes()).hexdigest(),samples=results,cross_tab=table,
        semantics='CLP 0 no-cloud/clear; 1 water; 2 ice; majority patch needs >=128 valid pixels and >50 percent agreement; missing remains separate',
        caveat='instantaneous target CLP versus 15-minute interval kt; spatial patch differs from point radiation; not weather ground truth')
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False))
if __name__=='__main__':main()
