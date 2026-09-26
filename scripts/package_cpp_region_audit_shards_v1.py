"""Verify and deterministically package completed regional CPP audit frames."""
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--times',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--shard-size',type=int,default=128)
a=p.parse_args()
items=json.loads(a.times.read_text())
report=json.loads((a.root/'report.json').read_text())
identity=json.loads((a.root/'identity.json').read_text())
assert len(items)==1000 and len(report['frames'])==1000
bytime={x['utc']:x for x in report['frames']}
assert set(items)==set(bytime)
a.output.mkdir(parents=True,exist_ok=False)
shards=[]
for start in range(0,len(items),a.shard_size):
    times=items[start:start+a.shard_size]
    arrays={k:[] for k in ('cot','clp','observation_valid','valid')}
    source=[]
    for stamp in times:
        tag=stamp.replace('-','').replace(':','').replace('T','')
        path=a.root/(tag+'.npz'); rec=bytime[stamp]
        assert digest(path)==rec['output_sha256']
        with np.load(path,allow_pickle=False) as z:
            assert z['cot'].shape==(256,256) and z['valid'].shape==(256,256)
            expected=z['observation_valid'] & np.isfinite(z['cot']) & (z['cot']>=0) & (z['cot']<=100) & (z['clp']<3)
            assert np.array_equal(z['valid'],expected)
            for k in arrays: arrays[k].append(z[k])
        source.append({'utc':stamp,'frame_sha256':rec['output_sha256'],'valid_fraction':rec['valid_fraction']})
    name=f'cpp_region_{start:04d}_{start+len(times):04d}.npz'
    path=a.output/name; tmp=path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,utc=np.asarray(times),**{k:np.stack(v) for k,v in arrays.items()})
    os.replace(tmp,path)
    shards.append({'file':name,'frames':len(times),'sha256':digest(path),'source_frames':source})
    print(json.dumps({'shard':name,'frames':len(times),'bytes':path.stat().st_size}),flush=True)
manifest={'state':'COMPLETE_AUDIT_SHARDS','test_used':False,'producer_identity':identity,
          'times_sha256':digest(a.times),'frames':len(items),'shard_size':a.shard_size,'shards':shards}
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
assert sum(s['frames'] for s in shards)==1000
print(json.dumps({'state':manifest['state'],'shards':len(shards),'frames':1000}),flush=True)
