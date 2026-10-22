"""Only after six fits and internal arm freeze: prepare unchanged val inputs."""
import csv,json,sys,time
from datetime import datetime
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;BASE=P.parent;sys.path.insert(0,str(BASE))
from g4_inference import sha,geometry
from g4_motion import phase,witness
arm=json.loads((P/'fits_attempt1/ARM_SELECTION.json').read_text())
assert arm['state']=='SIX_FITS_COMPLETE_ARM_FROZEN_BEFORE_OUTER_VAL' and len(arm['runs'])==6
for run in arm['runs']:
    assert sha(Path(run['run'])/'best.pt')==run['best_sha256']
out=Path('/dev/shm/hubei_trajectory_eval_pack_20261023_attempt1');assert not out.exists();out.mkdir()
selection=json.loads((BASE/'G4_SELECTION.json').read_text());keys=sorted(r['bjt'] for r in selection['frames']);ki={k:i for i,k in enumerate(keys)}
assert all(r['split']=='val' for r in selection['frames']);start=time.monotonic();n=len(keys)
log=np.lib.format.open_memmap(out/'history_log.npy',mode='w+',dtype='float32',shape=(n,256,256))
valid=np.lib.format.open_memmap(out/'history_valid.npy',mode='w+',dtype='bool',shape=(n,256,256))
c13=np.empty((n,256,256),np.float32);seen=set();sources={}
for shard in range(4):
    source=BASE/f'g4_attempt1/history{shard}/history.npz'
    receipt=json.loads(source.with_name('COMPLETE.json').read_text())
    assert receipt['state']=='FROZEN_G4_INFERENCE_COMPLETE' and receipt['mode']=='history'
    assert receipt['r_checkpoint_sha256']=='ed7be2ea2c987557d3753b3eb7d8d0f9ffb4513b63e4afedfcbb54be132600e2'
    sources[str(source)]=sha(source)
    with np.load(source) as z:
        kk=z['keys'];ll=z['log_cot'];vv=z['valid'];ii=z['c13']
        for j,key in enumerate(kk):
            key=str(key);assert key in ki and key not in seen;seen.add(key);ix=ki[key]
            log[ix]=ll[j];valid[ix]=vv[j];c13[ix]=ii[j]
assert seen==set(keys);log.flush();valid.flush();pair_motion={};inits={};rows=[]
target_keys=sorted({r['target_time_BJT'] for r in selection['pairs']});ti={k:i for i,k in enumerate(target_keys)}
for r in selection['pairs']:
    init=r['init_time_BJT'];hist=r['history_BJT'].split('|')
    if init not in inits:
        motions=[]
        for k1,k2 in zip(hist[:-1],hist[1:]):
            if (k1,k2) not in pair_motion:pair_motion[k1,k2]=phase(c13[ki[k1]],c13[ki[k2]])
            motions.append(pair_motion[k1,k2])
        inits[init]=np.median(np.stack(motions),axis=0)
    rows.append(dict(role='outer_val',init=init,target=r['target_time_BJT'],lead=int(r['lead_minutes']),
       history_indices='|'.join(str(ki[k]) for k in hist),target_index=ti[r['target_time_BJT']],
       motion_dy=float(inits[init][0]),motion_dx=float(inits[init][1])))
del c13
grid=dict(np.load(BASE/'grid_static256.npz'))
geo=np.lib.format.open_memmap(out/'target_geometry.npy',mode='w+',dtype='float32',shape=(len(target_keys),3,256,256))
for j,k in enumerate(target_keys):geo[j]=geometry(grid,datetime.strptime(k,'%Y%m%d%H%M'))
geo.flush()
with (out/'pairs.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/'SHA256.json').write_text(json.dumps({f.name:sha(f) for f in out.iterdir() if f.is_file()},indent=2))
report=dict(state='FROZEN_OUTER_VAL_INPUTS_READY_AFTER_ARM_SELECTION',pack=str(out),history_frames=n,pairs=len(rows),
    target_frames=len(target_keys),arm_selection_sha256=sha(P/'fits_attempt1/ARM_SELECTION.json'),source_sha256=sources,
    new_CPP_payloads_read=0,new_AGRI_payloads_read=0,known_geometry_only=True,test_buffer_payloads_read=0,
    motion_witness=witness(),seconds=time.monotonic()-start)
(out/'REPORT.json').write_text(json.dumps(report,indent=2));(P/'EVAL_PACK_COMPLETE.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
