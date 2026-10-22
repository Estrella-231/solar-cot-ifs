"""CPU assembly of train-only history maps, target CPP, geometry and motion."""
import argparse,csv,json,sys,time
from collections import Counter
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;BASE=P.parent
sys.path.insert(0,str(BASE))
from g4_inference import sha,geometry
from g4_motion import phase,witness
ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
assert not a.out.exists();a.out.mkdir(parents=True)
selection=json.loads((P/'SELECTION.json').read_text());keys=sorted(r['bjt'] for r in selection['frames']);ki={k:i for i,k in enumerate(keys)}
n=len(keys);start=time.monotonic()
log=np.lib.format.open_memmap(a.out/'history_log.npy',mode='w+',dtype='float32',shape=(n,256,256))
valid=np.lib.format.open_memmap(a.out/'history_valid.npy',mode='w+',dtype='bool',shape=(n,256,256))
c13=np.empty((n,256,256),np.float32);seen=set()
for folder in sorted(a.cache.glob('shard*')):
    receipt=json.loads((folder/'COMPLETE.json').read_text());assert receipt['state']=='FROZEN_R_TRAIN_HISTORY_CACHED'
    assert sha(folder/'history.npz')==receipt['sha256']
    with np.load(folder/'history.npz') as z:
        kk=z['keys'];ll=z['log_cot'];vv=z['valid'];ii=z['c13']
        for j,k in enumerate(kk):
            k=str(k);assert k in ki and k not in seen;seen.add(k);ix=ki[k]
            log[ix]=ll[j];valid[ix]=vv[j];c13[ix]=ii[j]
assert seen==set(keys);log.flush();valid.flush()
pair_motion={}
def motion(hist):
    values=[]
    for before,after in zip(hist[:-1],hist[1:]):
        k=(before,after)
        if k not in pair_motion:pair_motion[k]=phase(c13[ki[before]],c13[ki[after]])
        values.append(pair_motion[k])
    return np.median(np.stack(values),axis=0)
inits={};pairs=[];target_keys=sorted({r['target_time_BJT'] for r in selection['pairs']});ti={k:i for i,k in enumerate(target_keys)}
for r in selection['pairs']:
    init=r['init_time_BJT'];hist=r['history_BJT'].split('|')
    assert hist[-1]==init
    if init not in inits:inits[init]=motion(hist)
    lead=int(r['lead_minutes']);assert datetime.strptime(r['target_time_BJT'],'%Y%m%d%H%M')-datetime.strptime(init,'%Y%m%d%H%M')==timedelta(minutes=lead)
    pairs.append(dict(role=r['role'],init=init,target=r['target_time_BJT'],lead=lead,
       history_indices='|'.join(str(ki[k]) for k in hist),target_index=ti[r['target_time_BJT']],
       motion_dy=float(inits[init][0]),motion_dx=float(inits[init][1])))
del c13
pack=Path('/dev/shm/hubei_regional_r_pack_20261023_attempt2')
packrows={r['time_bjt']:r for r in csv.DictReader((pack/'rows.csv').open())};oldmask=np.load(pack/'mask.npy',mmap_mode='r')
cppmeta={r['time_bjt']:r for r in json.loads((BASE/'source_ready_attempt1/frames.json').read_text())}
grid=dict(np.load(BASE/'grid_static256.npz'));m=len(target_keys)
cot=np.lib.format.open_memmap(a.out/'target_cot.npy',mode='w+',dtype='float32',shape=(m,1,256,256))
mask=np.lib.format.open_memmap(a.out/'target_valid.npy',mode='w+',dtype='bool',shape=(m,1,256,256))
geo=np.lib.format.open_memmap(a.out/'target_geometry.npy',mode='w+',dtype='float32',shape=(m,3,256,256))
for j,k in enumerate(target_keys):
    assert packrows[k]['split']==cppmeta[k]['split']=='train'
    source=BASE/'source_ready_attempt1/payload/train'/(k+'.npz');assert sha(source)==cppmeta[k]['output_sha256']
    with np.load(source) as z:reference=z['cot']
    vv=np.asarray(oldmask[int(packrows[k]['index']),0]);assert np.isfinite(reference[vv]).all()
    cot[j,0]=np.where(vv,reference,0);mask[j,0]=vv;geo[j]=geometry(grid,datetime.strptime(k,'%Y%m%d%H%M'))
cot.flush();mask.flush();geo.flush()
with (a.out/'pairs.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
(a.out/'history_keys.json').write_text(json.dumps(keys));(a.out/'target_keys.json').write_text(json.dumps(target_keys))
hashes={p.name:sha(p) for p in a.out.iterdir() if p.is_file()};(a.out/'SHA256.json').write_text(json.dumps(hashes,indent=2))
report=dict(state='TRAIN_ONLY_TRAJECTORY_PAIR_PACK_COMPLETE_NOT_TRAINED',history_frames=n,target_frames=m,
    pairs=dict(Counter(r['role'] for r in pairs)),initializations=len(inits),phase_pairs=len(pair_motion),
    R_frozen=True,source_selection_sha256=sha(P/'SELECTION.json'),motion_witness=witness(),seconds=time.monotonic()-start,
    test_buffer_payloads_read=0,outer_validation_payloads_read=0,GHI_labels_read=0)
(a.out/'REPORT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
