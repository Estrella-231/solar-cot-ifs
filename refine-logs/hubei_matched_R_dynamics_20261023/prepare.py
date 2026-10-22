import csv,hashlib,json
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parent;B=ROOT/'hubei_solar_geometry_repair_20261021';H=ROOT/'hubei_forecast_cot_increment_20261022/prepared_attempt1'
coverage=json.loads((ROOT/'hubei_cot_calibration_coverage_20261023/METADATA_AUDIT.json').read_text());stage=list(csv.DictReader((B/'prepared_attempt12/stage_manifest.csv').open()));lookup={r['bjt']:r for r in stage};pairs=[];required=set()
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
for st in ['Niushou','Wangzhai','Mizhuang']:
    p=H/st;rr=list(csv.DictReader((p/'rows.csv').open()));roles=np.load(p/'roles.npy');wide=set(coverage[st]['wide_dates'])
    for i,(r,role) in enumerate(zip(rr,roles)):
        date=r['target_time_bjt'][:8]
        if not (r['split']=='train' and (date in wide or role=='C_select')):continue
        assert r['source_role'].startswith('fold_')
        init=datetime.strptime(r['init_time_bjt'],'%Y%m%d%H%M');target=datetime.strptime(r['target_time_bjt'],'%Y%m%d%H%M');assert target==init+timedelta(minutes=int(r['lead_minutes']))
        needed=[(init-timedelta(minutes=k)).strftime('%Y%m%d%H%M') for k in [105,90,75,60,45,30,15,0]]+[r['target_time_bjt']]
        available=all(t in lookup and lookup[t]['split']=='train' for t in needed)
        pairs.append(dict(station=st,prepared_index=i,scope='C_select' if role=='C_select' else 'C_fit',init_time=r['init_time_bjt'],target_time=r['target_time_bjt'],lead=int(r['lead_minutes']),source_role=r['source_role'],source_sequence_index=int(r['source_sequence_index']),all_frame_metadata_available=available))
        if available:required.update((r['source_role'],t) for t in needed)
out=P/'raw_attempt1';out.mkdir(exist_ok=False)
with (out/'pairs.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
times=sorted({t for _,t in required});raw=np.lib.format.open_memmap(out/'raw16.npy',mode='w+',dtype='float32',shape=(len(times),16,16,16));frames=[]
channels=[0,1,2,3,4,5,8,9,10,11,12,13,14];sl=(slice(103,119),slice(118,134))
for i,t in enumerate(times):
    r=lookup[t];assert r['utc']==(datetime.strptime(t,'%Y%m%d%H%M')-timedelta(hours=8)).strftime('%Y%m%d%H%M')
    assert sha(r['data_path'])==r['data_sha256'] and sha(r['aux_path'])==r['aux_sha256']
    scene=np.load(r['data_path'],mmap_mode='r');assert scene.shape==(20,256,256)
    with np.load(r['aux_path']) as aux:geom=np.stack([aux[k][sl] for k in ['cosSOZ','cosRAA','day_mask']])
    x=np.concatenate((scene[channels,sl[0],sl[1]],geom));assert x.shape==(16,16,16) and np.isfinite(x).all();raw[i]=x
    frames.append(dict(frame_index=i,bjt=t,utc=r['utc'],split='train',data_sha256=r['data_sha256'],aux_sha256=r['aux_sha256'],patch_sha256=hashlib.sha256(x.tobytes()).hexdigest()))
    if i%250==0:print('FRAME',i,len(times),flush=True)
raw.flush()
with (out/'frames.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(frames[0]));w.writeheader();w.writerows(frames)
with (out/'role_frames.csv').open('w',newline='') as f:w=csv.writer(f);w.writerow(['role','bjt']);w.writerows(sorted(required))
report=dict(state='MATCHED_FOLD_R_RAW_TRAIN_PATCHES_READY',candidate_pairs=len(pairs),usable_pairs=sum(r['all_frame_metadata_available'] for r in pairs),frames=len(times),role_frames=len(required),all_source_hashes_verified=True,raw_sha256=sha(out/'raw16.npy'),pairs_sha256=sha(out/'pairs.csv'),stage_manifest_sha256=sha(B/'prepared_attempt12/stage_manifest.csv'),outer_val_new_period_test_buffer_payloads_read=0)
(out/'COMPLETE.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
