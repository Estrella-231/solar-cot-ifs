"""F-role same-fold observed-R targets. No GHI values or outer payloads."""
import argparse,csv,hashlib,json
from collections import Counter
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np

P=Path(__file__).resolve().parent;ROOT=P.parent
B=ROOT/'hubei_solar_geometry_repair_20261021'
H=ROOT/'hubei_forecast_cot_increment_20261022/prepared_attempt1'
OLD=ROOT/'hubei_matched_R_dynamics_20261023/raw_attempt1'
ap=argparse.ArgumentParser();ap.add_argument('--extract',action='store_true');args=ap.parse_args()
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def writecsv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
stage=list(csv.DictReader((B/'prepared_attempt12/stage_manifest.csv').open()))
lookup={r['bjt']:r for r in stage};assert len(lookup)==len(stage)
coverage=json.loads((ROOT/'hubei_cot_calibration_coverage_20261023/METADATA_AUDIT.json').read_text())
pairs=[];required=set();audit={};source_hashes={}
for st in ['Niushou','Wangzhai','Mizhuang']:
    pp=H/st;rr=list(csv.DictReader((pp/'rows.csv').open()));roles=np.load(pp/'roles.npy')
    assert len(rr)==len(roles)
    dates={role:{r['target_time_bjt'][:8] for r,s in zip(rr,roles) if s==role} for role in ['F_fit','F_select','C_fit','C_select']}
    assert not (dates['F_fit']&dates['F_select'])
    assert not ((dates['F_fit']|dates['F_select'])&(dates['C_fit']|dates['C_select']))
    assert not ((dates['F_fit']|dates['F_select'])&set(coverage[st]['wide_dates']))
    source_hashes[st]={'rows':sha(pp/'rows.csv'),'roles':sha(pp/'roles.npy')}
    audit[st]={role:sorted(ds) for role,ds in dates.items()}
    for i,(r,role) in enumerate(zip(rr,roles)):
        if role not in ['F_fit','F_select']:continue
        assert r['split']=='train' and r['source_role'].startswith('fold_')
        init=datetime.strptime(r['init_time_bjt'],'%Y%m%d%H%M')
        target=datetime.strptime(r['target_time_bjt'],'%Y%m%d%H%M')
        assert target==init+timedelta(minutes=int(r['lead_minutes']))
        needed=[(init-timedelta(minutes=k)).strftime('%Y%m%d%H%M') for k in [105,90,75,60,45,30,15,0]]+[r['target_time_bjt']]
        available=all(t in lookup and lookup[t]['split']=='train' for t in needed)
        pairs.append(dict(station=st,prepared_index=i,scope=str(role),init_time=r['init_time_bjt'],target_time=r['target_time_bjt'],lead=int(r['lead_minutes']),source_role=r['source_role'],source_sequence_index=int(r['source_sequence_index']),all_frame_metadata_available=available))
        if available:required.update((r['source_role'],t) for t in needed)
out=P/'raw_attempt1'
if not args.extract:
    out.mkdir(exist_ok=False)
    writecsv(out/'pairs.csv',pairs)
    writecsv(out/'role_frames.csv',[dict(role=role,bjt=t) for role,t in sorted(required)])
    report=dict(state='F_ROLES_METADATA_READY',pairs=len(pairs),available_pairs=sum(r['all_frame_metadata_available'] for r in pairs),counts=dict(Counter(r['scope'] for r in pairs)),frames=len({t for _,t in required}),role_frames=len(required),station_dates=audit,source_hashes=source_hashes,stage_manifest_sha256=sha(B/'prepared_attempt12/stage_manifest.csv'),pair_sha256=sha(out/'pairs.csv'),GHI_values_read=0,official_val_new_period_outerbuffer_test_payloads_read=0)
    (out/'METADATA.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
else:
    report=json.loads((out/'METADATA.json').read_text())
    assert report['pair_sha256']==sha(out/'pairs.csv')
    assert report['stage_manifest_sha256']==sha(B/'prepared_attempt12/stage_manifest.csv')
    assert not (out/'raw16.npy').exists(),'Do not overwrite partial output; use a separately named attempt after inspection.'
    oldmeta=json.loads((OLD/'COMPLETE.json').read_text());assert oldmeta['raw_sha256']==sha(OLD/'raw16.npy')
    oldidx={r['bjt']:r for r in csv.DictReader((OLD/'frames.csv').open())}
    oldraw=np.load(OLD/'raw16.npy',mmap_mode='r')
    times=sorted({t for _,t in required});raw=np.lib.format.open_memmap(out/'raw16.npy',mode='w+',dtype='float32',shape=(len(times),16,16,16));frames=[];reused=0
    channels=[0,1,2,3,4,5,8,9,10,11,12,13,14];sl=(slice(103,119),slice(118,134))
    for i,t in enumerate(times):
        r=lookup[t];assert r['utc']==(datetime.strptime(t,'%Y%m%d%H%M')-timedelta(hours=8)).strftime('%Y%m%d%H%M')
        if t in oldidx:
            prev=oldidx[t];assert prev['data_sha256']==r['data_sha256'] and prev['aux_sha256']==r['aux_sha256']
            x=np.array(oldraw[int(prev['frame_index'])],copy=True);assert hashlib.sha256(x.tobytes()).hexdigest()==prev['patch_sha256'];reused+=1
        else:
            assert sha(r['data_path'])==r['data_sha256'] and sha(r['aux_path'])==r['aux_sha256']
            scene=np.load(r['data_path'],mmap_mode='r');assert scene.shape==(20,256,256)
            with np.load(r['aux_path']) as aux:geom=np.stack([aux[k][sl] for k in ['cosSOZ','cosRAA','day_mask']])
            x=np.concatenate((scene[channels,sl[0],sl[1]],geom))
        assert x.shape==(16,16,16) and np.isfinite(x).all();raw[i]=x
        frames.append(dict(frame_index=i,bjt=t,utc=r['utc'],split='train',data_sha256=r['data_sha256'],aux_sha256=r['aux_sha256'],patch_sha256=hashlib.sha256(x.tobytes()).hexdigest()))
        if i%500==0:print('FRAME',i,len(times),'reused',reused,flush=True)
    raw.flush();writecsv(out/'frames.csv',frames)
    report.update(state='MATCHED_FOLD_R_RAW_TRAIN_PATCHES_READY',candidate_pairs=len(pairs),usable_pairs=report['available_pairs'],frames=len(times),reused_frames=reused,newly_extracted_frames=len(times)-reused,raw_sha256=sha(out/'raw16.npy'),pairs_sha256=sha(out/'pairs.csv'),all_source_hashes_verified=True)
    (out/'COMPLETE.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
