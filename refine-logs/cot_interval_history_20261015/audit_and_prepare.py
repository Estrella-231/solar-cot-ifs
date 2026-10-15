"""Train/val-only source audit and common three-frame interval pack builder."""
import argparse, csv, hashlib, json, shutil
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from solar_geometry_corrected import compute_solar_angles

COORD = {'sili': (163, 148), 'zhujia': (73, 150)}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_csv(p, rows):
    if rows:
        with p.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--labels',type=Path,required=True)
    ap.add_argument('--pack',type=Path,required=True);ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--reuse-witness',type=Path);a=ap.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    labels=list(csv.DictReader(a.labels.open())); assert {r['new_split'] for r in labels}<={'train','val'}
    native={}; conflicts=[]
    for row in labels:
        for ts,v in zip(row['label_record_times_bjt'].split('|'),row['raw_ghi_5min_wm2'].split('|')):
            try: x=float(v)
            except ValueError: continue
            if not np.isfinite(x):continue
            key=(row['station'],ts)
            if key in native and native[key]!=x:conflicts.append(key)
            native[key]=x
    assert not conflicts
    events=[];longzero=set()
    for st in COORD:
        zero=sorted(datetime.fromisoformat(t) for (s,t),v in native.items() if s==st and v==0)
        runs=[]
        for t in zero:
            if not runs or t-runs[-1][-1]!=timedelta(minutes=5):runs.append([t])
            else:runs[-1].append(t)
        for run in runs:
            if len(run)<6:continue
            keys={(st,t.isoformat(sep=' ')) for t in run};longzero |= keys
            events.append(dict(station=st,start_bjt=run[0].isoformat(sep=' '),end_bjt=run[-1].isoformat(sep=' '),native_zero_count=len(run)))
    rows=list(csv.DictReader((a.pack/'rows.csv').open()));assert {r['split'] for r in rows}<={'train','val'}
    assert len({(r['station'],r['target_time_utc']) for r in rows})==len(rows)
    flagged=[]
    for r in rows:
        tb=datetime.fromisoformat(r['target_time_utc'])+timedelta(hours=8)
        bad=float(r['ghi'])==0 and float(r['clear'])>200 and all((r['station'],(tb-timedelta(minutes=k)).isoformat(sep=' ')) in longzero for k in (10,5,0))
        r['suspect_daylight_long_zero']=str(int(bad))
        if bad:flagged.append(dict(station=r['station'],split=r['split'],target_bjt=tb.isoformat(sep=' '),target_time_utc=r['target_time_utc'],ghi=float(r['ghi']),clear=float(r['clear'])))
    write_csv(a.output/'suspect_zero_labels.csv',flagged)
    relevant=[]
    for e in events:
        ff=[r for r in flagged if r['station']==e['station'] and e['start_bjt']<=r['target_bjt']<=e['end_bjt']]
        if ff: relevant.append(dict(**e,daylight_flagged_15min_labels=len(ff),splits='|'.join(sorted({r['split'] for r in ff}))))
    write_csv(a.output/'zero_events.csv',relevant)
    requests=[]
    for st in COORD:
        # Select dates by number of flagged labels, independent of model errors.
        dates=defaultdict(int)
        for r in flagged:
            if r['station']==st:dates[r['target_bjt'][:10]]+=1
        chosen=sorted(dates,key=lambda d:(-dates[d],d))[:4]
        if st=='sili' and '2024-09-05' not in chosen:chosen.append('2024-09-05')
        for (s,t),v in native.items():
            if s==st and t[:10] in chosen:requests.append(dict(station=st,time_bjt=t,expected=v))
    write_csv(a.output/'workbook_requests.csv',requests)

    meta=json.loads((a.pack/'pack_manifest.json').read_text());assert meta['state']=='COMPLETE' and meta['test_payload_opened'] is False
    grid=dict(np.load(a.data_root/'grid_static.npz'));geo=np.empty((len(rows),2,3,3),np.float32)
    for i,r in enumerate(rows):
        rr,cc=COORD[r['station']];sl=np.s_[rr-1:rr+2,cc-1:cc+2]
        zen,az=compute_solar_angles(grid['lat'][sl],grid['lon'][sl],datetime.fromisoformat(r['target_time_utc']))
        geo[i,0]=np.cos(np.deg2rad(zen));geo[i,1]=np.cos(np.deg2rad(az-grid['saa'][sl])) if 'saa' in grid else 0
    if 'saa' not in grid:
        # Fixed sensor azimuth may be named differently; never silently use zeros.
        raise ValueError('grid_static lacks saa; keys='+str(list(grid)))
    oldgeo=np.load(a.pack/'geometry.npy',mmap_mode='r')
    audit=dict(state='SOURCE_ZERO_AND_GEOMETRY_PREPARATION_COMPLETE',label_rows=len(labels),pack_rows=len(rows),
        native_conflicts=0,suspect_daylight_long_zero_by_station_split={f'{st}_{sp}':sum(r['station']==st and r['split']==sp for r in flagged) for st in COORD for sp in ('train','val')},
        zero_definition='native zero run >=6 adjacent five-minute records; three zero label records in run; interval clear GHI >200',
        primary_labels_modified=False,invalidity_confirmed=False,geometry_cosine_max_abs_delta=np.max(np.abs(geo-oldgeo),axis=(0,2,3)).tolist(),
        labels_sha256=sha(a.labels),source_pack_sha256=sha(a.pack/'pack_manifest.json'),test_payloads_read=0)
    (a.output/'source_audit.json').write_text(json.dumps(audit,indent=2))
    # Deterministic geometry witness: stratify selected val frames across months/local hours.
    selected={}
    for i,r in enumerate(rows):
        if r['split']!='val':continue
        t=datetime.fromisoformat(r['target_time_utc']);b=t+timedelta(hours=8)
        if b.hour not in (10,13,16) or b.minute!=0:continue
        selected.setdefault((r['station'],b.strftime('%Y-%m'),b.hour),(i,r))
    witness={};wrows=[]
    if a.reuse_witness:selected={}
    for k,(i,r) in enumerate(selected.values()):
        rr,cc=COORD[r['station']];sl=np.s_[rr-1:rr+2,cc-1:cc+2];t=datetime.fromisoformat(r['target_time_utc'])
        x=np.load(a.data_root/'data'/r['agri_frame_relpath'],mmap_mode='r')
        bt=t+timedelta(hours=8);af=list((a.data_root/'aux_256'/bt.strftime('%Y%m')).glob(bt.strftime('%Y%m%d%H%M')+'_*_aux.npz'))
        assert len(af)==1
        with np.load(af[0]) as aux:
            witness[f'{k}_offset']=aux['obs_time_offset_s'][sl];witness[f'{k}_aux_cos']=np.stack([aux['cosSOZ'][sl],aux['cosRAA'][sl]])
        witness[f'{k}_angles']=np.asarray(x[[15,17,18],rr-1:rr+2,cc-1:cc+2])
        witness[f'{k}_lat']=grid['lat'][sl];witness[f'{k}_lon']=grid['lon'][sl]
        witness[f'{k}_corrected']=geo[i];witness[f'{k}_pack_geo']=np.asarray(oldgeo[i])
        wrows.append(dict(index=k,station=r['station'],utc=r['target_time_utc']))
    if a.reuse_witness:
        for name in ('geometry_witness.npz','geometry_witness_rows.csv'):shutil.copy2(a.reuse_witness/name,a.output/name)
    else:
        np.savez_compressed(a.output/'geometry_witness.npz',**witness)
        write_csv(a.output/'geometry_witness_rows.csv',wrows)
    lookup={(r['station'],r['target_time_utc']):i for i,r in enumerate(rows)};pairs=[];paired_rows=[]
    for i,r in enumerate(rows):
        t=datetime.fromisoformat(r['target_time_utc']);prev=t-timedelta(minutes=15);earlier=t-timedelta(minutes=30)
        j=lookup.get((r['station'],prev.isoformat(sep=' ')));k=lookup.get((r['station'],earlier.isoformat(sep=' ')))
        if j is None or k is None or rows[j]['split']!=r['split'] or rows[k]['split']!=r['split'] or (earlier+timedelta(hours=8)).date()!=(t+timedelta(hours=8)).date():continue
        pairs.append((k,j,i));paired_rows.append(dict(**r,source_current_index=i,source_previous_index=j,source_earlier_index=k,previous_time_utc=prev.isoformat(sep=' '),earlier_time_utc=earlier.isoformat(sep=' ')))
    ix=np.asarray(pairs);train=np.array([r['split']=='train' for r in paired_rows]);assert train.any()
    for fn in ('agri_raw','agri_valid','cot_log1p'):
        src=np.load(a.pack/(fn+'.npy'),mmap_mode='r');np.save(a.output/(fn+'.npy'),np.asarray(src[ix]))
    np.save(a.output/'geometry.npy',geo[ix]);write_csv(a.output/'rows.csv',paired_rows)
    ag=np.load(a.output/'agri_raw.npy');valid=np.load(a.output/'agri_valid.npy').astype(bool);cot=np.load(a.output/'cot_log1p.npy')
    am=[];ast=[]
    for ch in range(13):
        q=ag[train,:,ch][valid[train,:,ch]];am.append(float(q.mean(dtype=np.float64)));ast.append(max(float(q.std(dtype=np.float64)),1e-5))
    vals=cot[train].astype(np.float64);manifest=dict(state='COMPLETE',split_id=meta['split_id'],rows=len(pairs),
        train=int(train.sum()),val=int((~train).sum()),history_offsets_minutes=[-30,-15,0],geometry='corrected NOAA nominal UTC; radians-to-degrees in EOT fixed',
        AGRI_norm_mean=am,AGRI_norm_std=ast,COT_norm_mean=float(vals.mean()),COT_norm_std=float(vals.std()),
        source_pack_sha256=sha(a.pack/'pack_manifest.json'),R_checkpoint_sha256=meta['R_checkpoint_sha256'],
        R_geometry_not_modified=True,primary_labels_modified=False,test_payload_opened=False,
        coverage={f'{st}_{sp}':sum(r['station']==st and r['split']==sp for r in paired_rows) for st in COORD for sp in ('train','val')},
        schema='[N,T=3,C,H=3,W=3]; preserve all temporal/spatial positions; no pooling')
    (a.output/'pack_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
