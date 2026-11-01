#!/usr/bin/env python3
"""Run the paired old/corrected frozen R models for the exact GHI train/val histories."""
import argparse,csv,hashlib,json
from collections import defaultdict
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
import torch

CHANNELS=(0,1,2,3,4,5,8,9,10,11,12,13,14)
STATIONS=('sili','zhujia')
COS80=float(np.cos(np.deg2rad(80.0)))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<22),b''):h.update(b)
    return h.hexdigest()

def write_json(p,obj):
    t=Path(str(p)+'.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)

def stamp(t):return datetime.fromisoformat(t).strftime('%Y%m%d%H%M')

def relpath(t):
    d=datetime.fromisoformat(t)
    return f'{d:%Y/%Y%m/%Y%m%d}/{d:%Y%m%d%H%M}.npy'

def locate_aux(root,t):
    bjt=datetime.fromisoformat(t)+timedelta(hours=8)
    directory=root/'aux_256'/bjt.strftime('%Y%m');prefix=bjt.strftime('%Y%m%d%H%M')
    expected=directory/(prefix+'_1_aux.npz')
    if expected.is_file():return expected
    files=sorted(directory.glob(prefix+'_*_aux.npz'))
    if len(files)!=1: raise FileNotFoundError(f'expected one aux sidecar for {bjt}: {files}')
    return files[0]

def main():
    ap=argparse.ArgumentParser()
    for n in ('pack','data-root','grid-static','monthly-norm','old-checkpoint','old-norm','corrected-checkpoint','corrected-norm','output'):
        ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--preflight',action='store_true');ap.add_argument('--batch',type=int,default=512)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError(f'output exists; select a new directory: {a.output}')
    mp=json.loads((a.pack/'pack_manifest.json').read_text());rows=list(csv.DictReader((a.pack/'rows.csv').open(newline='',encoding='utf-8')))
    if mp.get('state')!='COMPLETE' or mp.get('test_payload_opened') is not False or mp.get('split_id')!='hunan_monthly_block_2d_buffer1d_seed42_v1':raise ValueError('GHI pack gate failed')
    if len(rows)!=51929 or sum(r['split']=='train' for r in rows)!=47666 or sum(r['split']=='val' for r in rows)!=4263:raise ValueError('unexpected fixed GHI row counts')
    if {r['split'] for r in rows}!={'train','val'}:raise ValueError('non train/val split row present')
    if any(r['station'] not in STATIONS for r in rows):raise ValueError('unexpected station')
    coord={s:set() for s in STATIONS}; frame_rows=defaultdict(dict); frame_split={};sample_keys=set()
    for r in rows:
        s=r['station'];coord[s].add((int(r['row']),int(r['col'])))
        t=datetime.fromisoformat(r['target_time_utc'])
        if r['agri_frame_relpath']!=relpath(r['target_time_utc']):raise ValueError(f'AGRI relpath/time mismatch: {r}')
        sk=(s,r['target_time_utc'])
        if sk in sample_keys:raise ValueError(f'duplicate GHI station/target key: {sk}')
        sample_keys.add(sk)
        times=[datetime.fromisoformat(r['earlier_time_utc']),datetime.fromisoformat(r['previous_time_utc']),t]
        if [int((t-x).total_seconds()/60) for x in times]!=[30,15,0]:raise ValueError(f'bad three-frame offsets: {r}')
        if any((x+timedelta(hours=8)).date()!=(t+timedelta(hours=8)).date() for x in times):raise ValueError('cross-BJT-date history')
        for x in times:
            k=(s,x.strftime('%Y-%m-%d %H:%M:%S'));frame_rows[k][r['split']]=1
            if k in frame_split and frame_split[k]!=r['split']:raise ValueError('same frame key crosses split')
            frame_split[k]=r['split']
    if any(len(coord[s])!=1 for s in STATIONS):raise ValueError(f'station grid coordinates vary: {coord}')
    coords={s:next(iter(coord[s])) for s in STATIONS}
    keys=sorted(frame_rows,key=lambda k:(k[1],k[0]));times=sorted({k[1] for k in keys})
    root=a.data_root
    # Exact inputs, sidecars and their geometry/validity are audited before GPU inference.
    missing=[];sidecars={};raw_paths={}
    for t in times:
        rp=root/'data'/relpath(t);raw_paths[t]=rp
        if not rp.is_file():missing.append(str(rp))
        sidecars[t]=locate_aux(root,t)
    if missing:raise FileNotFoundError(f'{len(missing)} missing AGRI frames; first={missing[:5]}')
    sim=json.loads(a.monthly_norm.read_text());smean=np.asarray(sim['mean'],np.float32);sstd=np.asarray(sim['std'],np.float32)
    if smean.shape!=(13,) or sstd.shape!=(13,) or not np.isfinite(smean).all() or not np.isfinite(sstd).all() or (sstd<=0).any():raise ValueError('invalid upstream train-only AGRI normalization')
    with np.load(a.grid_static,allow_pickle=False) as g:
        lat=np.asarray(g['lat'],np.float32);lon=np.asarray(g['lon'],np.float32);saa=np.asarray(g['saa'],np.float32)
    if any(z.shape!=(256,256) for z in (lat,lon,saa)):raise ValueError('Hunan grid_static shape mismatch')
    # Validate representative raw and aux schema. Full reads follow only after preflight passes.
    sample=np.load(raw_paths[times[0]],mmap_mode='r',allow_pickle=False)
    if sample.shape!=(20,256,256) or sample.dtype!=np.float32:raise ValueError(f'AGRI source shape/dtype invalid: {sample.shape} {sample.dtype}')
    with np.load(sidecars[times[0]],allow_pickle=False) as z:
        for f in ('agri_valid_bits','cosSOZ','cosRAA','day_mask'):
            if z[f].shape!=(256,256):raise ValueError(f'aux field shape invalid: {f}')
    if a.preflight:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        report=dict(state='PASS_PAIRED_R_BANK_PREFLIGHT',pack_rows=len(rows),train_rows=mp['train'],val_rows=mp['val'],
                    unique_frame_keys=len(keys),unique_times=len(times),coordinates={s:list(coords[s]) for s in STATIONS},
                    input_frame_paths_present=len(raw_paths),sidecars_unique=len(sidecars),test_payloads_read=0,
                    monthly_norm_sha256=sha(a.monthly_norm),rows_sha256=sha(a.pack/'rows.csv'),pack_manifest_sha256=sha(a.pack/'pack_manifest.json'),
                    sample_agri_shape=list(sample.shape),sample_aux=str(sidecars[times[0]]))
        write_json(a.output,report);print(json.dumps(report,indent=2));return
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for full paired-R bank inference')
    from solar_geometry_corrected import compute_solar_angles
    from train_paired_r import COTUNet
    dev=torch.device('cuda:0');torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    models=[];norms=[]
    for ckpt,norm_path in ((a.old_checkpoint,a.old_norm),(a.corrected_checkpoint,a.corrected_norm)):
        model=COTUNet().to(dev).float().eval().requires_grad_(False)
        payload=torch.load(ckpt,map_location='cpu',weights_only=False)
        model.load_state_dict(payload['model'],strict=True)
        del payload
        n=json.loads(norm_path.read_text());mean=np.asarray(n['mean'],np.float32)[:,None,None];std=np.asarray(n['std'],np.float32)[:,None,None]
        if mean.shape!=(16,1,1) or std.shape!=(16,1,1) or (std<=0).any():raise ValueError('invalid paired R normalization')
        models.append(model);norms.append((mean,std))
    a.output.mkdir(parents=True)
    out=np.lib.format.open_memmap(a.output/'frame_cot_log1p.npy',mode='w+',dtype=np.float32,shape=(len(keys),2,1,3,3));out[:]=np.nan
    index={k:i for i,k in enumerate(keys)};pending_old=[];pending_new=[];pending_meta=[];done=0
    def infer():
        nonlocal pending_old,pending_new,pending_meta,done
        if not pending_meta:return
        bx=torch.from_numpy(np.stack(pending_old)).to(dev,dtype=torch.float32)
        nx=torch.from_numpy(np.stack(pending_new)).to(dev,dtype=torch.float32)
        with torch.inference_mode():
            pred=[models[0](bx).float().cpu().numpy(),models[1](nx).float().cpu().numpy()]
        for bi,meta in enumerate(pending_meta):
            s,t=meta
            out[index[(s,t)],0]=pred[0][bi,:,7:10,7:10]
            out[index[(s,t)],1]=pred[1][bi,:,7:10,7:10]
        done+=len(pending_meta);out.flush();write_json(a.output/'progress.json',dict(state='BUILDING_PAIRED_R_COT_BANK',frame_keys_done=done,frame_keys_total=len(keys),test_payloads_read=0))
        pending_old.clear();pending_new.clear();pending_meta.clear()
        print(f'PROGRESS {done}/{len(keys)}',flush=True)
    for ti,t in enumerate(times):
        raw=np.load(raw_paths[t],mmap_mode='r',allow_pickle=False)
        auxfile=sidecars[t]
        with np.load(auxfile,allow_pickle=False) as z:
            bits=np.asarray(z['agri_valid_bits']);oldgeo=[np.asarray(z[k],np.float32) for k in ('cosSOZ','cosRAA','day_mask')]
            for s in STATIONS:
                key=(s,t)
                if key not in index:continue
                rr,cc=coords[s];r0,c0=rr-8,cc-8;sl=np.s_[r0:r0+16,c0:c0+16]
                physical=np.asarray(raw[np.asarray(CHANNELS),sl[0],sl[1]],np.float32)
                valid=np.stack([((bits[sl]&(1<<k))!=0) for k in CHANNELS])&np.isfinite(physical)
                physical=np.where(valid,physical,smean[:,None,None]).astype(np.float32)
                # Old-R features use the exact cached producer geometry sidecar.
                xold=np.concatenate((physical,np.stack([g[sl] for g in oldgeo])),axis=0)
                dt=datetime.fromisoformat(t);soz,soa=compute_solar_angles(lat[sl],lon[sl],dt)
                soa=np.where(soz>=90.0,0.0,soa)
                corrected=np.stack([np.cos(np.deg2rad(soz)),np.cos(np.deg2rad(soa-saa[sl])),(soz<85.0).astype(np.float32)]).astype(np.float32)
                xnew=np.concatenate((physical,corrected),axis=0)
                xo=np.where(np.isfinite((xold-norms[0][0])/norms[0][1]),(xold-norms[0][0])/norms[0][1],0.).astype(np.float32)
                xn=np.where(np.isfinite((xnew-norms[1][0])/norms[1][1]),(xnew-norms[1][0])/norms[1][1],0.).astype(np.float32)
                pending_old.append(xo);pending_new.append(xn);pending_meta.append((s,t))
        if len(pending_meta)>=a.batch:infer()
    infer()
    if not np.isfinite(out).all():raise RuntimeError('paired R COT bank contains missing/nonfinite outputs')
    lookup={k:i for i,k in enumerate(keys)}
    cot_old=np.empty((len(rows),3,1,3,3),np.float32);cot_new=np.empty_like(cot_old)
    for i,r in enumerate(rows):
        ts=[r['earlier_time_utc'],r['previous_time_utc'],r['target_time_utc']]
        for j,t in enumerate(ts):
            k=(r['station'],t)
            if k not in lookup:raise KeyError(f'missing exact frame mapping {k}')
            cot_old[i,j]=out[lookup[k],0];cot_new[i,j]=out[lookup[k],1]
    np.save(a.output/'cot_old_log1p.npy',cot_old,allow_pickle=False);np.save(a.output/'cot_corrected_log1p.npy',cot_new,allow_pickle=False)
    with (a.output/'frame_keys.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['index','station','target_time_utc','split','agri_frame_relpath','row','col'])
        for i,(s,t) in enumerate(keys):w.writerow([i,s,t,frame_split[(s,t)],relpath(t),coords[s][0],coords[s][1]])
    manifest=dict(state='COMPLETE_PAIRED_R_COT_HISTORY_BANK',split_id=mp['split_id'],rows=len(rows),train=mp['train'],val=mp['val'],
        unique_station_frame_keys=len(keys),unique_utc_times=len(times),history_offsets_minutes=[-30,-15,0],
        input='real observed AGRI; paired fresh old-formula / corrected-geometry R; train-fitted per-R normalization; no CPP input',
        output='softplus head output interpreted as log1p(COT), full R patch center 3x3 readout; no masking, GAP, or spatial averaging',
        old_R_checkpoint_sha256=sha(a.old_checkpoint),corrected_R_checkpoint_sha256=sha(a.corrected_checkpoint),
        old_R_norm_sha256=sha(a.old_norm),corrected_R_norm_sha256=sha(a.corrected_norm),
        source_GHI_rows_sha256=sha(a.pack/'rows.csv'),source_manifest_sha256=sha(a.pack/'pack_manifest.json'),
        monthly_norm_sha256=sha(a.monthly_norm),grid_static_sha256=sha(a.grid_static),
        rows_sha256=sha(a.output/'frame_keys.csv'),frame_bank_sha256=sha(a.output/'frame_cot_log1p.npy'),
        cot_old_sha256=sha(a.output/'cot_old_log1p.npy'),cot_corrected_sha256=sha(a.output/'cot_corrected_log1p.npy'),
        test_payloads_read=0,all_rows_common=True,all_R_frozen=True,paired_row_mapping='(station, exact UTC frame time)')
    write_json(a.output/'bank_manifest.json',manifest);print(json.dumps(manifest,indent=2),flush=True)
if __name__=='__main__':main()
