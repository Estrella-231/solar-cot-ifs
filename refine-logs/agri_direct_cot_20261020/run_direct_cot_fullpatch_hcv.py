"""H-only date-block CV: does full 16x16 forecast COT context help station GHI?"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, random, time, shutil
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import torch
from torch import nn

STATIONS=('sili','zhujia'); ARMS=('none','pxc3','pxc16'); SEEDS=(42,43,44)
EPOCHS,BATCH=30,512
SOZ80=float(np.cos(np.deg2rad(80.0)))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def read_jsonl(p): return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines()]
def bjt_date(k): return (datetime.fromisoformat(k['target_time_utc'])+timedelta(hours=8)).date()
def lead_embedding(leads):
    x=torch.as_tensor(leads,dtype=torch.float32).reshape(-1,1)/240.0
    return torch.cat([z for f in (1,2,4,8) for z in (torch.sin(2*math.pi*f*x),torch.cos(2*math.pi*f*x))],1).numpy()

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.agri_encoder=nn.Sequential(nn.Conv2d(15,32,1),nn.SiLU(),nn.Conv2d(32,32,1),nn.SiLU())
        self.cot_encoder=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.SiLU(),nn.MaxPool2d(2),
            nn.Conv2d(16,16,3,padding=1),nn.SiLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.SiLU(),
            nn.Flatten(),nn.Linear(32*4*4,64),nn.SiLU())
        self.readout=nn.Sequential(nn.Linear(32*9+64+8,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1),nn.Softplus())
    def forward(self,agri,geometry,cot,lead):
        if cot.shape!=(len(agri),1,16,16) or lead.shape!=(len(agri),8): raise ValueError('fullpatch input shape mismatch')
        ag=self.agri_encoder(torch.cat((agri,geometry),1)).flatten(1)
        co=self.cot_encoder(cot)
        return self.readout(torch.cat((ag,co,lead),1)).squeeze(-1)

def station_centers(rpack):
    rows=list(csv.DictReader((rpack/'rows_u_supervised.csv').open(encoding='utf-8',newline='')))
    result={}
    for r in rows:
        rc=tuple(map(int,r['station_grid_rc'].split(',')))
        if r['station'] in result and result[r['station']]!=rc: raise ValueError('station center varies')
        result[r['station']]=rc
    if set(result)!=set(STATIONS): raise ValueError('station centers missing')
    return result

def build_sources(bank_dir,rows,inference_h,station_index,grid_path,center):
    from solar_geometry_corrected import compute_solar_angles
    lead=np.asarray([int(r['lead_minutes']) for r in rows],np.int64)
    if not np.isin(lead,np.arange(15,241,15)).all(): raise ValueError('unexpected lead')
    pxc=np.load(inference_h/'cot_p_xc.npy',mmap_mode='r',allow_pickle=False)
    idxrows=list(csv.DictReader((inference_h/'index.csv').open(encoding='utf-8',newline='')))
    if pxc.shape[1:]!=(2,16,1,16,16) or len(idxrows)!=pxc.shape[0]: raise ValueError(f'inference array/index mismatch: {pxc.shape}')
    seq_to_idx={r['seq_id']:int(r['sequence_index_within_split']) for r in idxrows}
    with np.load(grid_path,allow_pickle=False) as z: lat,lon=np.asarray(z['lat'],np.float32),np.asarray(z['lon'],np.float32)
    if lat.shape!=(256,256) or lon.shape!=lat.shape: raise ValueError('grid static shape mismatch')
    rr,cc=center; r0,c0=rr-8,cc-8
    if r0<0 or c0<0 or r0+16>256 or c0+16>256: raise ValueError('station crop out of grid')
    latp,lonp=lat[r0:r0+16,c0:c0+16],lon[r0:r0+16,c0:c0+16]
    # The archived rowbank stores this H-only artifact at train/p_xc.npy;
    # its manifest retains the logical cot_raw/p_xc.npy name.  The caller
    # verifies the physical file against that manifest hash before loading.
    bank3=np.load(bank_dir/'p_xc.npy',mmap_mode='r',allow_pickle=False)
    if bank3.shape!=(len(rows),1,3,3): raise ValueError('rowbank 3x3 shape mismatch')
    full=np.empty((len(rows),16,16),np.float32); cos=np.empty_like(full)
    geo_cache={}
    for i,(row,lm) in enumerate(zip(rows,lead)):
        init=datetime.fromisoformat(row['init_time_utc']); seq_id='hunan_'+(init-timedelta(minutes=105)).strftime('%Y%m%d%H%M')
        if seq_id not in seq_to_idx: raise ValueError(f'H sequence missing: {seq_id}')
        si=seq_to_idx[seq_id]; li=int(lm)//15-1
        field=np.asarray(pxc[si,station_index,li,0],np.float32)
        if not np.isfinite(field).all() or (field<0).any(): raise ValueError('invalid predicted COT')
        if not np.allclose(field[7:10,7:10],bank3[i,0],rtol=2e-5,atol=2e-5): raise ValueError(f'center 3x3 does not match rowbank at row {i}')
        full[i]=field
        t=datetime.fromisoformat(row['target_time_utc']).replace(tzinfo=None)
        if t not in geo_cache:
            soz,_=compute_solar_angles(latp,lonp,t)
            geo_cache[t]=np.cos(np.deg2rad(soz)).astype(np.float32)
        cos[i]=geo_cache[t]
    return full,cos,lead

def prepare_arrays(raw,full,cos,train_idx):
    ag,geo,y,clear=raw['agri'],raw['geometry'],raw['ghi'],raw['clear']
    am=ag[train_idx].mean((0,2,3),dtype=np.float64).astype(np.float32)
    ast=np.maximum(ag[train_idx].std((0,2,3),dtype=np.float64),1e-6).astype(np.float32)
    gm=geo[train_idx].mean((0,2,3),dtype=np.float64).astype(np.float32)
    gst=np.maximum(geo[train_idx].std((0,2,3),dtype=np.float64),1e-6).astype(np.float32)
    log=np.log1p(np.maximum(full,0)).astype(np.float64); sun=cos>SOZ80
    counts=sun[train_idx].sum(0)
    if (counts<1000).any(): raise RuntimeError('insufficient per-pixel sunlit training support')
    mu=(np.where(sun[train_idx],log[train_idx],0).sum(0)/counts).astype(np.float32)
    var=(np.where(sun[train_idx],(log[train_idx]-mu[None])**2,0).sum(0)/counts)
    sd=np.maximum(np.sqrt(var),1e-6).astype(np.float32)
    z=((log.astype(np.float32)-mu[None])/sd[None]).astype(np.float32); z[~sun]=0
    none=np.zeros((len(full),1,16,16),np.float32)
    pxc16=z[:,None].copy(); pxc3=np.zeros_like(pxc16); pxc3[:,:,7:10,7:10]=pxc16[:,:,7:10,7:10]
    agz=((ag-am[None,:,None,None])/ast[None,:,None,None]).astype(np.float32)
    gz=((geo-gm[None,:,None,None])/gst[None,:,None,None]).astype(np.float32)
    arr={'agri':agz,'geometry':gz,'none':none,'pxc3':pxc3,'pxc16':pxc16,
         'ghi':y.astype(np.float32),'clear':clear.astype(np.float32),'lead':lead_embedding(raw['lead']).astype(np.float32)}
    norm={'agri_mean':am.tolist(),'agri_std':ast.tolist(),'geometry_mean':gm.tolist(),'geometry_std':gst.tolist(),
          'cot_log1p_mean_by_pixel':mu.tolist(),'cot_log1p_std_by_pixel':sd.tolist(),'cot_fit_counts_by_pixel':counts.astype(int).tolist(),
          'cot_fit_partition':'current_fold_train_dates_only','cot_support':'per-pixel cosSOZ>cos80deg'}
    return arr,norm

def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def train_one(st,fold,arm,seed,tr,va,valkeys,out,bank_sha,norm):
    run=out/st/f'fold{fold}'/arm/f'seed{seed}'; run.mkdir(parents=True,exist_ok=False)
    base=seed+fold*100003; seed_all(base); model=Head().cuda()
    init=hashlib.sha256(b''.join(k.encode()+v.detach().cpu().contiguous().numpy().tobytes() for k,v in sorted(model.state_dict().items()))).hexdigest()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS,eta_min=1e-5)
    n=len(tr['ghi']); updates=0; curve=[]; t0=time.monotonic()
    for ep in range(EPOCHS):
        order=np.random.default_rng(base+ep).permutation(n); model.train(); sse=0.0
        for lo in range(0,n,BATCH):
            ids=torch.as_tensor(order[lo:lo+BATCH],device='cuda',dtype=torch.long)
            pred=model(tr['agri'][ids],tr['geometry'][ids],tr[arm][ids],tr['lead'][ids])
            loss=(((pred*tr['clear'][ids]-tr['ghi'][ids])/1000.0)**2).mean()
            if not torch.isfinite(loss): raise FloatingPointError(f'nonfinite {st}/fold{fold}/{arm}/{seed}')
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0,error_if_nonfinite=True); opt.step()
            sse+=float(loss.detach())*len(ids); updates+=1
        sch.step(); rec={'epoch':ep+1,'train_scaled_mse':sse/n,'updates':updates}; curve.append(rec)
        if ep in (0,4,9,19,29): print(json.dumps({'station':st,'fold':fold,'arm':arm,'seed':seed,**rec}),flush=True)
    model.eval(); preds=[]
    with torch.inference_mode():
        for lo in range(0,len(va['ghi']),2048):
            ix=slice(lo,min(lo+2048,len(va['ghi'])))
            preds.append((model(va['agri'][ix],va['geometry'][ix],va[arm][ix],va['lead'][ix]).double()*va['clear'][ix].double()).cpu().numpy())
    pred=np.concatenate(preds).astype(np.float64); truth=va['ghi'].double().cpu().numpy()
    torch.save({'state_dict':model.state_dict(),'station':st,'fold':fold,'arm':arm,'seed':seed,'base_seed':base,'epochs':EPOCHS,'batch':BATCH,'bank_sha256':bank_sha,'initial_state_sha256':init,'normalization':norm},run/'epoch_030.pt')
    np.savez_compressed(run/'oof_predictions.npz',prediction_ghi=pred,truth_ghi=truth)
    (run/'oof_keys.jsonl').write_text('\n'.join(json.dumps(k,sort_keys=True) for k in valkeys)+'\n',encoding='utf-8')
    (run/'training_curve.json').write_text(json.dumps(curve,indent=2)+'\n',encoding='utf-8')
    return {'station':st,'fold':fold,'arm':arm,'seed':seed,'n_train':n,'n_oof':len(truth),'epochs':EPOCHS,'updates':updates,
            'initial_state_sha256':init,'checkpoint_sha256':sha(run/'epoch_030.pt'),'prediction_sha256':sha(run/'oof_predictions.npz'),'elapsed_seconds':time.monotonic()-t0}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--bank',type=Path,required=True); ap.add_argument('--inference',type=Path,required=True); ap.add_argument('--r-pack',type=Path,required=True); ap.add_argument('--grid-static',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--smoke',action='store_true'); ap.add_argument('--resume',action='store_true'); a=ap.parse_args()
    bm=json.loads((a.bank/'bank_manifest.json').read_text(encoding='utf-8')); im=json.loads((a.inference/'inference_manifest.json').read_text(encoding='utf-8'))
    if bm.get('state')!='COMPLETE_DIRECT_COT_FORECAST_ROW_BANK' or bm.get('train_partition')!='H': raise RuntimeError('H-only rowbank required')
    if im.get('state')!='COMPLETE_CAUSAL_DIRECT_COT_HV_INFERENCE': raise RuntimeError('causal H/V forecast receipt required')
    for m in (bm,im):
        for k in ('test_payload_opened','buffer_payload_opened','future_observed_agri_used_as_input','cpp_used_as_inference_input','cpp_used_as_input'):
            if k in m and m[k] is not False: raise RuntimeError(f'forbidden source flag {k}')
    if im.get('future_observed_agri_read') is not False or im.get('cpp_read') is not False or im.get('ghi_labels_read') is not False: raise RuntimeError('inference read forbidden targets')
    if sha(a.grid_static)!=json.loads((a.r_pack/'pack_audit.json').read_text())['grid_static_sha256']: raise RuntimeError('grid_static does not match R_U pack provenance')
    centers=station_centers(a.r_pack)
    # This archived receipt stores the H-only arrays and index.csv directly
    # under inference/ (rather than inference/H/).  Bind the layout by its
    # manifest hash and sequence count; do not infer H from a directory name.
    inference_h=a.inference
    if sha(inference_h/'cot_p_xc.npy')!=im['artifact_sha256']['H/cot_p_xc.npy']:
        raise RuntimeError('root inference P-XC is not the manifest-bound H artifact')
    idxrows=list(csv.DictReader((inference_h/'index.csv').open(encoding='utf-8',newline='')))
    if len(idxrows)!=int(im['sequence_counts']['H']): raise RuntimeError('inference index is not exactly H-only')
    if a.smoke:
        idxrows=list(csv.DictReader((inference_h/'index.csv').open(encoding='utf-8',newline='')))
        if len(idxrows)!=int(im['sequence_counts']['H']): raise RuntimeError('H index count mismatch')
        if sha(inference_h/'cot_p_xc.npy')!=im['artifact_sha256']['H/cot_p_xc.npy']: raise RuntimeError('H P-XC hash mismatch')
        for st,si in zip(STATIONS,(0,1)):
            d=a.bank/st/'train'; rows=read_jsonl(d/'keys.jsonl')
            for nm in ('keys.jsonl','agri_raw.npy','geometry_raw.npy','ghi.npy','clear.npy'):
                if sha(d/nm)!=bm['artifact_sha256'][f'{st}/train/{nm}']: raise RuntimeError(f'rowbank SHA mismatch {st}/{nm}')
            if sha(d/'p_xc.npy')!=bm['artifact_sha256'][f'{st}/train/cot_raw/p_xc.npy']: raise RuntimeError(f'rowbank SHA mismatch {st}/train/cot_raw/p_xc.npy')
            full,cos,leads=build_sources(d,rows,inference_h,si, a.grid_static,centers[st])
            if full.shape!=(len(rows),16,16) or cos.shape!=full.shape or not np.isin(leads,np.arange(15,241,15)).all(): raise RuntimeError('fullpatch source shape mismatch')
            # Smoke validates geometry and tensor construction, so use the
            # full H-only train bank for stable per-pixel daylight support.
            arr,norm=prepare_arrays({'agri':np.load(d/'agri_raw.npy',mmap_mode='r'),'geometry':np.load(d/'geometry_raw.npy',mmap_mode='r'),'ghi':np.load(d/'ghi.npy',mmap_mode='r'),'clear':np.load(d/'clear.npy',mmap_mode='r'),'lead':leads},full,cos,np.arange(len(rows)))
            if arr['pxc16'].shape!=(len(rows),1,16,16) or arr['pxc3'].shape!=arr['pxc16'].shape or not np.allclose(arr['pxc3'][:,:,7:10,7:10],arr['pxc16'][:,:,7:10,7:10]): raise RuntimeError('3x3 control embedding mismatch')
        probe=Head().eval(); y=probe(torch.zeros(2,13,3,3),torch.zeros(2,2,3,3),torch.zeros(2,1,16,16),torch.zeros(2,8))
        if y.shape!=(2,) or not torch.isfinite(y).all(): raise RuntimeError('head smoke failed')
        print(json.dumps({'state':'PASS_DIRECT_COT_FULLPATCH_HCV_SOURCE_AND_MODEL_SMOKE','H_sequences':len(idxrows),'test_payload_opened':False,'buffer_payload_opened':False}),flush=True); return
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    progress_path=a.output/'progress.json'; completed={}; bank_manifest_sha=sha(a.bank/'bank_manifest.json'); inference_manifest_sha=sha(a.inference/'inference_manifest.json'); grid_sha=sha(a.grid_static); source_sha=sha(__file__)
    if a.output.exists():
        if not a.resume: raise FileExistsError(f'{a.output} exists; use --resume only after confirming no prior runner is active')
        if progress_path.exists():
            old=json.loads(progress_path.read_text(encoding='utf-8'))
            expected=(bank_manifest_sha,inference_manifest_sha,grid_sha,source_sha)
            actual=(old.get('bank_manifest_sha256'),old.get('inference_manifest_sha256'),old.get('grid_static_sha256'),old.get('runner_sha256'))
            if old.get('state')!='RUNNING_DIRECT_COT_FULLPATCH_H_CV' or actual!=expected: raise RuntimeError('resume progress does not match fixed source contract')
            for rec in old.get('runs',[]):
                run=a.output/rec['station']/f"fold{rec['fold']}"/rec['arm']/f"seed{rec['seed']}"
                for fn,key in [('epoch_030.pt','checkpoint_sha256'),('oof_predictions.npz','prediction_sha256')]:
                    if sha(run/fn)!=rec[key]: raise RuntimeError(f'resume artifact SHA mismatch: {run/fn}')
                completed[(rec['station'],rec['fold'],rec['arm'],rec['seed'])]=rec
        a.output.mkdir(parents=True,exist_ok=True)
    else:
        if a.resume: raise FileNotFoundError('--resume requested but output folder does not exist')
        a.output.mkdir(parents=True)
    rows_by_st={s:read_jsonl(a.bank/s/'train'/'keys.jsonl') for s in STATIONS}
    all_dates={bjt_date(k) for rs in rows_by_st.values() for k in rs}; months=sorted({d.strftime('%Y-%m') for d in all_dates}); mi={m:i for i,m in enumerate(months)}
    foldmap={d.isoformat():(((d.day-1)//2)+mi[d.strftime('%Y-%m')])%5 for d in sorted(all_dates)}
    allruns=list(completed.values())
    for st,si in zip(STATIONS,(0,1)):
        d=a.bank/st/'train'; rows=rows_by_st[st]
        for nm in ('keys.jsonl','agri_raw.npy','geometry_raw.npy','ghi.npy','clear.npy'):
            if sha(d/nm)!=bm['artifact_sha256'][f'{st}/train/{nm}']: raise RuntimeError(f'rowbank SHA mismatch {st}/{nm}')
        if sha(d/'p_xc.npy')!=bm['artifact_sha256'][f'{st}/train/cot_raw/p_xc.npy']: raise RuntimeError(f'rowbank SHA mismatch {st}/train/cot_raw/p_xc.npy')
        full,cos,leads=build_sources(d,rows,inference_h,si,a.grid_static,centers[st])
        raw={'agri':np.load(d/'agri_raw.npy',mmap_mode='r'),'geometry':np.load(d/'geometry_raw.npy',mmap_mode='r'),'ghi':np.load(d/'ghi.npy',mmap_mode='r'),'clear':np.load(d/'clear.npy',mmap_mode='r'),'lead':leads}
        dates=[bjt_date(k) for k in rows]
        for fold in range(5):
            held={datetime.fromisoformat(x).date() for x,f in foldmap.items() if f==fold}
            blocked={x for x in dates if any(abs((x-h).days)<=1 for h in held)}
            tridx=np.asarray([i for i,x in enumerate(dates) if x not in blocked],np.int64); vaidx=np.asarray([i for i,x in enumerate(dates) if x in held],np.int64)
            arr,norm=prepare_arrays(raw,full,cos,tridx)
            tr={k:torch.as_tensor(v[tridx],device='cuda',dtype=torch.float32) for k,v in arr.items()}
            va={k:torch.as_tensor(v[vaidx],device='cuda',dtype=torch.float32) for k,v in arr.items()}
            for seed in SEEDS:
                init={}
                for arm in ARMS:
                    key=(st,fold,arm,seed)
                    if key in completed:
                        init[arm]=completed[key]['initial_state_sha256']
                        continue
                    stale=a.output/st/f'fold{fold}'/arm/f'seed{seed}'
                    if stale.exists():
                        if not a.resume: raise FileExistsError(stale)
                        shutil.rmtree(stale)
                    rec=train_one(st,fold,arm,seed,tr,va,[rows[i] for i in vaidx],a.output,sha(a.bank/'bank_manifest.json'),norm)
                    allruns.append(rec); init[arm]=rec['initial_state_sha256']
                    completed[key]=rec
                    progress={'state':'RUNNING_DIRECT_COT_FULLPATCH_H_CV','completed_runs':len(allruns),'expected_runs':90,'runs':allruns,'bank_manifest_sha256':bank_manifest_sha,'inference_manifest_sha256':inference_manifest_sha,'grid_static_sha256':grid_sha,'runner_sha256':source_sha,'test_payload_opened':False,'buffer_payload_opened':False,'future_observed_agri_used_as_input':False,'cpp_used_as_input':False}
                    progress_path.write_text(json.dumps(progress,indent=2)+'\n',encoding='utf-8'); print(json.dumps({'completed_run':rec}),flush=True)
                if len(set(init.values()))!=1: raise RuntimeError('paired initialization mismatch')
    manifest={'state':'COMPLETE_DIRECT_COT_FULLPATCH_H_CV','arms':list(ARMS),'stations':list(STATIONS),'folds':5,'seeds':list(SEEDS),'epochs':EPOCHS,'batch_size':BATCH,'runs':allruns,'date_fold':foldmap,'bank_manifest_sha256':bank_manifest_sha,'inference_manifest_sha256':inference_manifest_sha,'grid_static_sha256':grid_sha,'runner_sha256':source_sha,'fold_assignment':'fixed H calendar two-day folds; ±1 BJT-day train gap','normalization_scope':'per-pixel training-fold dates only','spatial_encoding':'16x16 convolutional feature map flattened; no spatial global pooling','test_payload_opened':False,'buffer_payload_opened':False,'future_observed_agri_used_as_input':False,'cpp_used_as_input':False}
    (a.output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8'); print(json.dumps({'state':manifest['state'],'runs':len(allruns)}),flush=True)

if __name__=='__main__': main()
