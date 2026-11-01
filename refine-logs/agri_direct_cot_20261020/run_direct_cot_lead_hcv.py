"""Predeclared 5-fold H-only lead/COT factorial GHI experiment.

No V/test/buffer paths are opened by this runner. Each H key is scored once
by a model whose GHI head and normalization were fit on other H target dates.
"""
from __future__ import annotations
import argparse, hashlib, json, math, random, time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import nn

ARMS = ("none_no_lead", "pxc_no_lead", "none_lead", "pxc_lead")
STATIONS = ("sili", "zhujia")
BASE_SEEDS = (42, 43, 44)
EPOCHS = 30
BATCH = 512
SOZ_COS_80 = float(np.cos(np.deg2rad(80.0)))

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def read_jsonl(p: Path):
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]

def bjt_date(key):
    return (datetime.fromisoformat(key['target_time_utc'])+timedelta(hours=8)).date()

def assign_folds(keys_by_station):
    dates=sorted({bjt_date(k) for rows in keys_by_station.values() for k in rows})
    months=sorted({d.strftime('%Y-%m') for d in dates})
    mi={m:i for i,m in enumerate(months)}
    # Fixed calendar two-day blocks; rotation by month spreads blocks evenly.
    return {d:(((d.day-1)//2)+mi[d.strftime('%Y-%m')])%5 for d in dates}

def audit_folds(keys_by_station, fold_by_date):
    report={}
    all_dates=set(fold_by_date)
    for f in range(5):
        held={d for d,ff in fold_by_date.items() if ff==f}
        blocked={d for d in all_dates if any(abs((d-h).days)<=1 for h in held)}
        item={'heldout_dates':len(held),'buffer_dates_in_H':len(blocked-held),'stations':{}}
        for st,keys in keys_by_station.items():
            ds=[bjt_date(k) for k in keys]
            val=[i for i,d in enumerate(ds) if fold_by_date[d]==f]
            tr=[i for i,d in enumerate(ds) if d not in blocked]
            if not val or not tr: raise RuntimeError(f'empty fold split {f}/{st}')
            trdates={ds[i] for i in tr}; vdates={ds[i] for i in val}
            if any(abs((a-b).days)<=1 for a in trdates for b in vdates):
                raise RuntimeError(f'date-buffer violation fold={f}/{st}')
            item['stations'][st]={'train_rows':len(tr),'oof_rows':len(val),
                'train_dates':len(trdates),'oof_dates':len(vdates),
                'train_target_bjt_min':min(trdates).isoformat(),
                'train_target_bjt_max':max(trdates).isoformat(),
                'oof_target_bjt_min':min(vdates).isoformat(),
                'oof_target_bjt_max':max(vdates).isoformat()}
        report[str(f)]=item
    return report

def lead_encoding(lead):
    phase=lead.float().reshape(-1,1)/240.0
    cols=[]
    for k in (1,2,4,8):
        cols.extend((torch.sin(2*math.pi*k*phase),torch.cos(2*math.pi*k*phase)))
    return torch.cat(cols,dim=1)

def arm_uses_lead(arm):
    if arm not in ARMS: raise ValueError(f'unknown factorial arm: {arm}')
    return arm in ('none_lead','pxc_lead')

class Head(nn.Module):
    def __init__(self, use_lead):
        super().__init__(); self.use_lead=use_lead
        self.agri_encoder=nn.Sequential(nn.Conv2d(15,32,1),nn.SiLU(),nn.Conv2d(32,32,1),nn.SiLU())
        self.cot_encoder=nn.Sequential(nn.Conv2d(1,4,1),nn.SiLU())
        extra=8 if use_lead else 0
        self.readout=nn.Sequential(nn.Linear((32+4)*3*3+extra,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1),nn.Softplus())
    def forward(self,agri,geometry,cot,lead_emb=None):
        ag=self.agri_encoder(torch.cat((agri,geometry),dim=1))
        aux=self.cot_encoder(cot)
        z=torch.cat((ag,aux),dim=1).flatten(1)
        if self.use_lead:
            if lead_emb is None or lead_emb.shape!=(len(agri),8): raise ValueError('lead embedding shape')
            z=torch.cat((z,lead_emb),dim=1)
        return self.readout(z).squeeze(-1)

def norm_stats(x, axis):
    mean=x.mean(axis=axis,dtype=np.float64).astype(np.float32)
    std=x.std(axis=axis,dtype=np.float64).astype(np.float32)
    std=np.maximum(std,1e-6)
    return mean,std

def normalize_fold(raw, train_idx, val_idx):
    ag=raw['agri']; geo=raw['geometry']; y=raw['ghi']; clear=raw['clear']; cot=raw['p_xc']; leads=raw['lead']
    am,ast=norm_stats(ag[train_idx],(0,2,3)); gm,gst=norm_stats(geo[train_idx],(0,2,3))
    agz=((ag-am[None,:,None,None])/ast[None,:,None,None]).astype(np.float32)
    gz=((geo-gm[None,:,None,None])/gst[None,:,None,None]).astype(np.float32)
    sun=geo[:,0:1]>SOZ_COS_80
    logcot=np.log1p(np.maximum(cot,0)).astype(np.float32)
    fit=logcot[train_idx][sun[train_idx]]
    if fit.size<1000: raise RuntimeError('insufficient train COT pixels for fold normalization')
    cm=float(fit.mean(dtype=np.float64)); cs=max(float(fit.std(dtype=np.float64)),1e-6)
    cz=((logcot-cm)/cs).astype(np.float32); cz[~sun]=0.0
    # Standardized COT for the missing-COT controls is the exact neutral zero.
    none=np.zeros_like(cz,dtype=np.float32)
    return {'agri':agz,'geometry':gz,'p_xc':cz,'none':none,'ghi':y.astype(np.float32),
            'clear':clear.astype(np.float32),'lead_emb':lead_encoding(torch.from_numpy(leads)).numpy().astype(np.float32),
            'norm':{'agri_mean':am.tolist(),'agri_std':ast.tolist(),'geometry_mean':gm.tolist(),
                    'geometry_std':gst.tolist(),'p_xc_log1p_mean':cm,'p_xc_log1p_std':cs,
                    'cot_fit_pixels':int(fit.size),'fit_days_only':True}}

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def tensor_data(arrays, idx):
    return {k:torch.as_tensor(arrays[k][idx],device='cuda',dtype=torch.float32) for k in ('agri','geometry','p_xc','none','ghi','clear','lead_emb')}

def train_one(station,fold,arm,seed,train,val,output,pack_sha,norm):
    run=output/station/f'fold{fold}'/arm/f'seed{seed}'; run.mkdir(parents=True,exist_ok=False)
    base_seed=seed+fold*100003
    seed_all(base_seed)
    model=Head(use_lead=arm_uses_lead(arm)).cuda()
    init_hash=hashlib.sha256(b''.join(k.encode()+v.detach().cpu().contiguous().numpy().tobytes() for k,v in sorted(model.state_dict().items()))).hexdigest()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS,eta_min=1e-5)
    n=len(train['ghi']); updates=0; curve=[]; t0=time.monotonic()
    cot_name='none' if arm.startswith('none_') else 'p_xc'
    for ep in range(EPOCHS):
        order=np.random.default_rng(base_seed+ep).permutation(n)
        model.train(); sse=0.0
        for lo in range(0,n,BATCH):
            ids=torch.as_tensor(order[lo:lo+BATCH],device='cuda',dtype=torch.long)
            pred=model(train['agri'][ids],train['geometry'][ids],train[cot_name][ids],
                       train['lead_emb'][ids] if model.use_lead else None)
            loss=(((pred*train['clear'][ids]-train['ghi'][ids])/1000.0)**2).mean()
            if not torch.isfinite(loss): raise FloatingPointError(f'nonfinite {station}/fold{fold}/{arm}/{seed}')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0,error_if_nonfinite=True)
            optimizer.step(); sse+=float(loss.detach())*len(ids); updates+=1
        scheduler.step(); rec={'epoch':ep+1,'train_scaled_mse':sse/n,'updates':updates}
        curve.append(rec)
        if ep in (0,4,9,19,29): print(json.dumps({'station':station,'fold':fold,'arm':arm,'seed':seed,**rec}),flush=True)
    model.eval(); pred=[]
    with torch.inference_mode():
        for lo in range(0,len(val['ghi']),2048):
            sl=slice(lo,min(lo+2048,len(val['ghi'])))
            pred.append((model(val['agri'][sl],val['geometry'][sl],val[cot_name][sl],val['lead_emb'][sl] if model.use_lead else None).double()*val['clear'][sl].double()).cpu().numpy())
    pred=np.concatenate(pred).astype(np.float64); truth=val['ghi'].double().cpu().numpy()
    torch.save({'state_dict':model.state_dict(),'station':station,'fold':fold,'arm':arm,'seed':seed,
                'base_seed':base_seed,'epochs':EPOCHS,'batch':BATCH,'pack_sha256':pack_sha,
                'initial_state_sha256':init_hash,'normalization':norm},run/'epoch_030.pt')
    np.savez_compressed(run/'oof_predictions.npz',prediction_ghi=pred,truth_ghi=truth)
    (run/'training_curve.json').write_text(json.dumps(curve,indent=2)+'\n',encoding='utf-8')
    return {'station':station,'fold':fold,'arm':arm,'seed':seed,'n_train':n,'n_oof':len(truth),
            'updates':updates,'epochs':EPOCHS,'initial_state_sha256':init_hash,
            'checkpoint_sha256':sha(run/'epoch_030.pt'),'prediction_sha256':sha(run/'oof_predictions.npz'),
            'elapsed_seconds':time.monotonic()-t0}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--bank',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--audit-only',action='store_true'); ap.add_argument('--split-manifest',type=Path)
    a=ap.parse_args(); mpath=a.bank/'bank_manifest.json'
    if not mpath.is_file(): raise FileNotFoundError(mpath)
    bank=json.loads(mpath.read_text(encoding='utf-8'))
    if bank.get('state')!='COMPLETE_DIRECT_COT_FORECAST_ROW_BANK' or bank.get('train_partition')!='H': raise RuntimeError('H-only forecast row-bank required')
    for flag in ('test_payload_opened','buffer_payload_opened','future_observed_agri_used_as_input','cpp_used_as_input'):
        if bank.get(flag) is not False: raise RuntimeError(f'forbidden source flag {flag}')
    if bank.get('agri_source')!='fold1-SimVP-U-only-forecast': raise RuntimeError('unexpected forecast-AGRI source')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    keys={st:read_jsonl(a.bank/st/'train'/'keys.jsonl') for st in STATIONS}
    for st in STATIONS:
        d=a.bank/st/'train'
        for rel in ('keys.jsonl','agri_raw.npy','geometry_raw.npy','ghi.npy','clear.npy','cot_raw/p_xc.npy'):
            key=f'{st}/train/{rel}'; path=d/rel; expected=bank.get('artifact_sha256',{}).get(key)
            if not expected or sha(path)!=expected: raise RuntimeError(f'row-bank artifact hash mismatch: {key}')
    fold_by_date=assign_folds(keys); fold_audit=audit_folds(keys,fold_by_date)
    manifest={'state':'H_CV_SPLIT_AUDITED','arms':list(ARMS),'stations':list(STATIONS),'folds':5,
      'base_seeds':list(BASE_SEEDS),'epochs':EPOCHS,'batch_size':BATCH,'pack_manifest_sha256':sha(mpath),
      'source_bank_manifest_sha256':sha(a.bank/'bank_manifest.json'),'test_payload_opened':False,
      'buffer_payload_opened':False,'future_observed_agri_used_as_input':False,'cpp_used_as_input':False,
      'normalization_scope':'fit independently inside each H training fold only',
      'fold_assignment':'target BJT calendar two-day block, month-rotated round robin; no labels/scores used',
      'date_gap_days':1,'folds_detail':fold_audit,
      'date_fold':{d.isoformat():int(f) for d,f in sorted(fold_by_date.items())}}
    out_manifest=a.split_manifest or (a.output.with_suffix('.split_manifest.json'))
    out_manifest.parent.mkdir(parents=True,exist_ok=True); out_manifest.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'state':'H_CV_SPLIT_AUDITED','split_manifest':str(out_manifest),'folds':fold_audit},indent=2),flush=True)
    if a.audit_only: return
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    manifest['state']='RUNNING_DIRECT_COT_LEAD_H_CV'; manifest['runs']=[]
    for st in STATIONS:
        d=a.bank/st/'train'; keys_st=keys[st]
        arrays={name:np.load(d/name,mmap_mode='r',allow_pickle=False) for name in ('agri_raw.npy','geometry_raw.npy','ghi.npy','clear.npy')}
        cot=np.load(d/'cot_raw'/'p_xc.npy',mmap_mode='r',allow_pickle=False)
        leads=np.asarray([int(k['lead_minutes']) for k in keys_st],dtype=np.float32)
        dates=[bjt_date(k) for k in keys_st]
        raw={'agri':arrays['agri_raw.npy'],'geometry':arrays['geometry_raw.npy'],'ghi':arrays['ghi.npy'],
             'clear':arrays['clear.npy'],'p_xc':cot,'lead':leads}
        n=len(keys_st)
        if any(len(x)!=n for x in raw.values()): raise RuntimeError(f'row count mismatch {st}')
        if any(not np.isfinite(np.asarray(raw[k])).all() for k in raw): raise RuntimeError(f'nonfinite H data {st}')
        for fold in range(5):
            held={d for d,f in fold_by_date.items() if f==fold}
            blocked={d for d in fold_by_date if any(abs((d-h).days)<=1 for h in held)}
            val_idx=np.asarray([i for i,d0 in enumerate(dates) if d0 in held],dtype=np.int64)
            train_idx=np.asarray([i for i,d0 in enumerate(dates) if d0 not in blocked],dtype=np.int64)
            normed=normalize_fold(raw,train_idx,val_idx); norm=normed.pop('norm')
            tr=tensor_data(normed,train_idx); va=tensor_data(normed,val_idx)
            # Pair all four arms on the same fold/seed sample order. The two COT
            # availability contrasts are the only planned architectural factors.
            for seed in BASE_SEEDS:
                refs={}
                for arm in ARMS:
                    rec=train_one(st,fold,arm,seed,tr,va,a.output,sha(mpath),norm)
                    rec['normalization_sha256']=hashlib.sha256(json.dumps(norm,sort_keys=True).encode()).hexdigest()
                    refs[arm]=rec['initial_state_sha256']
                    manifest['runs'].append(rec)
                    (a.output/'progress.json').write_text(json.dumps({'state':'RUNNING_DIRECT_COT_LEAD_H_CV',
                        'completed_runs':len(manifest['runs']),'expected_runs':5*len(STATIONS)*len(BASE_SEEDS)*len(ARMS),
                        'runs':manifest['runs'],'test_payload_opened':False,'buffer_payload_opened':False,
                        'future_observed_agri_used_as_input':False,'cpp_used_as_input':False},indent=2)+'\n',encoding='utf-8')
                    print(json.dumps({'completed_run':rec}),flush=True)
                if refs['none_no_lead']!=refs['pxc_no_lead'] or refs['none_lead']!=refs['pxc_lead']:
                    raise RuntimeError(f'paired initialization mismatch {st}/fold{fold}/seed{seed}')
            # save fold-level row identity once; every arm must match by construction
            fold_dir=a.output/st/f'fold{fold}'; fold_dir.mkdir(parents=True,exist_ok=True)
            (fold_dir/'oof_keys.jsonl').write_text('\n'.join(json.dumps(keys_st[i],sort_keys=True) for i in val_idx)+'\n',encoding='utf-8')
            (fold_dir/'date_manifest.json').write_text(json.dumps({'heldout_dates':[d.isoformat() for d in sorted(held)],
                'buffer_dates_in_H':[d.isoformat() for d in sorted(blocked-held)],'train_rows':len(train_idx),'oof_rows':len(val_idx),
                'normalization':norm},indent=2)+'\n',encoding='utf-8')
    manifest['state']='COMPLETE_DIRECT_COT_LEAD_H_CV'; manifest['split_manifest_sha256']=sha(out_manifest)
    (a.output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'state':manifest['state'],'n_runs':len(manifest['runs']),'test_payload_opened':False}),flush=True)

if __name__=='__main__': main()
