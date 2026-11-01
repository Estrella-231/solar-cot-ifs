"""H-only date-block CV for train-fold monotone calibration of forecast COT."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, random, time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

STATIONS=('sili','zhujia'); ARMS=('none','pxc_raw','pxc_cal'); SEEDS=(42,43,44)
EPOCHS=30; BATCH=512; SOZ80=float(np.cos(np.deg2rad(80.0)))
KNOTS=(0.0,.5,1.0,2.0,3.0,4.6); IDENTITY_LAMBDA=1e-3

def sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def read_jsonl(p): return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def bjt_date(k): return (datetime.fromisoformat(k['target_time_utc'])+timedelta(hours=8)).date()
def lead_embedding(leads):
    x=torch.as_tensor(leads,dtype=torch.float32).reshape(-1,1)/240.0
    return torch.cat([v for f in (1,2,4,8) for v in (torch.sin(2*math.pi*f*x),torch.cos(2*math.pi*f*x))],1).numpy()

class MonotoneLogCot(nn.Module):
    """Fixed-knot positive-slope spline, anchored at f(0)=0."""
    def __init__(self,mu,sd,trainable):
        super().__init__(); knots=torch.tensor(KNOTS,dtype=torch.float32)
        self.register_buffer('knots',knots); self.register_buffer('mu',torch.tensor(float(mu)))
        self.register_buffer('sd',torch.tensor(float(sd)))
        theta=math.log(math.expm1(1.0))
        self.raw_slopes=nn.Parameter(torch.full((len(KNOTS),),theta),requires_grad=trainable)
    def slopes(self): return F.softplus(self.raw_slopes)
    def transform_log(self,u):
        s=self.slopes(); out=torch.zeros_like(u)
        for i in range(len(KNOTS)-1):
            width=KNOTS[i+1]-KNOTS[i]
            out=out+s[i]*torch.clamp(u-KNOTS[i],min=0.0,max=width)
        out=out+s[-1]*torch.clamp(u-KNOTS[-1],min=0.0)
        return out
    def forward(self,z,sun):
        u=z*self.sd+self.mu
        mapped=(self.transform_log(u)-self.mu)/self.sd
        return torch.where(sun,mapped,torch.zeros_like(mapped))
    def identity_penalty(self): return (self.slopes()-1.0).square().mean()

class Head(nn.Module):
    def __init__(self,mu,sd,train_cal,use_cot):
        super().__init__(); self.use_cot=use_cot
        self.agri_encoder=nn.Sequential(nn.Conv2d(15,32,1),nn.SiLU(),nn.Conv2d(32,32,1),nn.SiLU())
        self.cot_map=MonotoneLogCot(mu,sd,train_cal)
        self.cot_encoder=nn.Sequential(nn.Linear(36,36),nn.SiLU())
        self.readout=nn.Sequential(nn.Linear((32+4)*9+8,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1),nn.Softplus())
    def forward(self,agri,geometry,cot4,lead,cot_sun):
        if cot4.shape!=(len(agri),4,3,3) or cot_sun.shape!=(len(agri),1,3,3) or lead.shape!=(len(agri),8):
            raise ValueError('monotone-calibration input shape mismatch')
        ag=self.agri_encoder(torch.cat((agri,geometry),1)).flatten(1)
        feat=cot4
        if self.use_cot:
            z=self.cot_map(cot4[:,0:1],cot_sun.bool())
            feat=torch.cat((z,cot4[:,1:]),1)
        co=self.cot_encoder(feat.flatten(1))
        return self.readout(torch.cat((ag,co,lead),1)).squeeze(-1)

def build_sources(bank_dir,rows,inference_dir,station_index):
    lead=np.asarray([int(r['lead_minutes']) for r in rows],dtype=np.int64)
    if not np.isin(lead,np.arange(15,241,15)).all(): raise ValueError('unexpected forecast lead')
    pxc=np.load(inference_dir/'cot_p_xc.npy',mmap_mode='r',allow_pickle=False)
    ixrows=list(csv.DictReader((inference_dir/'index.csv').open(encoding='utf-8',newline='')))
    if len(ixrows)!=pxc.shape[0] or pxc.ndim!=6: raise ValueError('inference index/array mismatch')
    if pxc.shape[1:]!=(2,16,1,16,16): raise ValueError('future COT shape mismatch')
    seq_to_index={r['seq_id']:int(r['sequence_index_within_split']) for r in ixrows}; seq_idx=[]
    for r in rows:
        init=datetime.fromisoformat(r['init_time_utc']); first=init-timedelta(minutes=105)
        sid='hunan_'+first.strftime('%Y%m%d%H%M')
        if sid not in seq_to_index: raise ValueError(f'missing causal H sequence {sid}')
        seq_idx.append(seq_to_index[sid])
    bank_p=np.load(bank_dir/'p_xc.npy',mmap_mode='r',allow_pickle=False)
    if bank_p.shape!=(len(rows),1,3,3): raise ValueError('rowbank COT shape mismatch')
    out=np.empty((len(rows),1,3,3),np.float32)
    for i,(si,lm) in enumerate(zip(seq_idx,lead)):
        li=lm//15-1; out[i,0]=pxc[si,station_index,li,0,7:10,7:10]
        if not np.allclose(out[i,0],bank_p[i,0],rtol=2e-5,atol=2e-5):
            raise ValueError(f'rowbank/inference COT mismatch at row {i}')
    if not np.isfinite(out).all() or (out<0).any(): raise ValueError('invalid predicted COT')
    return out,lead

def fit_normalize(raw,sources,train_idx):
    ag,geo=raw['agri'],raw['geometry']; am=ag[train_idx].mean((0,2,3),dtype=np.float64).astype(np.float32)
    ast=np.maximum(ag[train_idx].std((0,2,3),dtype=np.float64),1e-6).astype(np.float32)
    gm=geo[train_idx].mean((0,2,3),dtype=np.float64).astype(np.float32); gst=np.maximum(geo[train_idx].std((0,2,3),dtype=np.float64),1e-6).astype(np.float32)
    sun=geo[:,0]>SOZ80; logged=np.log1p(np.maximum(sources[:,0],0)).astype(np.float32); fit=logged[train_idx][sun[train_idx]]
    if fit.size<1000: raise RuntimeError('insufficient sunlit fold-train COT pixels')
    mu=float(fit.mean(dtype=np.float64)); sd=max(float(fit.std(dtype=np.float64)),1e-6)
    z=((logged-mu)/sd).astype(np.float32); z[~sun]=0
    agz=((ag-am[None,:,None,None])/ast[None,:,None,None]).astype(np.float32)
    gz=((geo-gm[None,:,None,None])/gst[None,:,None,None]).astype(np.float32)
    none=np.zeros((len(z),4,3,3),np.float32); cot=np.zeros_like(none); cot[:,0]=z
    lead=lead_embedding(raw['lead'])
    arr={'agri':agz,'geometry':gz,'none':none,'pxc_raw':cot,'pxc_cal':cot.copy(),
         'cot_sun':sun[:,None].astype(np.float32),'ghi':raw['ghi'].astype(np.float32),'clear':raw['clear'].astype(np.float32),'lead':lead.astype(np.float32)}
    norm={'agri_mean':am.tolist(),'agri_std':ast.tolist(),'geometry_mean':gm.tolist(),'geometry_std':gst.tolist(),
          'cot_log1p_mean':mu,'cot_log1p_std':sd,'cot_fit_sunlit_pixels':int(fit.size),'fit_dates_only':True,'knot_log1p':list(KNOTS)}
    return arr,norm

def seed_all(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def train_one(st,fold,arm,seed,tr,va,va_keys,out,bank_sha,norm):
    run=out/st/f'fold{fold}'/arm/f'seed{seed}'; run.mkdir(parents=True,exist_ok=False)
    base=seed+fold*100003; seed_all(base); train_cal=arm=='pxc_cal'; use_cot=arm!='none'
    model=Head(norm['cot_log1p_mean'],norm['cot_log1p_std'],train_cal,use_cot).cuda()
    init=hashlib.sha256(b''.join(k.encode()+v.detach().cpu().contiguous().numpy().tobytes() for k,v in sorted(model.state_dict().items()))).hexdigest()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS,eta_min=1e-5)
    n=len(tr['ghi']); updates=0; curve=[]; t0=time.monotonic()
    for ep in range(EPOCHS):
        order=np.random.default_rng(base+ep).permutation(n); model.train(); loss_sum=0.0
        for lo in range(0,n,BATCH):
            ids=torch.as_tensor(order[lo:lo+BATCH],device='cuda',dtype=torch.long)
            pred=model(tr['agri'][ids],tr['geometry'][ids],tr[arm][ids],tr['lead'][ids],tr['cot_sun'][ids])
            loss=(((pred*tr['clear'][ids]-tr['ghi'][ids])/1000.0)**2).mean()
            if train_cal: loss=loss+IDENTITY_LAMBDA*model.cot_map.identity_penalty()
            if not torch.isfinite(loss): raise FloatingPointError(f'nonfinite loss {st}/{fold}/{arm}/{seed}')
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0,error_if_nonfinite=True); opt.step()
            loss_sum+=float(loss.detach())*len(ids); updates+=1
        sch.step(); rec={'epoch':ep+1,'train_scaled_mse_plus_prior':loss_sum/n,'updates':updates}
        if train_cal: rec['slopes']=model.cot_map.slopes().detach().cpu().tolist()
        curve.append(rec)
        if ep in (0,4,9,19,29): print(json.dumps({'station':st,'fold':fold,'arm':arm,'seed':seed,**rec}),flush=True)
    model.eval(); preds=[]
    with torch.inference_mode():
        for lo in range(0,len(va['ghi']),2048):
            ix=slice(lo,min(lo+2048,len(va['ghi'])))
            preds.append((model(va['agri'][ix],va['geometry'][ix],va[arm][ix],va['lead'][ix],va['cot_sun'][ix]).double()*va['clear'][ix].double()).cpu().numpy())
    pred=np.concatenate(preds).astype(np.float64); truth=va['ghi'].double().cpu().numpy()
    torch.save({'state_dict':model.state_dict(),'station':st,'fold':fold,'arm':arm,'seed':seed,'base_seed':base,'epochs':EPOCHS,'batch':BATCH,'bank_manifest_sha256':bank_sha,'initial_state_sha256':init,'normalization':norm,'knot_log1p':list(KNOTS),'final_slopes':model.cot_map.slopes().detach().cpu().tolist()},run/'epoch_030.pt')
    np.savez_compressed(run/'oof_predictions.npz',prediction_ghi=pred,truth_ghi=truth)
    (run/'training_curve.json').write_text(json.dumps(curve,indent=2)+'\n',encoding='utf-8'); (run/'oof_keys.jsonl').write_text('\n'.join(json.dumps(k,sort_keys=True) for k in va_keys)+'\n',encoding='utf-8')
    return {'station':st,'fold':fold,'arm':arm,'seed':seed,'n_train':n,'n_oof':len(truth),'epochs':EPOCHS,'updates':updates,'initial_state_sha256':init,'checkpoint_sha256':sha(run/'epoch_030.pt'),'prediction_sha256':sha(run/'oof_predictions.npz'),'elapsed_seconds':time.monotonic()-t0}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--bank',type=Path,required=True); p.add_argument('--inference',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--smoke',action='store_true'); a=p.parse_args()
    bm=json.loads((a.bank/'bank_manifest.json').read_text(encoding='utf-8')); im=json.loads((a.inference/'inference_manifest.json').read_text(encoding='utf-8'))
    if bm.get('state')!='COMPLETE_DIRECT_COT_FORECAST_ROW_BANK' or bm.get('train_partition')!='H': raise RuntimeError('H-only rowbank required')
    if im.get('state')!='COMPLETE_CAUSAL_DIRECT_COT_HV_INFERENCE': raise RuntimeError('causal H/V forecast inference receipt required')
    for m in (bm,im):
        for k in ('test_payload_opened','buffer_payload_opened','future_observed_agri_used_as_input','cpp_used_as_inference_input','cpp_used_as_input'):
            if k in m and m[k] is not False: raise RuntimeError(f'forbidden source flag {k}')
    if im.get('future_observed_agri_read') is not False or im.get('cpp_read') is not False or im.get('ghi_labels_read') is not False: raise RuntimeError('forecast inference read a forbidden target')
    if im.get('stations')!=list(STATIONS): raise RuntimeError('station order mismatch')
    if a.smoke:
        rows_by={s:read_jsonl(a.bank/s/'train'/'keys.jsonl') for s in STATIONS}
        for s,si in zip(STATIONS,(0,1)):
            d=a.bank/s/'train'
            for name in ('keys.jsonl','agri_raw.npy','geometry_raw.npy','ghi.npy','clear.npy'):
                if sha(d/name)!=bm.get('artifact_sha256',{}).get(f'{s}/train/{name}'): raise RuntimeError(f'bank hash mismatch {s}/{name}')
            if sha(d/'p_xc.npy')!=bm.get('artifact_sha256',{}).get(f'{s}/train/cot_raw/p_xc.npy'): raise RuntimeError(f'P-XC rowbank hash mismatch {s}')
            if sha(a.inference/'H'/'cot_p_xc.npy')!=im.get('artifact_sha256',{}).get('H/cot_p_xc.npy'): raise RuntimeError('P-XC inference hash mismatch')
            src,lead=build_sources(d,rows_by[s],a.inference/'H',si); raw={k:np.load(d/f'{nm}.npy',mmap_mode='r',allow_pickle=False)[:2048] for k,nm in (('agri','agri_raw'),('geometry','geometry_raw'),('ghi','ghi'),('clear','clear'))}; raw['lead']=lead[:2048]
            arr,norm=fit_normalize(raw,src[:2048],np.arange(2048)); model=Head(norm['cot_log1p_mean'],norm['cot_log1p_std'],True,True)
            x=torch.linspace(0,5,1000); y=model.cot_map.transform_log(x)
            if not torch.all(y[1:]>=y[:-1]) or abs(float(model.cot_map.transform_log(torch.tensor([0.]))[0]))>1e-7: raise RuntimeError('calibrator monotonicity/zero anchor failed')
            u=torch.rand(8,1,3,3)*6; z=(u-norm['cot_log1p_mean'])/norm['cot_log1p_std']; mask=torch.ones_like(z,dtype=torch.bool); ident=model.cot_map(z,mask)
            if not torch.allclose(ident,z,atol=2e-6,rtol=2e-6): raise RuntimeError('identity initialization failed')
            out=model(torch.zeros(8,13,3,3),torch.zeros(8,2,3,3),torch.zeros(8,4,3,3),torch.zeros(8,8),torch.ones(8,1,3,3))
            if out.shape!=(8,) or not torch.isfinite(out).all(): raise RuntimeError('head smoke failed')
        print(json.dumps({'state':'PASS_MONOCAL_HCV_INPUT_SPLINE_AND_HEAD_SMOKE','sili_rows':len(rows_by['sili']),'zhujia_rows':len(rows_by['zhujia']),'test_payload_opened':False,'buffer_payload_opened':False}),flush=True); return
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.mkdir(parents=True); rows_by={s:read_jsonl(a.bank/s/'train'/'keys.jsonl') for s in STATIONS}
    if sha(a.inference/'H'/'cot_p_xc.npy')!=im.get('artifact_sha256',{}).get('H/cot_p_xc.npy'): raise RuntimeError('P-XC inference hash mismatch')
    ixrows=list(csv.DictReader((a.inference/'H'/'index.csv').open(encoding='utf-8',newline='')))
    if len(ixrows)!=int(im.get('sequence_counts',{}).get('H',-1)): raise RuntimeError('H inference index count mismatch')
    all_dates={bjt_date(k) for rs in rows_by.values() for k in rs}; months=sorted({x.strftime('%Y-%m') for x in all_dates}); mi={m:i for i,m in enumerate(months)}
    fold_by_date={x.isoformat():(((x.day-1)//2)+mi[x.strftime('%Y-%m')])%5 for x in sorted(all_dates)}; runs=[]
    for station,si in zip(STATIONS,(0,1)):
        d=a.bank/station/'train'; rows=rows_by[station]
        for name in ('keys.jsonl','agri_raw.npy','geometry_raw.npy','ghi.npy','clear.npy'):
            if sha(d/name)!=bm.get('artifact_sha256',{}).get(f'{station}/train/{name}'): raise RuntimeError(f'bank hash mismatch {station}/{name}')
        if sha(d/'p_xc.npy')!=bm.get('artifact_sha256',{}).get(f'{station}/train/cot_raw/p_xc.npy'): raise RuntimeError(f'P-XC rowbank hash mismatch {station}')
        raw={k:np.load(d/f'{nm}.npy',mmap_mode='r',allow_pickle=False) for k,nm in (('agri','agri_raw'),('geometry','geometry_raw'),('ghi','ghi'),('clear','clear'))}
        src,leads=build_sources(d,rows,a.inference/'H',si); raw['lead']=leads; dates=[bjt_date(k) for k in rows]
        for fold in range(5):
            held={datetime.fromisoformat(x).date() for x,f in fold_by_date.items() if f==fold}; blocked={x for x in dates if any(abs((x-h).days)<=1 for h in held)}
            tridx=np.asarray([i for i,x in enumerate(dates) if x not in blocked],np.int64); vaidx=np.asarray([i for i,x in enumerate(dates) if x in held],np.int64)
            arr,norm=fit_normalize(raw,src,tridx)
            tr={k:torch.as_tensor(v[tridx],device='cuda',dtype=torch.float32) for k,v in arr.items()}; va={k:torch.as_tensor(v[vaidx],device='cuda',dtype=torch.float32) for k,v in arr.items()}
            for seed in SEEDS:
                hashes=[]
                for arm in ARMS:
                    rec=train_one(station,fold,arm,seed,tr,va,[rows[i] for i in vaidx],a.output,sha(a.bank/'bank_manifest.json'),norm); runs.append(rec); hashes.append(rec['initial_state_sha256'])
                    state={'state':'RUNNING_COT_MONOCAL_H_CV','completed_runs':len(runs),'expected_runs':90,'runs':runs,'test_payload_opened':False,'buffer_payload_opened':False,'future_observed_agri_used_as_input':False,'cpp_used_as_input':False}
                    (a.output/'progress.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8'); print(json.dumps({'completed_run':rec}),flush=True)
                if len(set(hashes))!=1: raise RuntimeError('paired initialization mismatch')
    manifest={'state':'COMPLETE_COT_MONOCAL_H_CV','arms':list(ARMS),'stations':list(STATIONS),'folds':5,'seeds':list(SEEDS),'epochs':EPOCHS,'batch_size':BATCH,'identity_lambda':IDENTITY_LAMBDA,'knots_log1p':list(KNOTS),'runs':runs,'date_fold':fold_by_date,'bank_manifest_sha256':sha(a.bank/'bank_manifest.json'),'inference_manifest_sha256':sha(a.inference/'inference_manifest.json'),'fold_assignment':'same H calendar two-day date folds; ±1 BJT-day training gap','normalization_scope':'each fold training dates only; COT fit on sunlit train pixels only','calibration_scope':'one shared monotone map per station/fold/seed; trained only with H train GHI; forecast-time causal P-XC COT only','test_payload_opened':False,'buffer_payload_opened':False,'future_observed_agri_used_as_input':False,'cpp_used_as_input':False}
    (a.output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8'); print(json.dumps({'state':manifest['state'],'runs':len(runs)}),flush=True)
if __name__=='__main__': main()
