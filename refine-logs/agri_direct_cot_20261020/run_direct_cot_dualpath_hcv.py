"""H-only 5-fold CV: test disagreement-aware fusion of two causal future-COT paths."""
from __future__ import annotations
import argparse, hashlib, json, math, random, time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import nn

STATIONS = ("sili", "zhujia")
ARMS = ("none", "pxc", "simvp_r", "dual")
SEEDS = (42, 43, 44)
EPOCHS, BATCH = 30, 512
SOZ80 = float(np.cos(np.deg2rad(80.0)))

def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def read_jsonl(p): return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def bjt_date(k): return (datetime.fromisoformat(k['target_time_utc'])+timedelta(hours=8)).date()

def lead_embedding(leads):
    x=torch.as_tensor(leads,dtype=torch.float32).reshape(-1,1)/240.0
    return torch.cat([v for f in (1,2,4,8) for v in (torch.sin(2*math.pi*f*x),torch.cos(2*math.pi*f*x))],1).numpy()

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.agri_encoder=nn.Sequential(nn.Conv2d(15,32,1),nn.SiLU(),nn.Conv2d(32,32,1),nn.SiLU())
        self.cot_encoder=nn.Sequential(nn.Linear(36,36),nn.SiLU())
        self.readout=nn.Sequential(nn.Linear((32+4)*9+8,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1),nn.Softplus())
    def forward(self,agri,geometry,cot4,lead):
        if cot4.shape!=(len(agri),4,3,3) or lead.shape!=(len(agri),8): raise ValueError('dual-path COT/lead shape mismatch')
        ag=self.agri_encoder(torch.cat((agri,geometry),1)).flatten(1)
        co=self.cot_encoder(cot4.flatten(1))
        return self.readout(torch.cat((ag,co,lead),1)).squeeze(-1)

def build_dual_sources(bank_dir, rows, inference_dir, station_index):
    lead=np.asarray([int(r['lead_minutes']) for r in rows],dtype=np.int64)
    if not np.isin(lead,np.arange(15,241,15)).all(): raise ValueError('unexpected forecast lead')
    pxc=np.load(inference_dir/'cot_p_xc.npy',mmap_mode='r',allow_pickle=False)
    simvp=np.load(inference_dir/'cot_simvp_r.npy',mmap_mode='r',allow_pickle=False)
    index_rows=list(__import__('csv').DictReader((inference_dir/'index.csv').open(encoding='utf-8',newline='')))
    if len(index_rows)!=pxc.shape[0] or len(index_rows)!=simvp.shape[0]: raise ValueError('sequence-index/array cardinality mismatch')
    if pxc.ndim!=6 or simvp.ndim!=6: raise ValueError('forecast COT array rank mismatch')
    if pxc.shape[1:]!=(2,16,1,16,16) or simvp.shape!=pxc.shape: raise ValueError(f'COT output shape mismatch: {pxc.shape}/{simvp.shape}')
    # The row-bank's sequence index belongs to the wider GHI crosswalk. Map by
    # immutable initialization time to the causal 24-frame H inference sequence.
    seq_to_index={r['seq_id']:int(r['sequence_index_within_split']) for r in index_rows}
    seq_idx=[]
    for r in rows:
        init=datetime.fromisoformat(r['init_time_utc'])
        first=init-timedelta(minutes=105)
        seq_id='hunan_'+first.strftime('%Y%m%d%H%M')
        if seq_id not in seq_to_index: raise ValueError(f'init time is absent from H inference index: {seq_id}')
        seq_idx.append(seq_to_index[seq_id])
    seq_idx=np.asarray(seq_idx,dtype=np.int64)
    bank_p=np.load(bank_dir/'cot_raw'/'p_xc.npy',mmap_mode='r',allow_pickle=False)
    bank_s=np.load(bank_dir/'cot_raw'/'simvp_r.npy',mmap_mode='r',allow_pickle=False)
    if bank_p.shape!=(len(rows),1,3,3) or bank_s.shape!=bank_p.shape: raise ValueError('row-bank COT shape mismatch')
    sources=np.empty((len(rows),2,3,3),dtype=np.float32)
    for i,(si,lm) in enumerate(zip(seq_idx,lead)):
        li=lm//15-1
        sources[i,0]=pxc[si,station_index,li,0,7:10,7:10]
        sources[i,1]=simvp[si,station_index,li,0,7:10,7:10]
        if not np.allclose(sources[i,0],bank_p[i,0],rtol=2e-5,atol=2e-5): raise ValueError(f'P-XC row-bank mismatch at row {i}')
        if not np.allclose(sources[i,1],bank_s[i,0],rtol=2e-5,atol=2e-5): raise ValueError(f'SimVP-R row-bank mismatch at row {i}')
    if not np.isfinite(sources).all() or (sources<0).any(): raise ValueError('invalid forecast COT source')
    return sources,lead

def fit_normalize(raw, sources, train_idx):
    ag,geo,y,clear=raw['agri'],raw['geometry'],raw['ghi'],raw['clear']
    am=ag[train_idx].mean((0,2,3),dtype=np.float64).astype(np.float32); ast=np.maximum(ag[train_idx].std((0,2,3),dtype=np.float64),1e-6).astype(np.float32)
    gm=geo[train_idx].mean((0,2,3),dtype=np.float64).astype(np.float32); gst=np.maximum(geo[train_idx].std((0,2,3),dtype=np.float64),1e-6).astype(np.float32)
    sun=geo[:,0]>SOZ80
    logged=np.log1p(np.maximum(sources,0)).astype(np.float32)
    params=[]; z=[]
    for source in range(2):
        fit=logged[train_idx,source][sun[train_idx]]
        if fit.size<1000: raise RuntimeError('insufficient sunlit COT pixels for train-fold normalization')
        mu=float(fit.mean(dtype=np.float64)); sd=max(float(fit.std(dtype=np.float64)),1e-6)
        params.append((mu,sd)); zsrc=((logged[:,source]-mu)/sd).astype(np.float32); zsrc[~sun]=0; z.append(zsrc)
    agz=((ag-am[None,:,None,None])/ast[None,:,None,None]).astype(np.float32)
    gz=((geo-gm[None,:,None,None])/gst[None,:,None,None]).astype(np.float32)
    none=np.zeros((len(z[0]),4,3,3),dtype=np.float32); pxc_map=np.zeros_like(none); simvp_map=np.zeros_like(none); dual_map=np.zeros_like(none)
    pxc_map[:,0]=z[0]; simvp_map[:,0]=z[1]; dual_map[:,0]=z[0]; dual_map[:,1]=z[1]
    dual_map[:,2]=0.5*(z[0]+z[1]); dual_map[:,3]=np.abs(z[0]-z[1])
    lead=lead_embedding(raw['lead'])
    norm={'agri_mean':am.tolist(),'agri_std':ast.tolist(),'geometry_mean':gm.tolist(),'geometry_std':gst.tolist(),
          'cot_log1p_mean_pxc':params[0][0],'cot_log1p_std_pxc':params[0][1],
          'cot_log1p_mean_simvp_r':params[1][0],'cot_log1p_std_simvp_r':params[1][1],
          'cot_fit_pixels':int(np.count_nonzero(sun[train_idx])),'fit_days_only':True,
          'dual_features':['z_pxc','z_simvp_r','mean_z','abs_disagreement_z']}
    arr={'agri':agz,'geometry':gz,'none':none,'pxc':pxc_map,'simvp_r':simvp_map,'dual':dual_map,
         'ghi':y.astype(np.float32),'clear':clear.astype(np.float32),'lead':lead.astype(np.float32)}
    return arr,norm

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def train_one(st,fold,arm,seed,tr,va,val_keys,output,bank_sha,norm):
    run=output/st/f'fold{fold}'/arm/f'seed{seed}'; run.mkdir(parents=True,exist_ok=False)
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
    (run/'training_curve.json').write_text(json.dumps(curve,indent=2)+'\n',encoding='utf-8')
    (run/'oof_keys.jsonl').write_text('\n'.join(json.dumps(k,sort_keys=True) for k in val_keys)+'\n',encoding='utf-8')
    return {'station':st,'fold':fold,'arm':arm,'seed':seed,'n_train':n,'n_oof':len(truth),'epochs':EPOCHS,'updates':updates,'initial_state_sha256':init,'checkpoint_sha256':sha(run/'epoch_030.pt'),'prediction_sha256':sha(run/'oof_predictions.npz'),'elapsed_seconds':time.monotonic()-t0}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--bank',type=Path,required=True); ap.add_argument('--inference',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args()
    bm=json.loads((a.bank/'bank_manifest.json').read_text(encoding='utf-8')); im=json.loads((a.inference/'inference_manifest.json').read_text(encoding='utf-8'))
    if bm.get('state')!='COMPLETE_DIRECT_COT_FORECAST_ROW_BANK' or bm.get('train_partition')!='H': raise RuntimeError('H-only row bank required')
    if im.get('state')!='COMPLETE_CAUSAL_DIRECT_COT_HV_INFERENCE': raise RuntimeError('complete causal H/V inference receipt required')
    for m in (bm,im):
        for flag in ('test_payload_opened','buffer_payload_opened','future_observed_agri_used_as_input','cpp_used_as_inference_input','future_observed_agri_used_as_input','cpp_used_as_input'):
            if flag in m and m[flag] is not False: raise RuntimeError(f'forbidden input flag {flag}')
    if im.get('future_observed_agri_read') is not False or im.get('cpp_read') is not False or im.get('ghi_labels_read') is not False:
        raise RuntimeError('H/V forecast inference receipt indicates forbidden observed targets were read')
    if im.get('stations')!=list(STATIONS) or int(im.get('sequence_counts',{}).get('H',-1))<=0:
        raise RuntimeError('inference station ordering or H sequence count mismatch')
    if a.smoke:
        index_rows=list(__import__('csv').DictReader((a.inference/'H'/'index.csv').open(encoding='utf-8',newline='')))
        if len(index_rows)!=int(im['sequence_counts']['H']): raise RuntimeError('H sequence index count mismatch')
        for name in ('cot_p_xc.npy','cot_simvp_r.npy'):
            expected=im.get('artifact_sha256',{}).get(f'H/{name}')
            if not expected or sha(a.inference/'H'/name)!=expected: raise RuntimeError(f'inference H array SHA mismatch: {name}')
        rows_by_station={st:read_jsonl(a.bank/st/'train'/'keys.jsonl') for st in STATIONS}
        for st,si in zip(STATIONS,(0,1)):
            d=a.bank/st/'train'
            for name in ('keys.jsonl','agri_raw.npy','geometry_raw.npy','ghi.npy','clear.npy','cot_raw/p_xc.npy','cot_raw/simvp_r.npy'):
                expected=bm.get('artifact_sha256',{}).get(f'{st}/train/{name}')
                if not expected or sha(d/name)!=expected: raise RuntimeError(f'bank artifact SHA mismatch {st}/{name}')
            for name in ('cot_p_xc.npy','cot_simvp_r.npy'):
                expected=im.get('artifact_sha256',{}).get(f'H/{name}')
                if not expected or sha(a.inference/'H'/name)!=expected: raise RuntimeError(f'inference H array SHA mismatch: {name}')
            traj,leads=build_dual_sources(d,rows_by_station[st],a.inference/'H',si)
            if traj.shape!=(len(rows_by_station[st]),2,3,3) or not np.isin(leads,np.arange(15,241,15)).all(): raise RuntimeError('dual-source COT contract mismatch')
            n=min(2048,len(rows_by_station[st])); raw={k:np.load(d/f'{name}.npy',mmap_mode='r',allow_pickle=False)[:n] for k,name in (('agri','agri_raw'),('geometry','geometry_raw'),('ghi','ghi'),('clear','clear'))}; raw['lead']=leads[:n]
            arr,norm=fit_normalize(raw,traj[:n],np.arange(n,dtype=np.int64))
            if norm.get('dual_features')!=['z_pxc','z_simvp_r','mean_z','abs_disagreement_z'] or any(arr[k].shape!=(n,4,3,3) for k in ARMS): raise RuntimeError('dual-path feature encoding smoke failed')
        model=Head().eval(); pred=model(torch.zeros(4,13,3,3),torch.zeros(4,2,3,3),torch.zeros(4,4,3,3),torch.zeros(4,8))
        if pred.shape!=(4,) or not torch.isfinite(pred).all(): raise RuntimeError('model smoke output invalid')
        print(json.dumps({'state':'PASS_COT_DUALPATH_HCV_INPUT_AND_MODEL_SMOKE','sili_rows':len(rows_by_station['sili']),'zhujia_rows':len(rows_by_station['zhujia']),'h_sequences':len(index_rows),'test_payload_opened':False,'buffer_payload_opened':False}),flush=True)
        return
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for fixed-budget training')
    out=a.output
    if out.exists(): raise FileExistsError(out)
    if not a.smoke: out.mkdir(parents=True)
    for name in ('cot_p_xc.npy','cot_simvp_r.npy'):
        expected=im.get('artifact_sha256',{}).get(f'H/{name}')
        if not expected or sha(a.inference/'H'/name)!=expected: raise RuntimeError(f'inference H array SHA mismatch: {name}')
    expected_n=int(im.get('sequence_counts',{}).get('H',-1))
    index_rows=list(__import__('csv').DictReader((a.inference/'H'/'index.csv').open(encoding='utf-8',newline='')))
    if expected_n<=0 or len(index_rows)!=expected_n: raise RuntimeError('H sequence index count differs from inference receipt')
    rows_by_station={st:read_jsonl(a.bank/st/'train'/'keys.jsonl') for st in STATIONS}
    all_dates={bjt_date(k) for rs in rows_by_station.values() for k in rs}
    months=sorted({x.strftime('%Y-%m') for x in all_dates}); mi={m:i for i,m in enumerate(months)}
    fold_by_date={x.isoformat():(((x.day-1)//2)+mi[x.strftime('%Y-%m')])%5 for x in sorted(all_dates)}
    all_runs=[]
    for station,si in zip(STATIONS,(0,1)):
        d=a.bank/station/'train'; rows=rows_by_station[station]
        for name in ('keys.jsonl','agri_raw.npy','geometry_raw.npy','ghi.npy','clear.npy','cot_raw/p_xc.npy','cot_raw/simvp_r.npy'):
            expected=bm.get('artifact_sha256',{}).get(f'{station}/train/{name}')
            if not expected or sha(d/name)!=expected: raise RuntimeError(f'bank artifact SHA mismatch {station}/{name}')
        raw={k:np.load(d/f'{k}.npy',mmap_mode='r',allow_pickle=False) for k in ('agri_raw','geometry_raw','ghi','clear')}
        traj,leads=build_dual_sources(d,rows,a.inference/'H',si); raw={'agri':raw['agri_raw'],'geometry':raw['geometry_raw'],'ghi':raw['ghi'],'clear':raw['clear'],'lead':leads}
        dates=[bjt_date(r) for r in rows]
        if any(fold_by_date[x.isoformat()]<0 for x in dates): raise RuntimeError('date not assigned')
        for fold in range(5):
            held={datetime.fromisoformat(x).date() for x,f in fold_by_date.items() if f==fold}
            blocked={x for x in dates if any(abs((x-h).days)<=1 for h in held)}
            tridx=np.asarray([i for i,x in enumerate(dates) if x not in blocked],dtype=np.int64); vaidx=np.asarray([i for i,x in enumerate(dates) if x in held],dtype=np.int64)
            arr,norm=fit_normalize(raw,traj,tridx)
            tr={k:torch.as_tensor(v[tridx],device='cuda',dtype=torch.float32) for k,v in arr.items()}
            va={k:torch.as_tensor(v[vaidx],device='cuda',dtype=torch.float32) for k,v in arr.items()}
            for seed in SEEDS:
                init_sha={}
                for arm in ARMS:
                    rec=train_one(station,fold,arm,seed,tr,va,[rows[i] for i in vaidx], out,sha(a.bank/'bank_manifest.json'),norm)
                    all_runs.append(rec); init_sha[arm]=rec['initial_state_sha256']
                    progress={'state':'RUNNING_COT_DUALPATH_H_CV','completed_runs':len(all_runs),'expected_runs':120,'runs':all_runs,'test_payload_opened':False,'buffer_payload_opened':False,'future_observed_agri_used_as_input':False,'cpp_used_as_input':False}
                    (out/'progress.json').write_text(json.dumps(progress,indent=2)+'\n',encoding='utf-8')
                    print(json.dumps({'completed_run':rec}),flush=True)
                if len(set(init_sha.values()))!=1: raise RuntimeError('paired initialization mismatch')
    manifest={'state':'COMPLETE_COT_DUALPATH_H_CV','arms':list(ARMS),'stations':list(STATIONS),'folds':5,'seeds':list(SEEDS),'epochs':EPOCHS,'batch_size':BATCH,'runs':all_runs,'date_fold':fold_by_date,'bank_manifest_sha256':sha(a.bank/'bank_manifest.json'),'inference_manifest_sha256':sha(a.inference/'inference_manifest.json'),'fold_assignment':'same H calendar two-day folds as lead-conditioned H-CV; ±1 BJT-day training gap','normalization_scope':'each fold training dates only; COT maps fit only on sunlit train pixels','cot_representation':'normalized P-XC, normalized SimVP-R, their mean, and absolute disagreement','test_payload_opened':False,'buffer_payload_opened':False,'future_observed_agri_used_as_input':False,'cpp_used_as_input':False}
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8'); print(json.dumps({'state':manifest['state'],'runs':len(all_runs)}),flush=True)

if __name__=='__main__': main()
