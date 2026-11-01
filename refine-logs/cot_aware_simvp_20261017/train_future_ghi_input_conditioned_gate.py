#!/usr/bin/env python3
"""Train an input-conditioned gate on future COT; preserve ACREL history and GHI contract."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "experiments" / "high_kt_loss_20261016"))
from train_loss_pair import IntervalHead
from train_future_ghi_heads_cot_reliability import load_station, seed_all, sha

class InputConditionedGate(nn.Module):
    """Identity-initialized 0..2 multiplier for forecast COT anomalies only."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(47, 32), nn.SiLU(), nn.Linear(32, 16), nn.SiLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, agri, geometry, cot, history_mask, frame_lead_minutes):
        # Inputs: [N,3,13,3,3], [N,3,2,3,3], [N,3,1,3,3].
        center_a = agri[..., 1, 1]
        mean_a = agri.mean(dim=(-1, -2))
        spread_a = agri.std(dim=(-1, -2), unbiased=False)
        center_g = geometry[..., 1, 1]
        mean_g = geometry.mean(dim=(-1, -2))
        center_c = cot[..., 1, 1]
        mean_c = cot.mean(dim=(-1, -2))
        spread_c = cot.std(dim=(-1, -2), unbiased=False)
        lead = frame_lead_minutes.clamp(0, 240).to(agri.dtype).unsqueeze(-1) / 240.0
        features = torch.cat((center_a, mean_a, spread_a, center_g, mean_g,
                              center_c, mean_c, spread_c, lead), dim=-1)
        if features.shape[-1] != 47:
            raise ValueError(f"gate expects 47 features, got {features.shape}")
        scale = 1.0 + torch.tanh(self.net(features).squeeze(-1))  # [N,3], identity at init
        future = (~history_mask.bool()).squeeze(-1).squeeze(-1).squeeze(-1).to(cot.dtype)
        multiplier = 1.0 + future.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * \
            (scale - 1.0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        adjusted = cot * multiplier
        return adjusted, scale

class GatedIntervalHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = IntervalHead()
        self.gate = InputConditionedGate()

    def forward(self, agri, geometry, cot, history_mask, frame_lead_minutes, return_scale=False):
        adjusted, scale = self.gate(agri, geometry, cot, history_mask, frame_lead_minutes)
        prediction = self.base(agri, geometry, adjusted)
        return (prediction, scale) if return_scale else prediction

def frame_leads(keys):
    offsets = (-30, -15, 0)
    return torch.as_tensor([[int(k['lead_minutes']) + off for off in offsets] for k in keys], dtype=torch.float32)

def score(model, data, cot, history, leads, batch, return_scales=False):
    model.eval(); preds=[]; scales=[]
    with torch.inference_mode():
        for lo in range(0, len(data['ghi']), batch):
            sl=slice(lo,min(lo+batch,len(data['ghi'])))
            pred,scale=model(data['agri'][sl],data['geometry'][sl],cot[sl],history[sl],leads[sl],return_scale=True)
            preds.append((pred*data['clear'][sl]).cpu().numpy())
            if return_scales: scales.append(scale.cpu().numpy())
    p=np.concatenate(preds).astype(np.float64)
    return (p,np.concatenate(scales)) if return_scales else p

def scale_summary(scales, future_mask):
    x=np.asarray(scales)[np.asarray(future_mask).astype(bool)]
    if not len(x): raise ValueError('no forecast COT slots for gate summary')
    return {'n':int(len(x)),'mean':float(x.mean()),'std':float(x.std()),
            'p05':float(np.quantile(x,.05)),'p50':float(np.quantile(x,.50)),
            'p95':float(np.quantile(x,.95)),'fraction_below_0_5':float(np.mean(x<.5)),
            'fraction_above_1_5':float(np.mean(x>1.5))}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pack',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch-size',type=int,default=1792)
    a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    manifest=json.loads((a.pack/'pack_manifest.json').read_text(encoding='utf-8'))
    cal=json.loads(a.calibration.read_text(encoding='utf-8'))
    if manifest.get('state')!='COMPLETE_CAUSAL_FUTURE_GHI_HEAD_PACK' or manifest.get('train_forecast_oof') is not True or manifest.get('forecast_bank_oof_fold_count')!=5: raise ValueError('five-fold OOF training inputs required')
    if manifest.get('test_payload_opened') is not False or manifest.get('buffer_payload_opened') is not False or manifest.get('future_observed_agri_used_as_input') is not False or manifest.get('cpp_used_as_input') is not False: raise ValueError('forecast pack source gate failed')
    if cal.get('state')!='COMPLETE_CROSSFIT_TRAIN_ONLY_OOF_FUTURE_COT_RELIABILITY_CALIBRATION' or cal.get('forecast_pack_manifest_sha256')!=sha(a.pack/'pack_manifest.json'): raise ValueError('train-only calibration source mismatch')
    if cal.get('test_payload_opened') is not False or cal.get('buffer_payload_opened') is not False or cal.get('future_observed_agri_used_as_inference_input') is not False: raise ValueError('calibration exclusion gate failed')
    a.output_root.mkdir(parents=True,exist_ok=True)
    if (a.output_root/f'seed{a.seed}_manifest.json').exists() or (a.output_root/f'seed{a.seed}_running.json').exists() or (a.output_root/'sili'/f'seed{a.seed}').exists():
        raise FileExistsError(f'output already exists for seed {a.seed}: {a.output_root}')
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True
    rec={'state':'RUNNING','arm':'ACRELGATE','seed':a.seed,'epochs':a.epochs,'batch_size':a.batch_size,
         'stations':['sili','zhujia'],'train_forecast_oof':True,'forecast_bank_oof_fold_count':5,
         'split_id':manifest['split_id'],'transform':'identity-initialized input-conditioned 0..2 gate scales only future normalized ACREL COT anomalies; observed-history COT is unchanged',
         'gate_inputs':'per-frame predicted/observed-available AGRI center/3x3 mean/3x3 spread, geometry center/mean, calibrated COT center/mean/spread, forecast lead; no labels or future observed AGRI',
         'gate_regularization':'none; GHI supervised objective only; gate starts exactly at 1',
         'center_calibration_sha256':sha(a.calibration),'pack_manifest_sha256':sha(a.pack/'pack_manifest.json'),
         'test_payload_opened':False,'buffer_payload_opened':False,'future_observed_agri_used_as_inference_input':False,'cpp_used_as_input':False}
    (a.output_root/f'seed{a.seed}_running.json').write_text(json.dumps(rec,indent=2)+'\n',encoding='utf-8')
    summaries=[]
    for station in ('sili','zhujia'):
        tr=load_station(a.pack,'train',station,device,manifest,cal);va=load_station(a.pack,'val',station,device,manifest,cal)
        n=len(tr['ghi'])
        if n//a.batch_size<12: raise ValueError('batch size leaves fewer than 12 updates per epoch')
        leads_tr=frame_leads(tr['keys']).to(device);leads_va=frame_leads(va['keys']).to(device)
        history_tr=tr['history_cot_mask'];history_va=va['history_cot_mask']
        future_va=(~history_va.bool()).squeeze(-1).squeeze(-1).squeeze(-1).cpu().numpy()
        seed_all(a.seed);base=IntervalHead().to(device);initial={k:v.detach().clone() for k,v in base.state_dict().items()};del base
        rng=np.random.default_rng(a.seed);orders=[rng.permutation(n) for _ in range(a.epochs)]
        run=a.output_root/station/f'seed{a.seed}'/'ACRELGATE';run.mkdir(parents=True,exist_ok=False)
        model=GatedIntervalHead().to(device);model.base.load_state_dict(initial,strict=True)
        # Exact function-match witness: gate starts at 1 and baseline COT channel is identical.
        ids=torch.arange(min(128,n),device=device)
        plain=IntervalHead().to(device);plain.load_state_dict(initial,strict=True);plain.eval();model.eval()
        with torch.inference_mode():
            p0=plain(tr['agri'][ids],tr['geometry'][ids],tr['cot_calibrated'][ids])
            p1=model(tr['agri'][ids],tr['geometry'][ids],tr['cot_calibrated'][ids],history_tr[ids],leads_tr[ids])
        init_delta=float((p1-p0).abs().max());assert init_delta==0.0,init_delta
        del plain,p0,p1;torch.cuda.empty_cache()
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs,eta_min=1e-5)
        best=float('inf');best_state=None;curve=[];updates=0;started=time.monotonic()
        for ep,order in enumerate(orders):
            model.train();weighted=0.
            for lo in range(0,n,a.batch_size):
                ix=torch.as_tensor(order[lo:lo+a.batch_size],device=device)
                pred=model(tr['agri'][ix],tr['geometry'][ix],tr['cot_calibrated'][ix],history_tr[ix],leads_tr[ix])
                loss=(((pred*tr['clear'][ix]-tr['ghi'][ix])/1000.0)**2).mean()
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite gated GHI loss')
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0,error_if_nonfinite=True);opt.step()
                updates+=1;weighted+=float(loss.detach())*len(ix)
            sched.step()
            pv,sv=score(model,va,va['cot_calibrated'],history_va,leads_va,a.batch_size,True)
            truth=va['ghi'].detach().cpu().numpy().astype(np.float64)
            rmse=float(np.sqrt(np.mean((pv-truth)**2)))
            curve.append({'epoch':ep+1,'train_scaled_mse':weighted/n,'val_rmse_wm2':rmse,'updates':updates})
            if rmse<best: best=rmse;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if ep==0 or (ep+1)%5==0: print(json.dumps({'seed':a.seed,'station':station,**curve[-1]},separators=(',',':')),flush=True)
        model.load_state_dict(best_state,strict=True);pred,scales=score(model,va,va['cot_calibrated'],history_va,leads_va,a.batch_size,True)
        truth=va['ghi'].detach().cpu().numpy().astype(np.float64)
        keys=np.asarray([json.dumps(k,sort_keys=True) for k in va['keys']])
        pp=run/'validation_predictions.npz';np.savez_compressed(pp,prediction_ghi_wm2=pred,truth_ghi_wm2=truth,keys=keys)
        ck=run/'best.pt';torch.save({'model':best_state,'arm':'ACRELGATE','station':station,'seed':a.seed,'best_val_rmse_wm2':best,'pack_manifest_sha256':rec['pack_manifest_sha256'],'calibration_sha256':rec['center_calibration_sha256']},ck)
        m={'station':station,'seed':a.seed,'arm':'ACRELGATE','n_train':n,'n_val':len(truth),'rmse_wm2':float(np.sqrt(np.mean((pred-truth)**2))),'mae_wm2':float(np.mean(np.abs(pred-truth))),'bias_wm2':float(np.mean(pred-truth)),'best_epoch':int(np.argmin([x['val_rmse_wm2'] for x in curve])+1),'updates':updates,'elapsed_seconds':time.monotonic()-started,'initial_function_max_abs_delta':init_delta,'gate_validation_scale':scale_summary(scales,future_va),'checkpoint_sha256':sha(ck),'predictions_sha256':sha(pp),'test_payload_opened':False}
        (run/'metrics.json').write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8');(run/'training_curve.json').write_text(json.dumps(curve,indent=2)+'\n',encoding='utf-8')
        summaries.append(m);print('ARM_COMPLETE '+json.dumps(m),flush=True)
        del tr,va,model,opt,sched;torch.cuda.empty_cache()
    (a.output_root/f'seed{a.seed}_running.json').unlink()
    rec.update({'state':'COMPLETE_FUTURE_GHI_COT_ABLATION_SEED_RUN','runs':summaries})
    (a.output_root/f'seed{a.seed}_manifest.json').write_text(json.dumps(rec,indent=2)+'\n',encoding='utf-8')
    print('GHI_COT_GATE_SEED_COMPLETE '+str(a.seed),flush=True)
if __name__=='__main__': main()
