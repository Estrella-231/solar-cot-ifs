#!/usr/bin/env python3
"""Train a GHI head after replacing only target COT by its 15-minute window mean."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/"experiments"/"high_kt_loss_20261016"))
from train_loss_pair import IntervalHead
from train_future_ghi_heads_cot_reliability import load_station,seed_all,score,sha

def window_average_input(data,norm):
    out=data['cot_calibrated'].clone()
    mean=torch.as_tensor(norm['COT_log1p_mean'],dtype=out.dtype,device=out.device)
    std=torch.as_tensor(norm['COT_log1p_std'],dtype=out.dtype,device=out.device)
    raw=out*std+mean
    previous=torch.expm1(raw[:,1,0]).clamp_min(0.)
    current=torch.expm1(raw[:,2,0]).clamp_min(0.)
    averaged=torch.log1p(0.5*(previous+current))
    out[:,2,0]=(averaged-mean)/std
    if not torch.isfinite(out).all():raise ValueError('nonfinite window-averaged COT')
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--pack',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch-size',type=int,default=1792);a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    m=json.loads((a.pack/'pack_manifest.json').read_text());cc=json.loads(a.calibration.read_text())
    if m.get('state')!='COMPLETE_CAUSAL_FUTURE_GHI_HEAD_PACK' or m.get('train_forecast_oof') is not True or m.get('forecast_bank_oof_fold_count')!=5:raise ValueError('five-fold OOF pack required')
    if m.get('test_payload_opened') is not False or m.get('buffer_payload_opened') is not False or m.get('future_observed_agri_used_as_input') is not False or m.get('cpp_used_as_input') is not False:raise ValueError('pack closure failed')
    if cc.get('forecast_pack_manifest_sha256')!=sha(a.pack/'pack_manifest.json') or cc.get('state')!='COMPLETE_CROSSFIT_TRAIN_ONLY_OOF_FUTURE_COT_RELIABILITY_CALIBRATION':raise ValueError('train-only COT calibration mismatch')
    if cc.get('test_payload_opened') is not False or cc.get('buffer_payload_opened') is not False:raise ValueError('calibration test/buffer closure failed')
    a.output_root.mkdir(parents=True,exist_ok=True);device=torch.device('cuda:0');torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;summaries=[]
    for st in ('sili','zhujia'):
        tr=load_station(a.pack,'train',st,device,m,cc);va=load_station(a.pack,'val',st,device,m,cc)
        trcot=window_average_input(tr,m);vacot=window_average_input(va,m);n=len(tr['ghi'])
        if n//a.batch_size<12:raise ValueError('insufficient updates per epoch')
        seed_all(a.seed);base=IntervalHead().to(device);initial={k:v.detach().clone() for k,v in base.state_dict().items()};rng=np.random.default_rng(a.seed);orders=[rng.permutation(n) for _ in range(a.epochs)]
        run=a.output_root/st/f"seed{a.seed}"/'ACRELWIN';run.mkdir(parents=True,exist_ok=False);model=IntervalHead().to(device);model.load_state_dict(initial,strict=True);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs,eta_min=1e-5);best=float('inf');beststate=None;curve=[];updates=0;started=time.monotonic()
        for ep,order in enumerate(orders):
            model.train();weighted=0.
            for lo in range(0,n,a.batch_size):
                ix=torch.as_tensor(order[lo:lo+a.batch_size],device=device);pred=model(tr['agri'][ix],tr['geometry'][ix],trcot[ix]);loss=(((pred*tr['clear'][ix]-tr['ghi'][ix])/1000.)**2).mean()
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite GHI loss')
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step();updates+=1;weighted+=float(loss.detach())*len(ix)
            sched.step();pred=score(model,{**va,'cot_calibrated':vacot},'ACREL',a.batch_size);truth=va['ghi'].cpu().numpy().astype(np.float64);r=float(np.sqrt(np.mean((pred-truth)**2)));curve.append({'epoch':ep+1,'train_scaled_mse':weighted/n,'val_rmse_wm2':r,'updates':updates})
            if r<best:best=r;beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if ep==0 or (ep+1)%5==0:print(json.dumps({'station':st,'seed':a.seed,**curve[-1]},separators=(',',':')),flush=True)
        model.load_state_dict(beststate,strict=True);preds=[]
        with torch.inference_mode():
            for lo in range(0,len(va['ghi']),a.batch_size):
                sl=slice(lo,min(lo+a.batch_size,len(va['ghi'])));preds.append((model(va['agri'][sl],va['geometry'][sl],vacot[sl])*va['clear'][sl]).cpu().numpy())
        pred=np.concatenate(preds).astype(np.float64);truth=va['ghi'].cpu().numpy().astype(np.float64);pp=run/'validation_predictions.npz';np.savez_compressed(pp,prediction_ghi_wm2=pred,truth_ghi_wm2=truth,keys=np.asarray([json.dumps(k,sort_keys=True) for k in va['keys']]))
        ck=run/'best.pt';torch.save({'model':beststate,'arm':'ACRELWIN','station':st,'seed':a.seed,'best_val_rmse_wm2':best,'pack_manifest_sha256':sha(a.pack/'pack_manifest.json'),'calibration_sha256':sha(a.calibration)},ck)
        met={'station':st,'seed':a.seed,'arm':'ACRELWIN','n_train':n,'n_val':len(truth),'best_val_rmse_wm2':float(np.sqrt(np.mean((pred-truth)**2))),'mae_wm2':float(np.mean(np.abs(pred-truth))),'bias_wm2':float(np.mean(pred-truth)),'updates':updates,'elapsed_seconds':time.monotonic()-started,'checkpoint_sha256':sha(ck),'predictions_sha256':sha(pp),'test_payload_opened':False};(run/'metrics.json').write_text(json.dumps(met,indent=2)+'\n');(run/'training_curve.json').write_text(json.dumps(curve,indent=2)+'\n');summaries.append(met);print('ARM_COMPLETE '+json.dumps(met),flush=True);del tr,va,trcot,vacot,model,opt,sched,base;torch.cuda.empty_cache()
    report={'state':'COMPLETE_FUTURE_GHI_COT_ABLATION_SEED_RUN','seed':a.seed,'arm':'ACRELWIN','stations':['sili','zhujia'],'epochs':a.epochs,'batch_size':a.batch_size,'train_forecast_oof':True,'forecast_bank_oof_fold_count':5,'split_id':m['split_id'],'transform':'target future COT slot replaced by trapezoidal mean physical COT over (T-15min,T] from previous/current ACREL frames; log1p converted and original train normalization retained','calibration_sha256':sha(a.calibration),'pack_manifest_sha256':sha(a.pack/'pack_manifest.json'),'runs':summaries,'test_payload_opened':False,'buffer_payload_opened':False,'future_observed_agri_used_as_inference_input':False,'cpp_used_as_input':False};(a.output_root/f'seed{a.seed}_manifest.json').write_text(json.dumps(report,indent=2)+'\n');print('WINDOW_COT_SEED_COMPLETE',a.seed,flush=True)
if __name__=='__main__':main()
