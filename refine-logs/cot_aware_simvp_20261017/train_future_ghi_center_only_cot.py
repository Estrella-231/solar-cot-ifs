#!/usr/bin/env python3
"""Train one seed of the future-COT center-only GHI diagnostic."""
from __future__ import annotations
import argparse,csv,json,random,sys,time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/"experiments"/"high_kt_loss_20261016"))
from train_loss_pair import IntervalHead
from train_future_ghi_heads_cot_reliability import load_station,seed_all,score,sha

def center_only(data,split):
    out=data["cot_calibrated"].clone(); offsets=(-30,-15,0)
    for pos,offset in enumerate(offsets):
        for i,key in enumerate(data["keys"]):
            if int(key["lead_minutes"])+offset<=0: continue
            center=out[i,pos,0,1,1].clone(); out[i,pos,0]=center
    if not torch.equal(out[:,:,:,1,1],data["cot_calibrated"][:,:,:,1,1]): raise ValueError("future center changed")
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument("--pack",type=Path,required=True);p.add_argument("--center-calibration",type=Path,required=True);p.add_argument("--output-root",type=Path,required=True);p.add_argument("--seed",type=int,required=True);p.add_argument("--epochs",type=int,default=30);p.add_argument("--batch-size",type=int,default=1792);a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    manifest=json.loads((a.pack/"pack_manifest.json").read_text());cal=json.loads(a.center_calibration.read_text())
    if manifest.get("state")!="COMPLETE_CAUSAL_FUTURE_GHI_HEAD_PACK" or manifest.get("train_forecast_oof") is not True:raise ValueError("five-fold OOF pack required")
    if manifest.get("test_payload_opened") is not False or manifest.get("buffer_payload_opened") is not False:raise ValueError("test/buffer not certified closed")
    if cal.get("forecast_pack_manifest_sha256")!=sha(a.pack/"pack_manifest.json"):raise ValueError("calibration source mismatch")
    a.output_root.mkdir(parents=True,exist_ok=True);device=torch.device("cuda:0");torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;summaries=[]
    for station in ("sili","zhujia"):
        tr=load_station(a.pack,"train",station,device,manifest,cal);va=load_station(a.pack,"val",station,device,manifest,cal)
        trcot=center_only(tr,"train");vacot=center_only(va,"val");n=len(tr["ghi"])
        if n//a.batch_size<12:raise ValueError("batch size leaves fewer than 12 updates/epoch")
        seed_all(a.seed);base=IntervalHead().to(device);initial={k:v.detach().clone() for k,v in base.state_dict().items()};rng=np.random.default_rng(a.seed);orders=[rng.permutation(n) for _ in range(a.epochs)]
        run=a.output_root/station/f"seed{a.seed}"/"ACRELCENTER";run.mkdir(parents=True,exist_ok=False)
        model=IntervalHead().to(device);model.load_state_dict(initial,strict=True);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs,eta_min=1e-5);best=float("inf");beststate=None;hist=[];updates=0;started=time.monotonic()
        for epoch,order in enumerate(orders):
            model.train();weighted=0.0
            for lo in range(0,n,a.batch_size):
                ix=torch.as_tensor(order[lo:lo+a.batch_size],device=device);pred=model(tr["agri"][ix],tr["geometry"][ix],trcot[ix]);loss=(((pred*tr["clear"][ix]-tr["ghi"][ix])/1000.0)**2).mean()
                if not torch.isfinite(loss):raise FloatingPointError("nonfinite GHI loss")
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0,error_if_nonfinite=True);opt.step();updates+=1;weighted+=float(loss.detach())*len(ix)
            sched.step();pred=score(model,{**va,"cot_calibrated":vacot},"ACREL",a.batch_size);truth=va["ghi"].cpu().numpy().astype(np.float64);r=float(np.sqrt(np.mean((pred-truth)**2)));hist.append({"epoch":epoch+1,"train_scaled_mse":weighted/n,"val_rmse_wm2":r,"updates":updates})
            if r<best:best=r;beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if epoch==0 or (epoch+1)%5==0:print(json.dumps({"station":station,"seed":a.seed,**hist[-1]},separators=(",",":")),flush=True)
        model.load_state_dict(beststate,strict=True);model.eval();preds=[]
        with torch.inference_mode():
            for lo in range(0,len(va["ghi"]),a.batch_size):
                sl=slice(lo,min(lo+a.batch_size,len(va["ghi"])));preds.append((model(va["agri"][sl],va["geometry"][sl],vacot[sl])*va["clear"][sl]).cpu().numpy())
        pred=np.concatenate(preds).astype(np.float64);truth=va["ghi"].cpu().numpy().astype(np.float64);pp=run/"validation_predictions.npz";np.savez_compressed(pp,prediction_ghi_wm2=pred,truth_ghi_wm2=truth,keys=np.asarray([json.dumps(x,sort_keys=True) for x in va["keys"]]))
        ck=run/"best.pt";torch.save({"model":beststate,"arm":"ACRELCENTER","station":station,"seed":a.seed,"best_val_rmse_wm2":best,"pack_manifest_sha256":sha(a.pack/"pack_manifest.json"),"center_calibration_sha256":sha(a.center_calibration)},ck)
        met={"station":station,"seed":a.seed,"arm":"ACRELCENTER","n_train":n,"n_val":len(truth),"best_val_rmse_wm2":float(np.sqrt(np.mean((pred-truth)**2))),"mae_wm2":float(np.mean(np.abs(pred-truth))),"bias_wm2":float(np.mean(pred-truth)),"updates":updates,"elapsed_seconds":time.monotonic()-started,"checkpoint_sha256":sha(ck),"predictions_sha256":sha(pp),"test_payload_opened":False};(run/"metrics.json").write_text(json.dumps(met,indent=2)+"\n");(run/"training_curve.json").write_text(json.dumps(hist,indent=2)+"\n");summaries.append(met);print("ARM_COMPLETE "+json.dumps(met),flush=True);del tr,va,trcot,vacot,model,opt,sched,base;torch.cuda.empty_cache()
    report={"state":"COMPLETE_FUTURE_GHI_COT_ABLATION_SEED_RUN","seed":a.seed,"arm":"ACRELCENTER","stations":["sili","zhujia"],"epochs":a.epochs,"batch_size":a.batch_size,"train_forecast_oof":True,"forecast_bank_oof_fold_count":5,"split_id":manifest["split_id"],"transform":"future ACREL-calibrated COT frames only: replace all 3x3 values with the existing center value; retain historical COT and center exactly","center_calibration_sha256":sha(a.center_calibration),"pack_manifest_sha256":sha(a.pack/"pack_manifest.json"),"runs":summaries,"test_payload_opened":False,"buffer_payload_opened":False,"future_observed_agri_used_as_inference_input":False,"cpp_used_as_input":False};(a.output_root/f"seed{a.seed}_manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    print("CENTER_ONLY_COT_SEED_COMPLETE",a.seed,flush=True)
if __name__=="__main__":main()
