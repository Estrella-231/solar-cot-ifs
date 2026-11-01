"""CPU-only gradient witness with the current frozen monthly station16 R."""
import argparse
import json
import sys
from pathlib import Path

import torch

from cot_auxiliary import frozen_r_cot_loss, mixed_cot_target, station_r_inputs


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--r-source",type=Path,required=True)
    p.add_argument("--r-checkpoint",type=Path,required=True)
    p.add_argument("--r-norm",type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.r_source))
    from train_station16_R import COTUNet
    norm=json.loads(a.r_norm.read_text(encoding="utf-8"))
    payload=torch.load(a.r_checkpoint,map_location="cpu",weights_only=False)
    r=COTUNet().float().eval().requires_grad_(False);r.load_state_dict(payload["model"],strict=True)
    pred=torch.randn(1,1,13,256,256,requires_grad=True)
    geo=torch.zeros(1,1,2,3,16,16);geo[:,:,0,0]=0.8;geo[:,:,1,0]=0.8;geo[:,:,:,2]=1
    centers=torch.tensor([[[[163,148],[73,150]]]])
    agri_mean=torch.zeros(13);agri_std=torch.ones(13)
    rmean=torch.tensor(norm["mean"],dtype=torch.float32);rstd=torch.tensor(norm["std"],dtype=torch.float32)
    rin=station_r_inputs(pred,geo,centers,agri_mean,agri_std,rmean,rstd)
    cpp=torch.full((1,1,2,1,16,16),5.0);cmask=torch.ones_like(cpp,dtype=torch.bool)
    rtarget=torch.log1p(torch.full_like(cpp,4.0));valid=torch.ones_like(cmask)
    target,mask,_,_=mixed_cot_target(cpp,cmask,rtarget,valid)
    loss,out=frozen_r_cot_loss(r,rin,target,mask)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all() and pred.grad.abs().sum()>0
    assert all(p.grad is None for p in r.parameters())
    print("REAL_FROZEN_R_CPU_GRADIENT_PASS",{"loss":float(loss),"prediction_grad_l1":float(pred.grad.abs().sum()),"R_parameter_grads":0})


if __name__=="__main__":main()
