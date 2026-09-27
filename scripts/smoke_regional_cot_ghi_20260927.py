"""Seeded kernel/causal-prefix witness, not experimental performance evidence."""
import argparse
import json
from pathlib import Path

import torch

from train_regional_cot_ghi_pilot_20260927 import RegionalStationHead


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args()
    torch.manual_seed(42); torch.cuda.manual_seed_all(42); torch.set_num_threads(2)
    model=RegionalStationHead().cuda().eval()
    local=torch.randn(2,16,64,64,device="cuda")
    region=torch.randn(2,16,32,32,device="cuda")
    lc=torch.rand(2,24,1,64,64,device="cuda",requires_grad=True)
    rc=torch.rand(2,24,1,32,32,device="cuda",requires_grad=True)
    st=torch.tensor([0,1],device="cuda"); lead=torch.tensor([2,15],device="cuda")
    z=model(local,region,lc,rc,st,lead)
    altered_lc=lc.detach().clone();altered_rc=rc.detach().clone()
    altered_lc[0,11:]+=100;altered_rc[0,11:]+=100
    with torch.no_grad():
        alt=model(local,region,altered_lc,altered_rc,st,lead)
    assert torch.equal(z.detach(),alt),"unavailable future prefix changed output"
    z.sum().backward()
    assert lc.grad[0,11:].abs().sum()==0 and rc.grad[0,11:].abs().sum()==0
    assert lc.grad[1,:8].abs().sum()>0 and rc.grad[1,:8].abs().sum()>0,"last lead cannot use history"
    assert torch.isfinite(z).all()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({"state":"PASS_REGIONAL_HEAD_CAUSAL_KERNEL_WITNESS",
        "test_used":False,"synthetic_witness":True,"seed":42,"unavailable_prefix_invariant":True,
        "last_lead_history_gradient_nonzero":True,"gpu":torch.cuda.get_device_name(0),
        "parameters":sum(p.numel() for p in model.parameters())},indent=2)+"\n")
    print("PASS_REGIONAL_HEAD_CAUSAL_KERNEL_WITNESS",flush=True)


if __name__=="__main__":main()
