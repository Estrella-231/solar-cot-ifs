#!/usr/bin/env python3
"""Synthetic witnesses for the causal, identity-initialized future-COT gate."""
import torch
from train_future_ghi_input_conditioned_gate import InputConditionedGate, GatedIntervalHead
from train_loss_pair import IntervalHead

def main():
    torch.manual_seed(7)
    n=4
    ag=torch.randn(n,3,13,3,3)
    geo=torch.randn(n,3,2,3,3)
    cot=torch.randn(n,3,1,3,3)
    history=torch.tensor([1,1,0],dtype=torch.bool).view(1,3,1,1,1).repeat(n,1,1,1,1)
    leads=torch.tensor([[ -15.,0.,15.]]*n)
    gate=InputConditionedGate().eval()
    adjusted,scale=gate(ag,geo,cot,history,leads)
    assert torch.equal(scale,torch.ones_like(scale))
    assert torch.equal(adjusted,cot)
    gate.net[-1].bias.data.fill_(0.2)
    adjusted,scale=gate(ag,geo,cot,history,leads)
    assert torch.equal(adjusted[:,0],cot[:,0]) and torch.equal(adjusted[:,1],cot[:,1])
    assert not torch.equal(adjusted[:,2],cot[:,2])
    assert torch.all(scale[:,2]>1)
    torch.manual_seed(11)
    base=IntervalHead(); gated=GatedIntervalHead(); gated.base.load_state_dict(base.state_dict(),strict=True)
    with torch.no_grad():
        p0=base(ag,geo,cot)
        p1=gated(ag,geo,cot,history,leads)
    assert torch.equal(p0,p1)
    print('INPUT_CONDITIONED_GATE_SYNTHETIC_WITNESS_PASS')
if __name__=='__main__':main()
