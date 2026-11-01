"""Shared frame head + scan-supported interval readout; no spatial averaging."""
import torch
from torch import nn

class IntervalGHIHead(nn.Module):
    def __init__(self,frame_head,mode='interval'):
        super().__init__()
        if mode not in ('last','interval'):raise ValueError(mode)
        self.frame_head=frame_head;self.mode=mode

    def forward(self,agri,geometry,cot,weights,scan_clear):
        if agri.ndim!=5 or agri.shape[1]!=3:raise ValueError('expected [B,3,C,H,W]')
        b,t=agri.shape[:2]
        if weights.shape!=(b,t) or scan_clear.shape!=(b,t):raise ValueError('weights/clear shape mismatch')
        if (weights<0).any() or not torch.allclose(weights.sum(1),torch.ones(b,device=weights.device),atol=1e-5):raise ValueError('invalid temporal integral weights')
        raw=self.frame_head(agri.flatten(0,1),geometry.flatten(0,1),None if cot is None else cot.flatten(0,1)).reshape(b,t)
        if self.mode=='last':return raw[:,-1]
        effective=weights*scan_clear
        if not torch.isfinite(effective).all() or (effective<0).any() or (effective.sum(1)<=0).any():raise ValueError('invalid clear-sky weighted support')
        effective=effective/effective.sum(1,keepdim=True)
        return (raw*effective).sum(1) # multiply by the unchanged label-window clear GHI outside this module

def witness(head_source):
    import importlib.util
    sp=importlib.util.spec_from_file_location('frozen_heads',head_source);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
    torch.set_num_threads(2);torch.manual_seed(42)
    a=torch.randn(4,3,13,3,3,requires_grad=True);g=torch.randn(4,3,2,3,3);c=torch.randn(4,3,1,3,3)
    w=torch.tensor([[.02,.646,.334]]).repeat(4,1);clear=torch.tensor([[400.,500.,600.]]).repeat(4,1)
    base=m.LocalGHIHead('AC',2).eval();last=IntervalGHIHead(base,'last');inter=IntervalGHIHead(base,'interval')
    direct=base(a[:,-1],g[:,-1],c[:,-1]);err=float((last(a,g,c,w,clear)-direct).abs().max())
    pred=inter(a,g,c,w,clear);pred.sum().backward();grad=a.grad.abs().sum((0,2,3,4)).tolist()
    assert err<1e-6 and all(v>0 for v in grad)
    assert sum(p.numel() for p in inter.parameters())==24489
    class Constant(nn.Module):
        def forward(self,a,g,c):return torch.ones(len(a),device=a.device)*.7
    kt=IntervalGHIHead(Constant())(a.detach(),g,c,w,clear);conserve=float((kt-.7).abs().max());assert conserve<1e-6
    return dict(state='PASS_CPU_ALGEBRA_SHAPE_AND_GRADIENT_WITNESS',parameter_count=24489,last_mode_max_abs_error=err,constant_kt_conservation_error=conserve,agri_gradient_by_frame=grad,scientific_result=False,training_updates=0,test_payloads_read=0)

if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('--head-source',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=witness(a.head_source);open(a.output,'w').write(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
