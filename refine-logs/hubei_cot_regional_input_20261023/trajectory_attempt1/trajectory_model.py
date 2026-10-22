"""Explicit per-history transport, followed by a bounded cloud-source term."""
import torch
from torch import nn
from torch.nn import functional as F

class TrajectorySource(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(20,32,3,padding=1),nn.GroupNorm(4,32),nn.SiLU(),
             nn.Conv2d(32,32,3,padding=1),nn.GroupNorm(4,32),nn.SiLU(),
             nn.Conv2d(32,16,3,padding=1),nn.GroupNorm(4,16),nn.SiLU())
        self.output=nn.Conv2d(16,1,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def transport_features(self,history,valid,motion,lead,geometry,arm):
        assert history.ndim==valid.ndim==4 and history.shape[1]==8 and history.shape==valid.shape
        b,_,h,w=history.shape;assert geometry.shape==(b,3,h,w)
        assert motion.shape==(b,2) and lead.shape==(b,) and arm in ('full','repeat_last')
        assert torch.all((lead>=15)&(lead<=240)&(lead.remainder(15)==0))
        safe=torch.where(valid.bool(),history,torch.zeros_like(history)).float()
        source_valid=valid.float()
        ages=lead.float()[:,None]/15+torch.arange(7,-1,-1,device=history.device,dtype=torch.float32)[None]
        if arm=='repeat_last':
            safe=safe[:,-1:].expand(-1,8,-1,-1);source_valid=source_valid[:,-1:].expand(-1,8,-1,-1)
            ages=(lead.float()[:,None]/15).expand(-1,8)
        yy,xx=torch.meshgrid(torch.arange(h,device=history.device,dtype=torch.float32),
                            torch.arange(w,device=history.device,dtype=torch.float32),indexing='ij')
        sy=yy[None,None]-ages[:,:,None,None]*motion[:,None,0,None,None]
        sx=xx[None,None]-ages[:,:,None,None]*motion[:,None,1,None,None]
        grid=torch.stack([sx*(2/(w-1))-1,sy*(2/(h-1))-1],-1).reshape(b*8,h,w,2)
        with torch.autocast(device_type=history.device.type,enabled=False):
            moved=F.grid_sample(safe.reshape(b*8,1,h,w),grid,mode='bilinear',padding_mode='zeros',align_corners=True).reshape(b,8,h,w)
            weights=F.grid_sample(source_valid.reshape(b*8,1,h,w),grid,mode='bilinear',padding_mode='zeros',align_corners=True).reshape(b,8,h,w)
        inside=(sy>=0)&(sy<=h-1)&(sx>=0)&(sx<=w-1)
        good=(weights>1-1e-6)&inside
        moved=torch.where(good,moved,torch.zeros_like(moved))
        last=torch.where(valid[:,-1:].bool(),history[:,-1:].float(),torch.zeros_like(history[:,-1:]).float())
        base=torch.where(good[:,-1:],moved[:,-1:],last)
        feature=torch.cat([moved/torch.log(torch.tensor(101.,device=history.device)),good.float(),
                           geometry.float(),(lead.float()/240)[:,None,None,None].expand(-1,1,h,w)],1)
        assert feature.shape==(b,20,h,w)
        return feature,base,good

    def forward(self,history,valid,motion,lead,geometry,arm='full'):
        feature,base,_=self.transport_features(history,valid,motion,lead,geometry,arm)
        residual=self.output(self.encoder(feature)).float()
        return (base+2*torch.tanh(residual)).clamp_min(0)

def loss_parts(pred_log,target_cot,valid):
    assert pred_log.shape==target_cot.shape==valid.shape
    pred=torch.expm1(pred_log.float())
    error=torch.where(valid.bool(),pred-target_cot.float(),torch.zeros_like(pred))
    return (error/100).square().sum(),valid.sum(dtype=torch.float64)

def ddp_loss(numerator,global_count,world):
    assert global_count>0
    return numerator*world/global_count
