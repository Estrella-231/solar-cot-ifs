import torch
from train_sili_transport_thickness_pilot_20260927 import TransportThicknessNet,warp_batch,make_input,training_loss

torch.manual_seed(42);torch.set_num_threads(2);device='cuda' if torch.cuda.is_available() else 'cpu'
h=torch.rand(2,8,1,64,64,device=device);f=torch.rand(2,16,1,64,64,device=device);motion=torch.tensor([[0.3,-0.8],[0.0,0.0]],device=device)
adv,support=warp_batch(h[:,-1],motion);x=make_input(h,f,adv,motion);m=TransportThicknessNet().to(device)
p=m(x,adv)
assert x.shape==(2,8,16,64,64) and p.shape==(2,1,16,64,64) and support.shape==(2,1,16,64,64)
target=torch.full((2,16,64,64),2.0,device=device)
mask=torch.ones_like(target,dtype=torch.bool)
loss=training_loss(p,target,mask);loss.backward()
assert torch.isfinite(p).all() and torch.isfinite(m.out.weight.grad).all()
assert m.out.weight.grad.abs().sum()>0
assert torch.equal(p.detach(),adv.detach()),'zero-init residual must start at pure transport'
# All future leads must contribute, while invalid reference pixels must not.
later=target.clone();later[:,1:]=3.0
assert not torch.isclose(training_loss(p.detach(),later,mask),loss.detach())
mask[:,1:]=False
assert torch.equal(training_loss(p.detach(),later,mask),training_loss(p.detach(),target,mask))
try:
    training_loss(p.detach(),target[:,0],mask[:,0])
except ValueError:
    pass
else:
    raise AssertionError('dropped target time axis must be rejected')
print('PASS_SILI_TRANSPORT_THICKNESS_SMOKE',flush=True)
