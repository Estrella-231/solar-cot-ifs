import torch
from train_sili_transport_thickness_pilot_20260927 import TransportThicknessNet,warp_batch,make_input

torch.manual_seed(42);torch.set_num_threads(2);device='cuda' if torch.cuda.is_available() else 'cpu'
h=torch.rand(2,8,1,64,64,device=device);f=torch.rand(2,16,1,64,64,device=device);motion=torch.tensor([[0.3,-0.8],[0.0,0.0]],device=device)
adv,support=warp_batch(h[:,-1],motion);x=make_input(h,f,adv,motion);m=TransportThicknessNet().to(device)
p=m(x,adv)
assert x.shape==(2,8,16,64,64) and p.shape==(2,1,16,64,64) and support.shape==(2,1,16,64,64)
loss=p.mean();loss.backward()
assert torch.isfinite(p).all() and torch.isfinite(m.out.weight.grad).all()
assert torch.equal(p.detach(),adv.detach()),'zero-init residual must start at pure transport'
print('PASS_SILI_TRANSPORT_THICKNESS_SMOKE',flush=True)
