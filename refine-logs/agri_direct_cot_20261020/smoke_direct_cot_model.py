import torch
import train_future_cot_u as m
torch.set_num_threads(1)
net=m.DirectCotForecaster()
x=torch.zeros(1,8,31,16,16)
g=torch.zeros(1,16,3,16,16)
y=net(x,g)
assert y.shape==(1,1,16,16,16) and torch.isfinite(y).all() and (y>=0).all()
loss=m.safe_huber(y,torch.ones_like(y),torch.ones_like(y,dtype=torch.bool))
loss.backward()
assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
print('PASS_CPU_MODEL_FORWARD_BACKWARD',tuple(y.shape),float(loss),float(y.min()),float(y.max()))
