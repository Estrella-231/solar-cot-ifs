"""Check capacity, full initial states and zero-COT pairing on CPU."""
import hashlib,json
import torch
from train_paired_forecast import HubeiLocalGHIHead
torch.set_num_threads(1)
torch.manual_seed(42);a=HubeiLocalGHIHead('FA')
torch.manual_seed(42);b=HubeiLocalGHIHead('FAC')
assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())==24489
assert all(torch.equal(x,y) for x,y in zip(a.state_dict().values(),b.state_dict().values()))
x=torch.randn(8,15,3,3);z=torch.zeros(8,1,3,3)
assert torch.equal(a(x),b(x,z))
assert torch.equal(a(x,z),a(x,torch.ones_like(z)*5))
out=b(x,torch.ones_like(z));out.sum().backward()
assert b.aux_encoder[0].weight.grad.abs().sum()>0
print(json.dumps({'state':'PAIRED_FA_FAC_CPU_WITNESS_PASS','parameters_each':24489,'same_initial_weights':True,'same_output_when_COT_zero':True,'FA_ignores_COT':True,'FAC_COT_gradient_nonzero':True}))
