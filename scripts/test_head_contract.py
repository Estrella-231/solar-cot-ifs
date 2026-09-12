"""Independent CPU objective/ablation witnesses using unequal group counts."""
import numpy as np
import torch
from train_head_pilot import Head

torch.set_num_threads(1);torch.manual_seed(42)
a=Head(False).eval();ac=Head(True).eval();ac.load_state_dict(a.state_dict())
x=torch.randn(6,13,16,16);c=torch.randn(6,4);g=torch.randn(6,3)
station=torch.tensor([0,1,0,1,0,1]);lead=torch.tensor([0,0,3,5,11,15])
with torch.no_grad():
    assert torch.equal(a(x,c,g,station,lead),a(x,c*100,g,station,lead))
    assert torch.equal(a(x,c,g,station,lead),ac(x,torch.zeros_like(c),g,station,lead))
    assert not torch.equal(ac(x,c,g,station,lead),ac(x,c*100,g,station,lead))
    assert (ac(x,c,g,station,lead)>=0).all()
# Every group represented, with unequal sizes. Check row weights vs direct macro MSE.
groups=np.repeat(np.arange(32),np.arange(1,33));errors=np.linspace(-2,3,len(groups))
counts=np.bincount(groups,minlength=32);w=len(groups)/(32*counts[groups])
actual=np.mean(errors**2*w)
expected=np.mean([np.mean(errors[groups==j]**2) for j in range(32)])
assert np.isclose(actual,expected,atol=1e-14)
assert np.isfinite(np.float32(477.55)**2)
print('PASS: A excludes C, common zero-slot equivalence, AC uses C, nonnegative output, equal32-group objective')
