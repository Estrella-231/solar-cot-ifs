import sys
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).parent))
from run_direct_cot_monocal_hcv import Head, MonotoneLogCot
from score_direct_cot_monocal_hcv import delta

def test_identity_zero_anchor_monotonic_and_gradient():
    m=MonotoneLogCot(mu=1.2,sd=.8,trainable=True)
    x=torch.linspace(0,6,501); y=m.transform_log(x)
    assert abs(float(m.transform_log(torch.tensor([0.]))[0]))<1e-7
    assert torch.all(y[1:]>=y[:-1])
    u=torch.rand(3,1,3,3)*6; z=(u-1.2)/.8; sun=torch.ones_like(z,dtype=torch.bool)
    assert torch.allclose(m(z,sun),z,atol=2e-6,rtol=2e-6)
    loss=m(z,sun).square().mean(); loss.backward()
    assert m.raw_slopes.grad is not None and torch.isfinite(m.raw_slopes.grad).all()

def test_mask_and_no_cot_path():
    m=MonotoneLogCot(mu=.7,sd=.9,trainable=True)
    z=torch.ones(2,1,3,3); sun=torch.zeros_like(z,dtype=torch.bool)
    assert torch.count_nonzero(m(z,sun))==0
    h=Head(.7,.9,train_cal=False,use_cot=False)
    out=h(torch.zeros(2,13,3,3),torch.zeros(2,2,3,3),torch.zeros(2,4,3,3),torch.zeros(2,8),sun.float())
    assert out.shape==(2,) and torch.isfinite(out).all()

def test_paired_delta_sign_is_candidate_minus_reference():
    agg={(s,seed,b):[[4.0,1],[9.0,1]] for s in ('sili','zhujia') for seed in (42,43,44) for b in ('d1','d2')}
    assert delta(agg,['d1','d2'],[1.,1.])==-1.0
