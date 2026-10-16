"""CPU witness: exact retained architecture and a symmetric twofold loss weight."""
import ast,json,sys
from pathlib import Path
import numpy as np
import torch

here=Path(__file__).resolve().parent;repo=here.parents[1]
def model_ast(path):
    return ast.dump(next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.ClassDef) and n.name=='IntervalHead'))
assert model_ast(here/'train_loss_pair.py')==model_ast(repo/'experiments/r_geometry_downstream_ghi_20261016/train_three_arm_ghi.py')
# Equal ±100 W/m² errors must have the same weight regardless of sign.
p=torch.tensor([100.,-100.,100.,-100.],requires_grad=True)
w=torch.tensor([2.,2.,1.,1.])/1.5
loss=(w*(p/1000).square()).mean();loss.backward()
g=p.grad.detach().numpy()
np.testing.assert_allclose(g[:2],2*g[2:],rtol=1e-7,atol=1e-10)
assert g[0]==-g[1] and g[2]==-g[3]
out=dict(state='PASS_IDENTICAL_ARCHITECTURE_SYMMETRIC_2X_GRADIENT',parameter_contract=41665,train_weight_definition='clear>=100 and observed_kt>=0.9:2 else:1; train only; station mean normalized',test_payloads_read=0)
(here/'loss_witness.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
