"""Independent movement/feature deletion and masked physical-loss checks."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import map_coordinates
from trajectory_model import TrajectorySource,loss_parts
torch.set_num_threads(1);torch.manual_seed(42)
m=TrajectorySource().eval();h=w=32
yy,xx=np.mgrid[:h,:w]
history=np.stack([(yy+2*xx+i)/100 for i in range(8)])[None].astype('float32')
valid=np.ones_like(history,dtype=bool)
args=[torch.from_numpy(history),torch.from_numpy(valid),torch.tensor([[.2,-.3]]),torch.tensor([60]),torch.zeros(1,3,h,w)]
with torch.no_grad():
    feature,base,good=m.transport_features(*args,arm='full')
    actual=m(*args,arm='full');repeat=m(*args,arm='repeat_last')
    assert torch.equal(actual,base) and torch.equal(repeat,base)
    expected=[]
    for i in range(8):
        age=4+7-i;coords=np.array([yy-age*.2,xx+age*.3])
        expected.append(map_coordinates(history[0,i],coords,order=1,mode='constant',cval=0))
    expected=np.stack(expected)
    comparison=np.max(abs(feature[0,:8].numpy()*np.log(101)-expected)[good[0].numpy()])
    assert comparison<2e-6,comparison
    changed=[v.clone() for v in args];changed[0][:,:7]+=1
    full2,base2,_=m.transport_features(*changed,arm='full')
    repeated2,_,_=m.transport_features(*changed,arm='repeat_last')
    repeated1,_,_=m.transport_features(*args,arm='repeat_last')
    assert not torch.equal(full2,feature) and torch.equal(repeated1,repeated2) and torch.equal(base,base2)
pred=torch.tensor([[[[np.log1p(12.),np.log1p(5.)]]]],dtype=torch.float32,requires_grad=True)
target=torch.tensor([[[[10.,float('nan')]]]])
mask=torch.tensor([[[[True,False]]]])
num,count=loss_parts(pred,target,mask)
assert abs(float(num)-.0004)<1e-8 and count==1
num.backward();assert torch.isfinite(pred.grad).all() and pred.grad[0,0,0,1]==0
result=dict(state='PASS_CPU_TRANSPORT_FEATURE_DELETION_AND_MASKED_PHYSICAL_LOSS',
    parameters=sum(p.numel() for p in m.parameters()),baseline_step0_difference=0,
    scipy_transport_max_log_difference=float(comparison),old_history_changes_repeat_last=False,
    masked_nan_gradient_finite=True,GPU_profile_passed=False,training_started=False)
Path(__file__).with_name('CPU_WITNESS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
