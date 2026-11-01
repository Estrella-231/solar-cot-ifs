#!/usr/bin/env python3
"""Synthetic invariant test for the future-only center-preserving transform."""
import torch
from train_future_ghi_spatial_contrast import spatial_input
from train_future_ghi_center_only_cot import center_only
from train_future_ghi_temporal_increment_cot import temporal_input
from train_future_ghi_window_average_cot import window_average_input

def main():
    x=torch.arange(27,dtype=torch.float32).reshape(1,3,1,3,3)
    data={"cot_calibrated":x.clone(),"history_cot_mask":torch.tensor([[[[[1.]]],[[[0.]]],[[[0.]]]]]),
          "keys":[{"lead_minutes":"30","sequence_index_within_split":"7"}]}
    cal={"fold_by_sequence_index":{"7":2},"parameters_excluding_each_fold":{},"parameters_full_train":{"sili":{"15":{"gamma":0.25},"30":{"gamma":0.5}}}}
    for fold in range(5): cal["parameters_excluding_each_fold"][str(fold)]={"sili":{"15":{"gamma":0.25},"30":{"gamma":0.5}}}
    y=spatial_input(data,"train","sili",cal)
    assert torch.equal(y[:,:, :,1,1],x[:,:,:,1,1])
    assert torch.equal(y[:,0],x[:,0])
    assert torch.allclose(y[0,1,0],x[0,1,0,1,1]+0.25*(x[0,1,0]-x[0,1,0,1,1]))
    assert torch.allclose(y[0,2,0],x[0,2,0,1,1]+0.5*(x[0,2,0]-x[0,2,0,1,1]))
    z=center_only(data,"train")
    assert torch.equal(z[:,:, :,1,1],x[:,:,:,1,1])
    assert torch.equal(z[:,0],x[:,0])
    assert torch.all(z[0,1,0]==x[0,1,0,1,1]) and torch.all(z[0,2,0]==x[0,2,0,1,1])
    tc={"fold_by_sequence_index":{"7":2},"parameters_excluding_each_fold":{},"parameters_full_train":{"sili":{"30":{"alpha":0.25}}}}
    for fold in range(5):tc["parameters_excluding_each_fold"][str(fold)]={"sili":{"30":{"alpha":0.25}}}
    q=temporal_input(data,"train","sili",tc)
    assert torch.equal(q[:,0],x[:,0]) and torch.equal(q[:,1],x[:,1])
    assert torch.allclose(q[0,2,0],x[0,1,0]+0.25*(x[0,2,0]-x[0,1,0]))
    logprev=torch.log(torch.tensor(2.));logcurr=torch.log(torch.tensor(5.));mu=.5;sd=1.5
    winin=torch.zeros((1,3,1,3,3),dtype=torch.float32);winin[:,1]=((logprev-mu)/sd);winin[:,2]=((logcurr-mu)/sd)
    w=window_average_input({"cot_calibrated":winin},{"COT_log1p_mean":mu,"COT_log1p_std":sd})
    expected=(torch.log1p(torch.tensor((1.+4.)/2.))-mu)/sd
    assert torch.equal(w[:,0],winin[:,0]) and torch.equal(w[:,1],winin[:,1])
    assert torch.allclose(w[:,2],torch.ones_like(w[:,2])*expected)
    print("SPATIAL_COT_SYNTHETIC_INVARIANTS_PASS")

if __name__=="__main__":main()
