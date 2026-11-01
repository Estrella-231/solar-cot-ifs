"""Feature-formula and train-only normalization regression for dual-path COT H-CV."""
from __future__ import annotations
import numpy as np
from run_direct_cot_dualpath_hcv import fit_normalize

def main():
    rng=np.random.default_rng(21); n=200
    src=rng.uniform(0,20,(n,2,3,3)).astype(np.float32)
    geo=np.zeros((n,2,3,3),dtype=np.float32); geo[:,0]=0.8; geo[:,1]=0.0
    raw={'agri':rng.normal(size=(n,13,3,3)).astype(np.float32),'geometry':geo,
         'ghi':np.full(n,500,dtype=np.float32),'clear':np.ones(n,dtype=np.float32),
         'lead':np.full(n,15,dtype=np.int64)}
    train=np.arange(150)
    arr,norm=fit_normalize(raw,src,train)
    p=arr['dual'][:,0]; s=arr['dual'][:,1]
    assert np.allclose(arr['dual'][:,2],0.5*(p+s))
    assert np.allclose(arr['dual'][:,3],np.abs(p-s))
    assert np.allclose(arr['pxc'][:,0],p) and np.allclose(arr['simvp_r'][:,0],s)
    assert np.all(arr['none']==0)
    # Perturb held-out source values: training-fit normalizers must not change.
    src2=src.copy(); src2[150:]*=100
    _,norm2=fit_normalize(raw,src2,train)
    for k in ('cot_log1p_mean_pxc','cot_log1p_std_pxc','cot_log1p_mean_simvp_r','cot_log1p_std_simvp_r'):
        assert norm[k]==norm2[k]
    assert norm['fit_days_only'] is True
    print('DIRECT_COT_DUALPATH_FEATURE_TESTS_PASS')

if __name__=='__main__': main()
