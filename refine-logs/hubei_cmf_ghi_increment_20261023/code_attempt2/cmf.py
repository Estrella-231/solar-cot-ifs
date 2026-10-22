"""Published fixed CMF parameterization, AMT17:1851(2024), Eq2; z in degrees."""
import numpy as np

def attenuation(cot,sza):
    cot=np.asarray(cot,dtype=np.float64);z=np.asarray(sza,dtype=np.float64)
    assert np.isfinite(cot).all() and np.isfinite(z).all()
    assert np.all((cot>=0)&(cot<=100)) and np.all((z>=0)&(z<=80.0001))
    a=.224+.000281*z-.0000218*z*z+.000000371*z**3-.00000000265*z**4
    b=12.2+.00527*z-.00224*z*z+.00000833*z**3+.0000000394*z**4
    assert np.all(a>0) and np.all(b>0)
    phi=np.tanh(np.power(cot,a))**b
    assert np.isfinite(phi).all() and np.all((phi>=0)&(phi<=1))
    return phi

if __name__=='__main__':
    tau=np.linspace(0,100,1001)[None,:];z=np.arange(0,81)[:,None]
    q=attenuation(tau,z)
    assert np.all(q[:,0]==0) and np.all(np.diff(q,axis=1)>=-1e-14)
    assert np.all(q[:,-1]>q[:,10])
    print('PASS_PUBLISHED_CMF_DOMAIN_ZERO_AND_MONOTONICITY_NOT_GHI_RESULT')
