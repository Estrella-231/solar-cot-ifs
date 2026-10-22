import numpy as np,json
from screen import features
c=np.array([[0.,1.,2.],[1.,2.,3.],[0.,0.,0.],[3.,2.,1.]])
v=np.array([[1,1,1],[1,1,1],[0,0,0],[1,1,1]],bool)
c=np.tile(c,(1,3));v=np.tile(v,(1,3))
x,scale=features(c,v,np.array([15,60,120,240]),np.array([1,1,0,0],bool))
assert x.shape==(4,27) and np.all(x[2]==0)
xf=x[:2];y=np.array([.1,-.2]);beta=np.linalg.solve(xf.T@xf/2+.01*np.eye(27),xf.T@y/2)
assert float(x[2]@beta)==0 and np.isfinite(beta).all()
print('LINEAR_ZERO_INPUT_AND_NORMAL_EQUATION_WITNESS_PASS',scale)
