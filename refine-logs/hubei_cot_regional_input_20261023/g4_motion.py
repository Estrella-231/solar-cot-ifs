"""Inherited full-region historical C13 phase motion; no validation tuning."""
import numpy as np
from scipy.ndimage import map_coordinates
def phase(a,b):
    a=np.asarray(a,np.float64);b=np.asarray(b,np.float64);assert a.shape==b.shape==(256,256)
    win=np.outer(np.hanning(256),np.hanning(256));a=(a-a.mean())*win;b=(b-b.mean())*win
    f=np.fft.fft2(b)*np.conj(np.fft.fft2(a));f/=np.maximum(abs(f),1e-12)
    c=np.fft.ifft2(f).real;y,x=np.unravel_index(c.argmax(),c.shape)
    def sub(v,m,p):
        d=m-2*v+p
        return .5*(m-p)/d if abs(d)>1e-12 else 0.
    dy=y+sub(c[y,x],c[(y-1)%256,x],c[(y+1)%256,x])
    dx=x+sub(c[y,x],c[y,(x-1)%256],c[y,(x+1)%256])
    return np.array([dy-256 if dy>128 else dy,dx-256 if dx>128 else dx])
def advect(log,valid,coords):
    value=map_coordinates(log,coords,order=1,mode='constant',cval=0.)
    good=map_coordinates(valid.astype(float),coords,order=1,mode='constant',cval=0.)>1-1e-6
    return value,good
def witness():
    x=np.zeros((256,256));x[111,124]=1
    motion=phase(x,np.roll(x,(2,-3),(0,1)))
    assert np.max(abs(motion-[2,-3]))<.02,motion
    q,valid=advect(x,np.ones_like(x,dtype=bool),np.array([[113-2.],[121+3.]]))
    assert valid.all() and q[0]==1
    return dict(state='PASS_PHASE_AND_BACKTRACE_SIGN',shift=motion.tolist(),interpolation='bilinear log1p COT; all source weights valid')
