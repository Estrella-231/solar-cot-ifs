"""QC boundary, original-row identity and affine refit CPU witnesses."""
from datetime import datetime
import numpy as np
from prepare_head_qc_view import affected,moments
intervals={'sili':[(datetime.fromisoformat('2024-07-01T05:50:00+08:00'),datetime.fromisoformat('2024-07-01T08:45:00+08:00'))]}
def row(station,stamps):return dict(station=station,label_record_times_bjt='|'.join('2024-07-01T'+s+':00+08:00' for s in stamps))
assert affected(row('sili',['05:40','05:45','05:50']),intervals)
assert affected(row('sili',['08:40','08:45','08:50']),intervals)
assert not affected(row('sili',['08:50','08:55','09:00']),intervals)
assert not affected(row('zhujia',['05:40','05:45','05:50']),intervals)
physical=np.array([[3.,4.],[7.,8.],[100.,200.],[1000.,2000.]])
oldmean=physical[:3].mean(0);oldstd=physical[:3].std(0)
base=(physical-oldmean)/oldstd
stats=moments(base,np.array([0,1]),False)
view=(base-np.array(stats['mean']))/np.array(stats['std'])
direct=(physical-physical[:2].mean(0))/physical[:2].std(0)
assert np.allclose(view,direct,atol=1e-12)
assert np.allclose(view[:2].mean(0),0,atol=1e-12)
print('PASS QC boundary/same-station/source-row keys and retained-train affine refit')
