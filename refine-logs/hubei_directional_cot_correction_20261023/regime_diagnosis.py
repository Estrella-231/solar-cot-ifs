"""Complete F_select date and reference-thickness breakdown, no threshold tuning."""
import csv,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;r=list(csv.DictReader((P/'correction_samples.csv').open()));lookup={}
for q in r:lookup.setdefault((q['station'],q['field_index']),{})[q['method']]=q
records=list(lookup.values());y=np.array([float(q['raw_forecast']['target']) for q in records]);raw=np.array([float(q['raw_forecast']['prediction']) for q in records]);cor=np.array([[float(q['seed'+str(s)]['prediction']) for q in records] for s in [42,43,44]])
st=np.array([q['raw_forecast']['station'] for q in records]);day=np.array([q['raw_forecast']['date'] for q in records]);delta=cor-raw
def metric(m):
    assert m.any();return dict(n=int(m.sum()),raw_rmse=float(np.sqrt(np.mean((raw[m]-y[m])**2))),corrected_seed_mean_rmse=float(np.sqrt(np.mean((cor[:,m]-y[m])**2,axis=1)).mean()),raw_bias=float(np.mean(raw[m]-y[m])),corrected_bias=float(np.mean(cor[:,m]-y[m])),mean_adjustment=float(delta[:,m].mean()),upward_fraction=float((delta[:,m]>0).mean()))
out={}
for station in ['Niushou','Wangzhai','Mizhuang']:
    sm=st==station;out[station]={'all':metric(sm),'dates':{d:metric(sm&(day==d)) for d in sorted(set(day[sm]))},'reference_COT_bins':{}}
    for label,lo,hi in [('0_to_1',0,1),('1_to_10',1,10),('10_to_30',10,30),('ge30',30,np.inf)]:
        m=sm&(y>=lo)&(y<hi)
        if m.any():out[station]['reference_COT_bins'][label]=metric(m)
(P/'REGIME_DIAGNOSTIC.json').write_text(json.dumps(dict(scope='All saved F_select station-center records; R proxy bins, not CPP/weather truth; posthoc description only',statistics=out),indent=2))
print(json.dumps({s:v['reference_COT_bins'] for s,v in out.items()},indent=2))
