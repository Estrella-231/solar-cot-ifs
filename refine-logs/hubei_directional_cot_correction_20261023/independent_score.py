"""Independent fsum aggregation and date bootstrap from saved scalar predictions."""
import csv,json,math
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;rows=list(csv.DictReader((P/'correction_samples.csv').open()));report=json.loads((P/'CORRECTION_SCORES.json').read_text())
methods=['raw_forecast','persistence','seed42','seed43','seed44'];stations=['Niushou','Wangzhai','Mizhuang'];dates=report['dates'];buckets={};counts={}
for r in rows:
    key=(methods.index(r['method']),stations.index(r['station'])*16+int(r['lead'])//15-1,dates.index(r['date']))
    buckets.setdefault(key,[]).append((float(r['prediction'])-float(r['target']))**2)
sse=np.zeros((5,48,len(dates)));den=np.zeros((48,len(dates)))
for (m,g,d),v in buckets.items():
    sse[m,g,d]=math.fsum(v)
    if m==0:den[g,d]=len(v)
    else:assert den[g,d]==len(v)
arch=np.load(P/'F_SELECT_DATE_SUMS.npz');assert np.array_equal(den,arch['den']) and np.allclose(sse,arch['sse'],rtol=1e-12,atol=1e-7)
point=np.array([math.fsum(math.sqrt(math.fsum(sse[m,g])/math.fsum(den[g])) for g in range(48))/48 for m in range(5)])
maxdiff=max(abs(point[i]-report['equal_station_lead_rmse'][name]) for i,name in enumerate(methods));assert maxdiff<1e-10
w=np.random.default_rng(20261023).multinomial(len(dates),np.ones(len(dates))/len(dates),5000);div=w@den.T;w=w[(div>0).all(1)];div=w@den.T
boot=np.stack([np.sqrt((w@sse[m].T)/div).mean(1) for m in range(5)],1);cm=boot[:,2:].mean(1);intervals={};cierror=0.
for i,name in enumerate(methods[:2]):
    ci=np.quantile(cm-boot[:,i],[.025,.975]);intervals[name]=ci.tolist();cierror=max(cierror,float(abs(ci-report['comparisons'][name]['ci95']).max()))
assert cierror<1e-10,cierror
result=dict(state='PASS_INDEPENDENT_SAVED_PREDICTION_FSUM_AND_DATE_BOOTSTRAP',rows=len(rows),point_max_abs_difference=float(maxdiff),ci_max_abs_difference=cierror,point=dict(zip(methods,point.tolist())),ci95=intervals,outer_payloads_read=0)
(P/'INDEPENDENT_SCORE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
