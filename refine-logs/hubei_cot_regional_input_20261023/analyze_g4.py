"""Independent FP64 common-pixel scoring and target-date block bootstrap."""
import argparse,csv,json
from pathlib import Path
import numpy as np

ap=argparse.ArgumentParser();ap.add_argument('--pixels',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
raw=list(csv.DictReader(a.pixels.open()));methods=['persistence','transport','forecast'];sites=['Niushou','Wangzhai','Mizhuang']
dates=sorted({r['date'] for r in raw});di={d:i for i,d in enumerate(dates)}
report=dict(state='INDEPENDENT_FP64_G4_SAVED_PAIRS_ANALYSIS',dates=dates,reference='CPP retrieval, not independent truth',
            scope='validation future COT, no GHI',test_payloads_read=0,bootstrap_replicates=5000,
            bootstrap_unit='target BJT date; draws shared across sites/leads/methods',by_scope={})
for scope in ['1x1','3x3']:
    rr=[r for r in raw if scope=='3x3' or r['dr']==r['dc']=='0']
    nn=np.zeros((len(dates),3,16),np.int64);ss=np.zeros((3,len(dates),3,16),np.float64);se=ss.copy();sa=ss.copy()
    common=[r for r in rr if r['common']=='1']
    for r in common:
        d=di[r['date']];s=sites.index(r['station']);l=int(r['lead'])//15-1;truth=float(r['reference']);nn[d,s,l]+=1
        for m,name in enumerate(methods):
            e=float(r[name])-truth;ss[m,d,s,l]+=e*e;se[m,d,s,l]+=e;sa[m,d,s,l]+=abs(e)
    counts=nn.sum(0);assert np.all(counts>0),counts
    rmse=np.sqrt(ss.sum(1)/counts[None]);bias=se.sum(1)/counts[None];mae=sa.sum(1)/counts[None]
    rng=np.random.default_rng(20261023);draws=rng.integers(0,len(dates),(5000,len(dates)))
    bootstrap=[]
    for ids in draws:
        n=nn[ids].sum(0)
        if np.all(n>0):bootstrap.append(np.sqrt(ss[:,ids].sum(1)/n[None]).mean((1,2)))
    bootstrap=np.asarray(bootstrap);assert len(bootstrap)>4900
    comparisons={}
    for m,name in enumerate(methods[1:],1):
        delta=bootstrap[:,m]-bootstrap[:,0]
        comparisons[name+'_minus_persistence']=dict(delta_rmse=float(rmse[m].mean()-rmse[0].mean()),
              ci95=np.quantile(delta,[.025,.975]).tolist(),relative_reduction_percent=float(100*(1-rmse[m].mean()/rmse[0].mean())))
    result=dict(reference_pixel_records=sum(r['reference_valid']=='1' for r in rr),common_pixel_records=len(common),
       persistence_available=sum(r['reference_valid']==r['persistence_valid']=='1' for r in rr),
       transport_available=sum(r['reference_valid']==r['transport_valid']=='1' for r in rr),
       common_coverage_fraction=len(common)/sum(r['reference_valid']=='1' for r in rr),
       methods={name:dict(equal_station_equal_lead_rmse=float(rmse[m].mean()),
             by_station={st:dict(rmse=float(rmse[m,s].mean()),mae=float(mae[m,s].mean()),bias=float(bias[m,s].mean())) for s,st in enumerate(sites)},
             exact_hours={str(h):dict(equal_station_rmse=float(rmse[m,:,h*4-1].mean()),n=int(counts[:,h*4-1].sum())) for h in range(1,5)},
             hourly_lead_bands={str(h):float(rmse[m,:,(h-1)*4:h*4].mean()) for h in range(1,5)}) for m,name in enumerate(methods)},
       comparisons=comparisons,bootstrap_valid_replicates=len(bootstrap),thickness={})
    for label,lo,hi in [('near_zero',0,.1),('thin',.1,10),('moderate',10,30),('thick',30,float('inf'))]:
        rows=[r for r in common if lo<=float(r['reference'])<hi]
        result['thickness'][label]={}
        for name in methods:
            e=np.array([float(r[name])-float(r['reference']) for r in rows])
            result['thickness'][label][name]=dict(n=len(e),pooled_rmse=float(np.sqrt(np.mean(e*e))) if len(e) else None,
                 bias=float(e.mean()) if len(e) else None)
    report['by_scope'][scope]=result
a.out.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
