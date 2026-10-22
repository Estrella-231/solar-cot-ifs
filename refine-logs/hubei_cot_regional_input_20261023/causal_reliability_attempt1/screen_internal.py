"""Past-only motion reliability; frozen rule, train-internal screening only."""
import csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
P=Path(__file__).resolve().parent;BASE=P.parent;T=BASE/'trajectory_attempt1'
out=P/'internal_attempt1';assert not out.exists()
pack=Path(json.loads((T/'PAIR_PACK_COMPLETE.json').read_text())['pack'])
assert json.loads((pack/'REPORT.json').read_text())['state']=='TRAIN_ONLY_TRAJECTORY_PAIR_PACK_COMPLETE_NOT_TRAINED'
rows=[r for r in csv.DictReader((pack/'pairs.csv').open()) if r['role']=='select']
log=np.load(pack/'history_log.npy',mmap_mode='r');valid=np.load(pack/'history_valid.npy',mmap_mode='r');target=np.load(pack/'target_cot.npy',mmap_mode='r');mask=np.load(pack/'target_valid.npy',mmap_mode='r')
sites=['Niushou','Wangzhai','Mizhuang'];yy=np.array([111,112,110]);xx=np.array([124,126,128]);records=[];start=time.monotonic()
def sample(i,coords):
    v=map_coordinates(log[i],coords,order=1,mode='constant',cval=0.)
    good=map_coordinates(valid[i].astype(float),coords,order=1,mode='constant',cval=0.)>1-1e-6
    return np.expm1(v.astype(np.float64)),good
for r in rows:
    hist=[int(i) for i in r['history_indices'].split('|')];ti=int(r['target_index']);lead=int(r['lead'])
    motion=np.array([float(r['motion_dy']),float(r['motion_dx'])])[:,None]
    point=np.array([yy,xx],np.float64);source=point-(lead/15)*motion
    p=np.expm1(log[hist[-1],yy,xx].astype(np.float64));pv=valid[hist[-1],yy,xx];tv,tgood=sample(hist[-1],source)
    ep=np.zeros(3);et=np.zeros(3);count=np.zeros(3,dtype=int)
    for earlier,later in zip(hist[:-1],hist[1:]):
        truth,good1=sample(later,source);persist,good2=sample(earlier,source);transport,good3=sample(earlier,source-motion)
        good=good1 & good2 & good3;ep+=np.where(good,(persist-truth)**2,0.);et+=np.where(good,(transport-truth)**2,0.);count+=good
    q=np.where((count>=4)&tgood,np.clip((ep-et)/(ep+1e-6),0.,1.),0.)
    mix=p+q*(tv-p)
    assert np.isfinite(mix).all() and np.all(mix>=np.minimum(p,tv)-1e-8) and np.all(mix<=np.maximum(p,tv)+1e-8)
    common=pv & tgood & mask[ti,0,yy,xx]
    for j,st in enumerate(sites):
        if common[j]:records.append(dict(station=st,init=r['init'],target=r['target'],date=r['target'][:8],lead=lead,
            reference=float(target[ti,0,yy[j],xx[j]]),persistence=float(p[j]),transport=float(tv[j]),reliability=float(mix[j]),q=float(q[j]),past_pairs=int(count[j])))
assert len(records)==7617
y=np.array([r['reference'] for r in records]);pred=np.array([[r[m] for r in records] for m in ['persistence','transport','reliability']]);e=pred-y
sitesid=np.array([sites.index(r['station']) for r in records]);lead=np.array([r['lead'] for r in records]);days,dayid=np.unique([r['date'] for r in records],return_inverse=True)
cell=sitesid*16+lead//15-1;sse=np.zeros((3,48,len(days)));n=np.zeros((48,len(days)))
for c in range(48):
    m=cell==c;n[c]=np.bincount(dayid[m],minlength=len(days))
    for j in range(3):sse[j,c]=np.bincount(dayid[m],weights=e[j,m]**2,minlength=len(days))
main=np.sqrt(sse.sum(2)/n.sum(1)).mean(1)
w=np.random.default_rng(20261023).multinomial(len(days),np.ones(len(days))/len(days),5000);den=n@w.T;assert np.all(den>0)
boot=np.sqrt(np.einsum('mcd,bd->mcb',sse,w)/den[None]).mean(1);ci=np.quantile(boot[2]-boot[0],[.025,.975]).tolist()
by_site={st:np.sqrt(sse[:,i*16:(i+1)*16].sum(2)/n[i*16:(i+1)*16].sum(1)).mean(1).tolist() for i,st in enumerate(sites)}
thick=y>=30;thick_rmse=np.sqrt((e[:,thick]**2).mean(1));thick_bias=e[:,thick].mean(1)
q=np.array([r['q'] for r in records]);gates=dict(ci_upper_below_zero=ci[1]<0,every_site_not_worse=all(v[2]<=v[0] for v in by_site.values()),thick_not_worse=bool(thick_rmse[2]<=thick_rmse[0]))
report=dict(state='PAST_ONLY_RELIABILITY_INTERNAL_SCREEN_COMPLETE',methods=['persistence','transport','reliability'],main_rmse=main.tolist(),
    delta_ci95=ci,by_site=by_site,thick_n=int(thick.sum()),thick_rmse=thick_rmse.tolist(),thick_bias=thick_bias.tolist(),
    active_q_fraction=float(np.mean(q>0)),q_mean=float(q.mean()),gates=gates,may_advance_to_outer_validation=all(gates.values()),
    rows=len(records),target_dates=days.tolist(),seconds=time.monotonic()-start,plan_sha256=hashlib.sha256((P/'EXPERIMENT_PLAN.md').read_bytes()).hexdigest(),
    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),outer_validation_payloads_read=0,test_buffer_payloads_read=0,GHI_labels_read=0,
    note='Internal development screening only; q uses past R-COT errors, not future CPP; no parameters fitted')
out.mkdir();(out/'SCORES.json').write_text(json.dumps(report,indent=2))
with (out/'pixels.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
print(json.dumps(report,indent=2))
