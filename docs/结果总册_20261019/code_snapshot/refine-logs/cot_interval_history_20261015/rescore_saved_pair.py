"""Second implementation: score the compact saved-prediction archive, no training needed."""
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[2]/'results/cot_interval_history_20261015'
z=np.load(root/'paired_validation_scoring.npz');p=z['prediction'];truth=z['truth'];st=z['station'];dates=z['dates'];days=np.unique(dates);di=np.searchsorted(days,dates)
rng=np.random.default_rng(20261015);count=rng.multinomial(len(days),np.full(len(days),1/len(days)),2000)
point=[];boots=[]
for station in ('sili','zhujia'):
    keep=st==station;err=(p[:,:,keep]-truth[keep])**2
    point.append(np.sqrt(err.mean(2)))
    rw=count[:,di[keep]];den=rw.sum(1)
    # Direct per-row weighting, independent of analyzer's per-date binning.
    boot=np.empty((2000,3,2))
    for seed in range(3):
        for arm in range(2):boot[:,seed,arm]=np.sqrt((rw@err[seed,arm])/den)
    boots.append(boot)
means=np.stack(point).mean((0,1));b=np.stack(boots).mean((0,2));interval=np.quantile(b[:,1]-b[:,0],[.025,.975]);ref=json.loads((root/'independent_score.json').read_text())['groups']['all_validation']
assert abs(means[0]-ref['A_RMSE'])<1e-8 and abs(means[1]-ref['AC_RMSE'])<1e-8
assert np.max(np.abs(interval-ref['date_cluster_CI95']))<1e-8
result=dict(state='PASS_SECOND_IMPLEMENTATION_ROW_WEIGHTED_BOOTSTRAP',A_RMSE=float(means[0]),AC_RMSE=float(means[1]),delta_RMSE=float(means[1]-means[0]),CI95=interval.tolist(),test_payloads_read=0)
(root/'secondary_recompute.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
