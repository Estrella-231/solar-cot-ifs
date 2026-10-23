"""Independent math.fsum scoring from saved physical predictions; no model reads."""
import json, math
from pathlib import Path
import numpy as np

P = Path(__file__).resolve().parent
report = json.loads((P/'RESPONSE_SCORES.json').read_text())
sites = ['Niushou', 'Wangzhai', 'Mizhuang']
arms = ['G1', 'no_future_original', 'future_original', 'no_future_joint',
        'forecast_joint', 'persistence_joint', 'observed_R_joint_OFFLINE']
rows = []; max_error = 0.; max_ci_error = 0.
for scope in ['all_rows', 'joint_available']:
    by_horizon = {}
    for horizon in ['all', '1h', '2h', '3h', '4h']:
        scores = []; stats = []; dates = report['summary'][scope][horizon]['dates']
        for site in sites:
            seed_scores = []; seed_stats = []
            for seed in [42, 43, 44]:
                with np.load(P/'replay'/site/f'seed{seed}_response.npz') as z:
                    d = {k:z[k] for k in z.files}
                mask = np.ones(len(d['truth']), bool)
                if scope != 'all_rows': mask &= d['joint_available'] > 0
                if horizon != 'all':
                    h = int(horizon[0]); mask &= (d['lead']>(h-1)*60)&(d['lead']<=h*60)
                values = []; day_stats = []
                for ai, arm in enumerate(arms):
                    errors = d['prediction'][ai]-d['truth']
                    values.append(math.sqrt(math.fsum(float(e)**2 for e in errors[mask])/int(mask.sum())))
                    day_stats.append([[math.fsum(float(e)**2 for e in errors[mask&(d['date']==day)]), int((mask&(d['date']==day)).sum())] for day in dates])
                seed_scores.append(values); seed_stats.append(day_stats)
            scores.append(seed_scores); stats.append(seed_stats)
            if horizon == 'all':
                avg=np.array(seed_scores).mean(0)
                rows.append(dict(scope=scope,station=site,rmse=dict(zip(arms,avg.tolist())),
                    teacher_minus_forecast_by_seed=(np.array(seed_scores)[:,6]-np.array(seed_scores)[:,4]).tolist()))
        points=np.array(scores).mean((0,1)); expected=report['summary'][scope][horizon]
        max_error=max(max_error,max(abs(points[i]-expected['arms'][a]) for i,a in enumerate(arms)))
        # Independent date accumulator, multinomial paired resampling, one date weight shared across all arms.
        stats=np.array(stats).transpose(3,0,1,2,4)
        weights=np.random.default_rng(20261023).multinomial(len(dates),np.ones(len(dates))/len(dates),5000)
        samples=[]
        for w in weights:
            sums=np.tensordot(w,stats,axes=(0,0))
            samples.append(np.sqrt(sums[...,0]/sums[...,1]).mean((0,1)))
        samples=np.array(samples)
        for j,k in [(6,4),(5,4),(4,2),(6,0),(4,0),(4,3),(6,3)]:
            key=f'{arms[j]}_minus_{arms[k]}'
            ci=np.quantile(samples[:,j]-samples[:,k],[.025,.975])
            max_ci_error=max(max_ci_error,float(abs(ci-expected['comparisons'][key]['ci95']).max()))
        by_horizon[horizon]=dict(zip(arms,points.tolist()))
assert max_error<1e-10 and max_ci_error<1e-10
out=dict(state='PASS',max_point_difference=max_error,max_ci_difference=max_ci_error,
         station_results=rows,new_fits=0,test_payloads_read=0)
(P/'INDEPENDENT_SCORE.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
