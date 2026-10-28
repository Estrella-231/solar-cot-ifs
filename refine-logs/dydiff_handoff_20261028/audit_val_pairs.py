"""Independent paired development-validation audit, no test payloads."""
import csv,hashlib,json
from pathlib import Path
import numpy as np

P=Path(__file__).resolve().parent; META=P/'metadata'; ROOT=P.parents[1]
own_file=ROOT/'refine-logs/hubei_cot_independent_20261014/frozen_split_20261015_v1/day_assignment.csv'
def mapping(file):
    with file.open(encoding='utf-8-sig',newline='') as f:return {r['date_BJT']:r['split'] for r in csv.DictReader(f)}
ours=mapping(own_file);theirs=mapping(META/'dydiff_day_assignment_common.csv')
remote=json.loads((META/'VALIDATION_PREDICTION_SHA256.json').read_text())['sha256']
stations=['Niushou','Wangzhai','Mizhuang'];seeds=[42,43,44];arms=['REAL','REAL_COT']
assert len(remote)==18
content={}; days=set();stats=[];station_reference={}
for st in stations:
    for seed in seeds:
        pair=[]
        for arm in arms:
            f=META/f'DyDiff150_{st}_seed{seed}_{arm}.npz'
            assert hashlib.sha256(f.read_bytes()).hexdigest()==remote[f.name]
            with np.load(f,allow_pickle=False) as z:d={k:z[k] for k in z.files}
            assert np.all(d['station']==st) and np.isfinite(d['truth_ghi_wm2']).all() and np.isfinite(d['pred_ghi_wm2']).all()
            for token in np.concatenate([d['init_time_bjt'],d['target_time_bjt']]):
                day=str(token)[:8]; day=day[:4]+'-'+day[4:6]+'-'+day[6:8]
                assert ours.get(day)=='val' and theirs.get(day)=='val', (st,day,ours.get(day),theirs.get(day))
            pair.append(d)
        a,b=pair
        for k in ['station','init_time_bjt','target_time_bjt','lead_minutes','pack_index','truth_ghi_wm2','clear_sky_wm2']:
            assert np.array_equal(a[k],b[k]),(st,seed,k)
        if st not in station_reference:station_reference[st]=a
        else:
            for k in ['station','init_time_bjt','target_time_bjt','lead_minutes','pack_index','truth_ghi_wm2','clear_sky_wm2']:
                assert np.array_equal(a[k],station_reference[st][k]),(st,seed,'cross_seed',k)
        ini=np.array([np.datetime64(str(t)[:4]+'-'+str(t)[4:6]+'-'+str(t)[6:8]+'T'+str(t)[8:10]+':'+str(t)[10:12]) for t in a['init_time_bjt']])
        tar=np.array([np.datetime64(str(t)[:4]+'-'+str(t)[4:6]+'-'+str(t)[6:8]+'T'+str(t)[8:10]+':'+str(t)[10:12]) for t in a['target_time_bjt']])
        assert np.array_equal((tar-ini).astype('timedelta64[m]').astype(int),a['lead_minutes'])
        assert set(a['lead_minutes']).issubset(set(range(15,241,15)))
        keys=list(zip(a['init_time_bjt'],a['target_time_bjt'],a['lead_minutes']));assert len(set(keys))==len(keys)
        dates=np.array([str(t)[:8] for t in a['target_time_bjt']]);days.update(dates.tolist())
        content[(st,seed)]=(a,b,dates)
        rmse=[float(np.sqrt(np.mean((z['pred_ghi_wm2']-z['truth_ghi_wm2'])**2))) for z in pair]
        stats.append(dict(station=st,seed=seed,rows=len(dates),no_COT=rmse[0],with_COT=rmse[1],delta=rmse[1]-rmse[0]))
days=sorted(days);assert len(days)==17
weights=np.random.default_rng(20261028).multinomial(len(days),np.repeat(1./len(days),len(days)),5000)
boot={arm:[] for arm in arms}
for st in stations:
    for seed in seeds:
        a,b,dates=content[(st,seed)];idx=np.searchsorted(days,dates);counts=np.bincount(idx,minlength=len(days));den=weights@counts
        assert (den>0).all()
        for arm,d in zip(arms,[a,b]):
            squared=(d['pred_ghi_wm2']-d['truth_ghi_wm2'])**2
            sums=np.bincount(idx,weights=squared,minlength=len(days))
            boot[arm].append(np.sqrt((weights@sums)/den))
means={arm:float(np.mean([s['no_COT' if arm=='REAL' else 'with_COT'] for s in stats])) for arm in arms}
delta=np.mean(boot['REAL_COT'],axis=0)-np.mean(boot['REAL'],axis=0)
point=means['REAL_COT']-means['REAL'];ci=np.quantile(delta,[.025,.975]).tolist()
result={'state':'PASS_EXISTING_DEVELOPMENT_VALIDATION_ONLY','no_COT_rmse':means['REAL'],'with_COT_rmse':means['REAL_COT'],
        'paired_delta_rmse':point,'relative_reduction_percent':-100*point/means['REAL'],
        'paired_date_bootstrap_95pct':ci,'bootstrap_draws':5000,'bootstrap_seed':20261028,
        'dates':days,'unique_station_rows':sum(next(s['rows'] for s in stats if s['station']==st) for st in stations),
        'station_seed_scores':stats,'heads_retrained':False,'input':'DyDiff forecast AGRI -> frozen R -> COT',
        'model_training':'heads trained on same-target observed AGRI with or without R COT; evaluated on forecasts',
        'test_payloads_read':0,'interval_units':'W/m2','hashes_verified':18}
(P/'INDEPENDENT_VAL_AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='station_seed_scores'},indent=2))
