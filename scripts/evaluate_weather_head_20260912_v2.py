"""Recompute paired existing GHI predictions by causal predicted weather labels."""
import json
from pathlib import Path
import numpy as np
from train_weather_head_20260912_v2 import sha,write,classification
from evaluate_clp_effect_20260912 import score
ROOT=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907')
def main():
    run=ROOT/'runs/weather_coverage_head_seed42_20260912_v2';pack=ROOT/'data/head_pack_trainval_20260908_v1'
    dest=run/'paired_cot_effect.json';assert not dest.exists()
    done=json.loads((run/'complete.json').read_text());assert done['test_used'] is False
    assert sha(run/'validation_predictions.npz')==done['predictions_sha256']
    audit=json.loads((pack/'audit.json').read_text());data={}
    for k in ('split','station','lead','ghi','clear'):
        assert sha(pack/audit['arrays'][k]['name'])==audit['arrays'][k]['sha256'];data[k]=np.load(pack/audit['arrays'][k]['name'])
    val=np.flatnonzero(data['split']==1)
    with np.load(run/'validation_predictions.npz') as z:
        assert np.array_equal(z['pack_row'],val) and len(val)==99849
        assert np.array_equal(z['station'],data['station'][val]) and np.array_equal(z['lead'],data['lead'][val])
        pred=z['predicted_class'];ref=z['reference_class'];prob=z['probabilities'];target=z['target_id']
        assert np.array_equal(pred,prob.argmax(1)) and np.isfinite(prob).all() and np.allclose(prob.sum(1),1,atol=1e-6)
    valid=ref>=0
    metric=classification(ref[valid],prob[valid]);assert abs(metric['accuracy']-done['metrics']['forecast_rows']['accuracy'])<1e-12
    groups={'all':np.ones(len(val),dtype=bool),'reference_available':valid}
    for k,name in enumerate(('sunny','partly_cloudy','overcast')):
        groups['predicted_'+name]=pred==k
        groups['predicted_labeled_'+name]=(pred==k)&valid
        groups['reference_'+name]=ref==k
    assert sum(int(groups['predicted_'+n].sum()) for n in ('sunny','partly_cloudy','overcast'))==len(val)
    counts={k:dict(rows=int(m.sum()),unique_station_targets=len(np.unique(target[m])),stations={str(s):int((m&(data['station'][val]==s)).sum()) for s in (0,1)}) for k,m in groups.items()}
    results={};summary={}
    for cohort,lr in [('original',.0003),('qc1',.001)]:
        results[cohort]={}
        for seed in (42,43,44):
            dirname=f'head_ghi_{cohort}_seed42_20260910_v1' if seed==42 else f'head_ghi_multiseed_{cohort}_seed{seed}_20260910_v1'
            arms={}
            for arm in ('A','AC'):
                src=ROOT/'runs'/dirname/f'{arm}_lr{lr:g}_seed{seed}'
                receipt=json.loads((src/'complete.json').read_text());assert receipt['test_used'] is False
                assert sha(src/'validation_predictions.npz')==receipt['predictions_sha256']
                with np.load(src/'validation_predictions.npz') as z:
                    assert np.array_equal(z['pack_row'],val)
                    for a,b in [('observed_ghi','ghi'),('clear_sky_ghi','clear'),('station','station'),('lead','lead')]:assert np.array_equal(z[a],data[b][val])
                    e=z['pred_kt'].astype('float64')*data['clear'][val]-data['ghi'][val]
                arms[arm]={k:score(e,data['station'][val],m) for k,m in groups.items()}
            results[cohort][str(seed)]=arms
        summary[cohort]={}
        for name in groups:
            a=[results[cohort][str(s)]['A'][name]['station_equal_rmse'] for s in (42,43,44)]
            c=[results[cohort][str(s)]['AC'][name]['station_equal_rmse'] for s in (42,43,44)]
            if None in a+c:summary[cohort][name]=dict(state='INSUFFICIENT_STATION_COVERAGE');continue
            delta=np.array(c)-a;summary[cohort][name]=dict(A=float(np.mean(a)),AC=float(np.mean(c)),delta_mean=float(delta.mean()),delta_std=float(delta.std(ddof=1)),delta_values=delta.tolist())
    write(dest,dict(state='COMPLETE_PREDICTED_WEATHER_COT_COMPARISON',test_used=False,weather_classifier_seeds=[42],ghi_seeds=[42,43,44],counts=counts,classification=metric,summary=summary,results=results,
        classification_reference='instantaneous patch cloud coverage pseudo-label; not ground daily weather',
        scope='all validation rows grouped by one common causal classifier; missing reference only excluded from accuracy, never from predicted grouping',
        caveats=['single classifier seed pilot','three GHI seeds do not measure classifier seed uncertainty','source CPP producer binding unresolved','unique targets may occur in different predicted groups at different leads','reference groups diagnostic only; no oracle model routing']))
    print('WEATHER_COMPARISON_COMPLETE',json.dumps(dict(classification=metric,summary=summary)),flush=True)
if __name__=='__main__':main()
