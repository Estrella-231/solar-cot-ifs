"""Saved-prediction behavior for the Sili predicted-sunny anomaly; no training."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907')
OUT=ROOT/'audits/station_weather_prediction_shift_20260912.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def summary(x):
    x=np.asarray(x,np.float64);return dict(mean=float(x.mean()),std=float(x.std()),quantiles=np.quantile(x,[0,.01,.1,.5,.9,.99,1]).tolist())
def main():
    assert not OUT.exists();pack=ROOT/'data/head_pack_trainval_20260908_v1';audit=json.loads((pack/'audit.json').read_text())
    d={}
    for k in ('split','station','ghi','clear'):
        spec=audit['arrays'][k];assert sha(pack/spec['name'])==spec['sha256'];d[k]=np.load(pack/spec['name'])
    val=np.flatnonzero(d['split']==1);wrun=ROOT/'runs/weather_coverage_head_seed42_20260912_v2'
    done=json.loads((wrun/'complete.json').read_text());wp=wrun/'validation_predictions.npz';assert sha(wp)==done['predictions_sha256']
    with np.load(wp) as z:
        assert np.array_equal(z['pack_row'],val);focus=(d['station'][val]==0)&(z['predicted_class']==0)
    result={}
    for cohort,lr in (('original',.0003),('qc1',.001)):
        result[cohort]={}
        for seed in (42,43,44):
            dirname=f'head_ghi_{cohort}_seed42_20260910_v1' if seed==42 else f'head_ghi_multiseed_{cohort}_seed{seed}_20260910_v1'
            pred={}
            for arm in ('A','AC'):
                run=ROOT/'runs'/dirname/f'{arm}_lr{lr:g}_seed{seed}';receipt=json.loads((run/'complete.json').read_text());f=run/'validation_predictions.npz'
                assert sha(f)==receipt['predictions_sha256']
                with np.load(f) as z:
                    assert np.array_equal(z['pack_row'],val);pred[arm]=z['pred_kt'].astype(np.float64)
            truth=d['ghi'][val][focus];clear=d['clear'][val][focus]
            item={'truth_ghi':summary(truth),'clear_ghi':summary(clear)}
            for arm in ('A','AC'):
                ghi=pred[arm][focus]*clear;error=ghi-truth
                item[arm]=dict(pred_kt=summary(pred[arm][focus]),pred_ghi=summary(ghi),error=summary(error),
                    rmse=float(np.sqrt(np.mean(error**2))),mae=float(np.mean(abs(error))),bias=float(error.mean()))
            shift=(pred['AC'][focus]-pred['A'][focus])*clear
            item['AC_minus_A']=dict(predicted_ghi_shift=summary(shift),fraction_increase=float((shift>0).mean()),
                error_correlation=float(np.corrcoef(shift,item['A']['error']['mean']+np.zeros(len(shift)))[0,1]) if False else None)
            result[cohort][str(seed)]=item
    OUT.write_text(json.dumps(dict(state='COMPLETE_SILI_SUNNY_PREDICTION_SHIFT',test_used=False,n=int(focus.sum()),result=result),indent=2,allow_nan=False))
    print('COMPLETE',int(focus.sum()))
if __name__=='__main__':main()
