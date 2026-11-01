"""FP64 recomputation from the saved nine prediction files, without model execution."""
import json
from pathlib import Path
import numpy as np

P=Path(__file__).resolve().parent/'results_attempt2'
raw=json.loads((P/'summary.json').read_text())
assert raw['state']=='COMPLETE_HISTORY_COT_FORECAST_GHI_PAIRED' and raw['test_used'] is False
pred={}
for seed in (42,43,44):
    for arm in ('A','transport','hybrid'):
        with np.load(P/f'{arm}_seed{seed}'/'predictions.npz',allow_pickle=False) as f:pred[seed,arm]={k:f[k] for k in f.files}
ref=pred[42,'A']
for p in pred.values():
    for k in ('row_ids','true_ghi','lead_minutes','target_day','init_time','target_time'):assert np.array_equal(ref[k],p[k]),k
days=np.unique(ref['target_day']);rng=np.random.default_rng(20261015)
# Cluster multiplicities yield the same bootstrap sample as concatenating sampled dates.
sampled=rng.integers(len(days),size=(2000,len(days)))
weights=np.stack([np.bincount(row,minlength=len(days)) for row in sampled])
result={};max_delta=0.
for arm in ('transport','hybrid'):
    result[arm]={}
    for lead in (0,*range(15,241,15)):
        mask=np.ones(len(ref['row_ids']),bool) if lead==0 else ref['lead_minutes']==lead
        n=np.array([(mask&(ref['target_day']==day)).sum() for day in days],np.float64)
        errors={a:np.stack([(pred[s,a]['pred_ghi'].astype(np.float64)-ref['true_ghi'].astype(np.float64)) for s in (42,43,44)]) for a in ('A',arm)}
        point={a:np.sqrt((e[:,mask]**2).mean(1)).mean() for a,e in errors.items()}
        sums={a:np.stack([(e[:,mask&(ref['target_day']==day)]**2).sum(1) for day in days]) for a,e in errors.items()}
        denom=weights@n;keep=denom>0
        boot={a:np.sqrt((weights[keep]@v)/denom[keep,None]).mean(1) for a,v in sums.items()}
        ci=np.quantile(boot[arm]-boot['A'],[.025,.975])
        row=dict(n=int(mask.sum()),dates=int((n>0).sum()),A_rmse=float(point['A']),AC_rmse=float(point[arm]),AC_minus_A=float(point[arm]-point['A']),reduction_percent=float(100*(point['A']-point[arm])/point['A']),delta_ci95=ci.tolist())
        for key in ('A_rmse','AC_rmse','AC_minus_A','reduction_percent','delta_ci95'):
            delta=float(np.max(np.abs(np.array(row[key])-np.array(raw['results'][arm][str(lead)][key]))));max_delta=max(max_delta,delta);assert delta<1e-8,(arm,lead,key,delta)
        result[arm][str(lead)]=row
out=dict(state='PASS_INDEPENDENT_FP64_PREDICTION_RECOMPUTATION',max_difference=max_delta,validation_rows=len(ref['row_ids']),target_dates=len(days),test_payload_read=False,results=result)
(P/'independent_score.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({arm:{k:v for k,v in rr.items() if k in ('0','60','120','180','240')} for arm,rr in result.items()},indent=2))
