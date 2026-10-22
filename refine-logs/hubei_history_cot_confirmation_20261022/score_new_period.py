"""FP64 common-key scores and paired target-date intervals; no model selection."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False)
    receipt=json.loads((a.predictions/'COMPLETE.json').read_text());assert receipt['state']=='FROZEN_NEW_PERIOD_PAIRED_GHI_PREDICTIONS_COMPLETE'
    loaded={}
    for r in receipt['predictions']:
        f=Path(r['file']);assert hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256']
        with np.load(f,allow_pickle=False) as z:loaded[(r['station'],r['seed'],r['arm'])]={k:z[k].copy() for k in z.files}
    reports={}
    for scope in ['primary_shared_dates','two_station_extension']:
        cells=[];stations=['Niushou','Wangzhai','Mizhuang'] if scope=='primary_shared_dates' else ['Niushou','Wangzhai']
        for st in stations:
            for seed in [42,43,44]:
                x,y=loaded[(st,seed,'G0')],loaded[(st,seed,'G1')]
                for key in ['ghi','clear','init_time_BJT','target_time_BJT','lead_minutes','analysis']:
                    assert np.array_equal(x[key],y[key]),(st,seed,key)
                mask=x['analysis']==scope;assert mask.any();truth=x['ghi'][mask].astype(np.float64)
                errors=np.stack([x['pred_ghi'][mask]-truth,y['pred_ghi'][mask]-truth])
                assert np.isfinite(errors).all();dates=np.array([t[:10] for t in x['target_time_BJT'][mask]])
                cells.append(dict(station=st,seed=seed,errors=errors,dates=dates,truth=truth,leads=x['lead_minutes'][mask]))
        days=sorted(set(t for c in cells for t in c['dates']));index={t:i for i,t in enumerate(days)}
        squared=np.zeros((2,len(cells),len(days)),np.float64);counts=np.zeros((len(cells),len(days)),np.int64)
        table=[]
        for j,c in enumerate(cells):
            e=c['errors'];table.append(dict(station=c['station'],seed=c['seed'],rows=e.shape[1],dates=len(set(c['dates'])),
                no_cot_rmse=float(np.sqrt(np.mean(e[0]**2))),history_cot_rmse=float(np.sqrt(np.mean(e[1]**2))),
                no_cot_mae=float(np.mean(abs(e[0]))),history_cot_mae=float(np.mean(abs(e[1]))),
                no_cot_bias=float(np.mean(e[0])),history_cot_bias=float(np.mean(e[1])),
                high_ghi_rows=int((c['truth']>=800).sum()),
                no_cot_high_ghi_bias=float(np.mean(e[0,c['truth']>=800])) if (c['truth']>=800).any() else None,
                history_cot_high_ghi_bias=float(np.mean(e[1,c['truth']>=800])) if (c['truth']>=800).any() else None))
            for d in set(c['dates']):
                m=c['dates']==d;counts[j,index[d]]=int(m.sum());squared[:,j,index[d]]=np.sum(e[:,m]**2,axis=1)
        rng=np.random.default_rng(20261022);weights=rng.multinomial(len(days),np.full(len(days),1/len(days)),size=5000)
        den=counts@weights.T;usable=(den>0).all(axis=0);assert usable.sum()>4900
        rmse=np.sqrt((squared@weights[usable].T)/den[:,usable][None,:,:]).mean(axis=1)
        diff=rmse[1]-rmse[0];percent=100*(rmse[0]-rmse[1])/rmse[0]
        no=float(np.mean([r['no_cot_rmse'] for r in table]));yes=float(np.mean([r['history_cot_rmse'] for r in table]))
        hour=[]
        for lo,hi in [(0,60),(60,120),(120,180),(180,240)]:
            values=[]
            for c in cells:
                m=(c['leads']>lo)&(c['leads']<=hi);assert m.any();values.append(np.sqrt(np.mean(c['errors'][:,m]**2,axis=1)))
            v=np.mean(values,axis=0);hour.append(dict(lead_from_exclusive=lo,lead_to_inclusive=hi,no_cot_rmse=float(v[0]),history_cot_rmse=float(v[1])))
        bysite={st:dict(no_cot_rmse=float(np.mean([r['no_cot_rmse'] for r in table if r['station']==st])),
            history_cot_rmse=float(np.mean([r['history_cot_rmse'] for r in table if r['station']==st]))) for st in stations}
        ci=np.quantile(diff,[.025,.975]).tolist();seed_directions=[np.mean([r['history_cot_rmse']-r['no_cot_rmse'] for r in table if r['seed']==s]) for s in [42,43,44]]
        reports[scope]=dict(no_cot_rmse=no,history_cot_rmse=yes,difference_wm2=yes-no,relative_rmse_reduction_percent=100*(no-yes)/no,
            difference_ci95=ci,relative_reduction_ci95=np.quantile(percent,[.025,.975]).tolist(),dates=days,
            date_clusters=len(days),bootstrap_draws=5000,usable_draws=int(usable.sum()),station_seed_metrics=table,
            by_station=bysite,hour_bands=hour,paired_forecast_rows_per_seed=sum(r['rows'] for r in table if r['seed']==42),
            candidate_confirmation_supported=bool(ci[1]<0 and sum(v<0 for v in seed_directions)>=2 and
                all(v['history_cot_rmse']<=v['no_cot_rmse']*1.01 for v in bysite.values())))
    report=dict(state='NEW_SOURCE_CONFIRMATION_SCORED_NO_REFIT',scopes=reports,old_test_payloads_read=0,old_buffer_payloads_read=0,
        input_mode='historical observed AGRI and R-COT + frozen SimVP future AGRI, predictable solar geometry; no future COT',
        scope_note='summer new-source confirmation; source and timestamp limitations remain; not full-year evidence',
        scoring_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.out/'SCORES.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# 新时段历史COT辅助GHI预报确认','','固定模型；无重训、无新时段选模。主范围为三站共同日期7月15日至8月9日，后续两站日期单列。','','|范围|无COT RMSE|历史COT RMSE|降幅|日期配对ΔRMSE 95%区间|天气日期数|','|---|---:|---:|---:|---|---:|']
    for scope,r in reports.items():
        lines.append(f"|{scope}|{r['no_cot_rmse']:.3f}|{r['history_cot_rmse']:.3f}|{r['relative_rmse_reduction_percent']:.2f}%|[{r['difference_ci95'][0]:.3f}, {r['difference_ci95'][1]:.3f}]|{r['date_clusters']}|")
    lines+=['','单位W/m²；先逐站逐seed计算RMSE，再等权平均。以同一目标BJT日期共同重采样，三seed不增加天气日期数。旧封存test/buffer未读取；本结果是新增夏季时段确认，不是全年泛化。','']
    (a.out/'RESULT_REPORT.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps(reports['primary_shared_dates'],indent=2),flush=True)
if __name__=='__main__':main()
