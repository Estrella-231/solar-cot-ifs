"""Independent FP64 re-score of immutable four-arm predictions, by BJT date."""
import argparse,json,csv,hashlib
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--training',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    assert json.loads((a.training/'ALL_COMPLETE.json').read_text())['state']=='ALL_36_HEADS_COMPLETE';a.out.mkdir(parents=True,exist_ok=False)
    arms=['G0','G1','G2','G3'];sites=['Niushou','Wangzhai','Mizhuang'];seeds=[42,43,44];records=[];daily={};pred_by={};rows_count=0;hourly=[]
    site_reference={}
    for si,s in enumerate(sites):
        for sj,seed in enumerate(seeds):
            root=a.training/'pairs'/f'{s}_seed{seed}';previous=None;common=[]
            receipt=json.loads((root/'COMPLETE.json').read_text());assert receipt['base_unchanged']
            for group in [('G0','G1'),('G2','G3')]:
                x,y=[receipt['arms'][k] for k in group]
                for k in ['initialization_sha256','order_sha256','updates','parameters']:assert x[k]==y[k],(s,seed,k)
            for j,arm in enumerate(arms):
                p=root/(arm+'_validation_predictions.npz');z=np.load(p)
                keys=list(zip(z['station'],z['init_time_bjt'],z['target_time_bjt'],z['lead_minutes']));assert len(keys)==len(set(keys))
                if previous is not None:
                    assert keys==previous[0];assert np.array_equal(z['truth_ghi_wm2'],previous[1]);assert np.array_equal(z['clear_sky_wm2'],previous[2])
                else:
                    previous=(keys,z['truth_ghi_wm2'].copy(),z['clear_sky_wm2'].copy())
                    if s in site_reference:
                        assert keys==site_reference[s][0] and np.array_equal(z['truth_ghi_wm2'],site_reference[s][1]) and np.array_equal(z['clear_sky_wm2'],site_reference[s][2])
                    else:site_reference[s]=previous
                pred=np.asarray(z['pred_ghi_wm2'],np.float64);truth=np.asarray(z['truth_ghi_wm2'],np.float64);assert np.isfinite(pred).all();err=pred-truth;common.append(err)
                metric={'station':s,'seed':seed,'arm':arm,'n':len(err),'rmse':float(np.sqrt(np.mean(err**2))),'mae':float(np.mean(abs(err))),'bias':float(np.mean(err)),'high_ghi_n':int((truth>=800).sum()),'high_ghi_bias':float(np.mean(err[truth>=800])) if (truth>=800).any() else None}
                records.append(metric);dates=np.array([v[:8] for v in z['target_time_bjt']]);lead=z['lead_minutes'].astype(int)
                for h in range(4):
                    m=(lead>h*60)&(lead<=(h+1)*60)
                    hourly.append({'station':s,'seed':seed,'arm':arm,'hour_bin':h+1,'n':int(m.sum()),'rmse':float(np.sqrt(np.mean(err[m]**2)))})
                for d in set(dates):
                    if d not in daily:daily[d]=np.zeros((3,3,4,4),np.float64)
                    v=err[dates==d];daily[d][si,sj,j]=[np.sum(v*v),np.sum(abs(v)),np.sum(v),len(v)]
                pred_by[(s,seed,arm)]=pred
            rows_count+=len(common[0])
    ds=sorted(daily);day=np.stack([daily[d] for d in ds]);total=day.sum(0);point=np.sqrt(total[...,0]/total[...,3]).mean((0,1))
    rng=np.random.default_rng(20261022);weights=rng.multinomial(len(ds),np.ones(len(ds))/len(ds),size=5000)
    sse=weights@day[...,0].reshape(len(ds),-1);nn=weights@day[...,3].reshape(len(ds),-1)
    bs=np.sqrt(sse/nn).reshape(5000,3,3,4).mean((1,2))
    comps={}
    for candidate,control in [('G1','G0'),('G3','G2'),('G3','G1'),('G2','G1'),('G3','G0')]:
        j=arms.index(candidate);k=arms.index(control);ci=np.percentile(bs[:,j]-bs[:,k],[2.5,97.5]).tolist()
        site=[];seed_ok=0
        for s in sites:
            v={ar:np.mean([r['rmse'] for r in records if r['station']==s and r['arm']==ar]) for ar in [candidate,control]};site.append({'station':s,'relative_change_percent':100*(v[candidate]/v[control]-1)})
        for seed in seeds:
            vals={ar:np.mean([r['rmse'] for r in records if r['seed']==seed and r['arm']==ar]) for ar in [candidate,control]};seed_ok+=int(vals[candidate]<vals[control])
        comps[candidate+'_minus_'+control]={'rmse_difference':float(point[j]-point[k]),'relative_rmse_reduction_percent':float(100*(1-point[j]/point[k])),'ci95_date_block':ci,'by_station':site,'improving_seed_count':seed_ok,'passes_statistical_and_site_gate':bool(point[j]<point[k] and ci[1]<0 and max(x['relative_change_percent'] for x in site)<=1 and seed_ok>=2)}
    result={'state':'COMPLETE_FP64_PAIRED_VALIDATION_SCORE','scope':'development validation, no independent test','equal_station_equal_seed_rmse':dict(zip(arms,map(float,point))),'comparisons':comps,'target_dates':ds,'paired_station_seed_rows':rows_count,'unique_station_rows':rows_count//3,'test_payloads_read':0,'buffer_payloads_read':0,'source_manifest_sha256':hashlib.sha256((a.training/'ALL_COMPLETE.json').read_bytes()).hexdigest()}
    (a.out/'score.json').write_text(json.dumps(result,indent=2))
    for name,recs in [('arm_metrics.csv',records),('by_hour.csv',hourly)]:
        with (a.out/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(recs[0]));w.writeheader();w.writerows(recs)
    text=['# 四组湖北GHI预报配对结果','', '开发验证；预测AGRI及历史/预测COT均来自冻结资产。不是独立test。','', '|输入|三站、三seed等权RMSE W/m²|','|---|---:|']
    for label,arm in [('无COT','G0'),('历史COT','G1'),('历史COT＋无未来COT修正头','G2'),('历史COT＋未来COT修正头','G3')]:text.append(f'|{label}|{result["equal_station_equal_seed_rmse"][arm]:.3f}|')
    for key,v in comps.items():text+=['',f'{key}: ΔRMSE={v["rmse_difference"]:.3f} W/m²，日期块95%区间 {v["ci95_date_block"]}，相对降幅{v["relative_rmse_reduction_percent"]:.3f}%。']
    text+=['','G3对G1的收益仍需与G2比较；未来COT不足以超过同容量修正对照时，不声称COT净增量。','归一化/选模只用各自训练日期；validation只在36个模型完成后评分。未改变COT预报本身，因此不存在新增云检测POD/FAR改善的模型结论。']
    (a.out/'RESULT_REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
