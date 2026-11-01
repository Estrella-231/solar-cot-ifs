"""Independent saved-prediction scoring; does not import the training scorer."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
P=Path(__file__).parent;B=P/'recovered'
rows=list(csv.DictReader((B/'pack/rows.csv').open(encoding='utf-8-sig')))
records=[];all_dates=set();pairs=[]
for station in ['Niushou','Wangzhai','Mizhuang']:
 for seed in [42,43,44]:
  d=B/'training'/f'{station}_seed{seed}'
  meta=json.loads((d/'COMPLETE.json').read_text());assert meta['state']=='COMPLETE' and meta['test_payloads_read']==0
  assert meta['epochs']==30 and meta['global_batch']==512
  a=np.load(d/'A_validation.npz');b=np.load(d/'AC_validation.npz')
  assert np.array_equal(a['global_indices'],b['global_indices']) and np.array_equal(a['truth'],b['truth']) and np.array_equal(a['clear'],b['clear'])
  ii=a['global_indices'];t=a['truth'].astype(np.float64);pred=np.stack([a['prediction'],b['prediction']]).astype(np.float64);ds=np.array([rows[i]['time_bjt'][:8] for i in ii]);all_dates.update(ds)
  assert all(rows[i]['station']==station and rows[i]['split']=='val' for i in ii)
  err=pred-t;rmse=np.sqrt(np.mean(err**2,axis=1));mae=np.mean(abs(err),axis=1);bias=np.mean(err,axis=1)
  records.append({'station':station,'seed':seed,'n':len(t),'A_rmse':rmse[0],'AC_rmse':rmse[1],'A_mae':mae[0],'AC_mae':mae[1],'A_bias':bias[0],'AC_bias':bias[1]});pairs.append((station,seed,ds,err))
dates=sorted(all_dates);sums=[];counts=[]
for st,seed,ds,err in pairs:
 sums.append([[float(np.sum(err[a,ds==d]**2)) for d in dates] for a in [0,1]]);counts.append([int(np.sum(ds==d)) for d in dates])
sums=np.array(sums);counts=np.array(counts)
rng=np.random.default_rng(20261102);draws=np.stack([rng.choice(dates,len(dates),replace=True) for _ in range(5000)])
w=np.stack([np.sum(draws==d,axis=1) for d in dates],axis=1)
den=w@counts.T
sq=np.stack([w@sums[:,a,:].T for a in [0,1]],axis=2)
boot=np.sqrt(sq/den[:,:,None]);point=np.array([[r['A_rmse'],r['AC_rmse']] for r in records]);mean=point.mean(0);macro_boot=boot.mean(1);delta_boot=macro_boot[:,1]-macro_boot[:,0]
ci=np.quantile(delta_boot,[.025,.975]).tolist();by={}
for st in ['Niushou','Wangzhai','Mizhuang']:
 ids=[i for i,r in enumerate(records) if r['station']==st];v=point[ids].mean(0);bb=boot[:,ids].mean(1)
 by[st]={'A_rmse':v[0],'AC_rmse':v[1],'gain_pct':100*(v[0]-v[1])/v[0],'delta_ci95':np.quantile(bb[:,1]-bb[:,0],[.025,.975]).tolist()}
main=json.loads((B/'RESULTS.json').read_text());assert abs(main['station_seed_equal_RMSE']['A']-mean[0])<1e-9 and abs(main['station_seed_equal_RMSE']['AC']-mean[1])<1e-9
assert np.max(abs(np.array(main['date_block_95CI'])-ci))<1e-9
r={'status':'PASS','computed_from_saved_predictions':True,'imported_original_scorer':False,'A_rmse':mean[0],'AC_rmse':mean[1],'gain_pct':100*(mean[0]-mean[1])/mean[0],'gain_ci95':np.quantile(100*(macro_boot[:,0]-macro_boot[:,1])/macro_boot[:,0],[.025,.975]).tolist(),'delta':mean[1]-mean[0],'delta_ci95':ci,'by_station':by,'target_dates':dates,'n_validation_station_times':sum(r['n'] for r in records)//3,'improved_pairs':int(np.sum(point[:,1]<point[:,0])),'cells':records,'test_payloads_read':0}
(P/'INDEPENDENT_FP64_RESULTS.json').write_text(json.dumps(r,indent=2))
with (P/'station_seed_metrics.csv').open('w',newline='') as f:writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
lines=['# 湖北真实三帧同期GHI：修正太阳几何后的配对结果','','验证集；冻结正确太阳几何R。AGRI与几何相同，唯一变化为是否输入同图反演COT。','', '|站点|无COT RMSE|加COT RMSE|降幅|ΔRMSE 95%日期块区间|','|---|---:|---:|---:|---|']
for st,v in by.items():lines.append(f"|{st}|{v['A_rmse']:.3f}|{v['AC_rmse']:.3f}|{v['gain_pct']:.3f}%|{v['delta_ci95']}|")
lines+= [f"|三站等权|{mean[0]:.3f}|{mean[1]:.3f}|{r['gain_pct']:.3f}%|{ci}|",'',f"单位W/m²；{r['n_validation_station_times']}条共同站点—时次，{len(dates)}个BJT目标日期。三站各3个种子，{r['improved_pairs']}/9配对改善。先逐站逐种子算RMSE，再等权平均；5000次配对目标日期块bootstrap。",'', '旧10.54%为单帧、原几何、另一共同样本范围；两轮百分比不能直接解释为几何修复的因果效应。当前结果为验证，不是独立test。', '', '权重、逐样本预测、norm、初始化及顺序摘要、实际更新数见recovered/training/；代码与计划在本目录。']
(P/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps({k:v for k,v in r.items() if k not in ('cells','target_dates')},indent=2))
