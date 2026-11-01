"""Evaluate existing observed-trained paired heads on the newly saved forecasts."""
import csv,hashlib,json,shutil,sys,time
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
import torch
B=Path(__file__).resolve().parent
P=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907')
H=P/'experiments/cloud_probability_input_20261020'
OLD=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/HunanDyDiff_20261028/hunan_full_validation_20261030/evaluation_attempt1')
sys.path.insert(0,str(H))
from train_four_arm import FourArmIntervalHead,EXPECTED_COT_MEAN,EXPECTED_COT_STD
def key(r):return(r['station'],str(datetime.fromisoformat(r['init_time_utc'])),str(datetime.fromisoformat(r['target_time_utc'])))
def main():
 start=time.monotonic();out=B/'observed_trained_heads_evaluation_v1';out.mkdir(exist_ok=False)
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 rows=list(csv.DictReader((B/'pack/val/keys.csv').open()));n=len(rows);assert n==66086
 assert len({key(r) for r in rows})==n
 st=np.array([r['station'] for r in rows]);leads=np.array([int(r['lead_minutes']) for r in rows]);dt=np.array([(datetime.fromisoformat(r['target_time_utc'])+timedelta(hours=8)).date().isoformat() for r in rows])
 y=np.load(B/'pack/val/ghi.npy');cl=np.load(B/'pack/val/clear.npy');assert np.isfinite(y).all() and np.isfinite(cl).all() and (cl>0).all()
 ag=np.load(B/'pack/val/agri_raw.npy',mmap_mode='r');geo=np.load(B/'pack/val/geometry.npy',mmap_mode='r');cot=np.load(B/'pack/val/cot_log1p.npy',mmap_mode='r')
 norm=json.loads((P/'experiments/cot_interval_history_20261015_v1/data_attempt2/pack_manifest.json').read_text());selection=json.loads((P/'configs/hunan_active_ghi_heads.json').read_text())
 predictions={a:np.full((3,n),np.nan) for a in ('A','AC')};head_sources={}
 for site in ('sili','zhujia'):
  ids=np.flatnonzero(st==site)
  x=np.concatenate([(ag[ids]-np.asarray(norm['AGRI_norm_mean'],np.float32)[None,None,:,None,None])/np.asarray(norm['AGRI_norm_std'],np.float32)[None,None,:,None,None],geo[ids],(cot[ids]-EXPECTED_COT_MEAN)/EXPECTED_COT_STD,np.zeros((len(ids),3,1,3,3),np.float32)],axis=2).astype(np.float32)
  assert np.isfinite(x).all()
  for si,seed in enumerate((42,43,44)):
   for arm,oldarm in (('A','M00'),('AC','M10')):
    source=selection['heads'][f'{site}/{seed}/{oldarm}'];cp=Path(source['path']);assert hashlib.sha256(cp.read_bytes()).hexdigest()==source['sha256']
    ck=torch.load(cp,map_location='cpu',weights_only=False);assert ck['station']==site and ck['seed']==seed and ck['arm']==oldarm
    model=FourArmIntervalHead().eval().requires_grad_(False);model.load_state_dict(ck['model']);xx=x.copy()
    if arm=='A':xx[:,:,15]=0
    with torch.inference_mode():
     for lo in range(0,len(ids),1792):
      ii=ids[lo:lo+1792];predictions[arm][si,ii]=model(torch.from_numpy(xx[lo:lo+1792])).numpy().astype(np.float64)*cl[ii].astype(np.float64)
    head_sources[f'{site}/{seed}/{oldarm}']=source
   print('PAIR_COMPLETE',site,seed,flush=True)
 assert all(np.isfinite(a).all() for a in predictions.values())
 np.savez_compressed(out/'predictions.npz',truth=y,clear=cl,prediction_A=predictions['A'],prediction_AC=predictions['AC'],seeds=np.array([42,43,44]))
 shutil.copyfile(B/'pack/val/keys.csv',out/'keys.csv')
 oldrows=list(csv.DictReader((OLD/'ghi_rows.csv').open()));oldindex={key(r):i for i,r in enumerate(oldrows)};common=np.array([key(r) in oldindex for r in rows]);assert int(common.sum())==63480
 with np.load(OLD/'ghi_predictions.npz',allow_pickle=False) as z:
  ids=np.flatnonzero(common);oi=np.array([oldindex[key(rows[i])] for i in ids]);assert np.max(abs(y[ids]-z['truth'][oi]))<.001
 def score(mask):
  days=np.unique(dt[mask]);w=np.random.default_rng(20261102).multinomial(len(days),np.full(len(days),1/len(days)),size=5000).astype(float);points=[];boots=[];per=[]
  for site in ('sili','zhujia'):
   keep=mask&(st==site)
   if not keep.any():continue
   pos={d:i for i,d in enumerate(days)};di=np.array([pos[d] for d in dt[keep]]);counts=np.bincount(di,minlength=len(days));den=w@counts
   for si,seed in enumerate((42,43,44)):
    p=np.stack([predictions['A'][si,keep],predictions['AC'][si,keep]]);e=(p-y[keep].astype(float))**2;rmse=np.sqrt(e.mean(axis=1));points.append(rmse)
    sums=np.stack([np.bincount(di,weights=ee,minlength=len(days)) for ee in e]);boots.append(np.sqrt((w@sums.T)/np.maximum(den[:,None],1)))
    per.append({'station':site,'seed':seed,'rows':int(keep.sum()),'no_cot_rmse':float(rmse[0]),'cot_rmse':float(rmse[1]),'delta_rmse':float(rmse[1]-rmse[0])})
  v=np.mean(points,axis=0);bs=np.mean(boots,axis=0)
  return {'no_cot_rmse':float(v[0]),'cot_rmse':float(v[1]),'reduction_percent':float(100*(1-v[1]/v[0])),'delta_ci95':np.quantile(bs[:,1]-bs[:,0],[.025,.975]).tolist(),'rows':int(mask.sum()),'target_bjt_dates':len(days),'runs':per}
 result={'state':'COMPLETE','scope':'newly generated DyDiff forecasts plus observed-trained frozen heads; development validation','heads_training_domain':'observed same-time AGRI/COT','new_training':False,'test_payloads_read':0,'bootstrap_repeats':5000,'head_sources':head_sources,'comparisons':{}}
 for name,mask in [('full',np.ones(n,bool)),('sili',st=='sili'),('zhujia',st=='zhujia'),('lead15_30',leads<=30),('lead45_240',leads>=45),('common_old',common),('common_sili',common&(st=='sili')),('common_zhujia',common&(st=='zhujia'))]:result['comparisons'][name]=score(mask)
 leadmetrics=[{'lead_minutes':lead,**score(leads==lead)} for lead in range(15,241,15)];result['lead_metrics']=leadmetrics
 result['elapsed_seconds']=time.monotonic()-start
 (out/'RESULTS.json').write_text(json.dumps(result,indent=2))
 md='# 本轮预测云图/COT接真实图训练旧头\n\n|范围|无COT RMSE|加COT RMSE|降幅|差值95%区间|\n|---|---:|---:|---:|---|\n'
 for name,q in result['comparisons'].items():md+=f"|{name}|{q['no_cot_rmse']:.3f}|{q['cot_rmse']:.3f}|{q['reduction_percent']:.3f}%|{q['delta_ci95']}|\n"
 md+='\nW/m²；两站逐种子计算后等权。旧头使用自身归一化；不训练，不读test。full为66086条共同有效行，common_old为与原1.737%相同的63480条行。+15/+30三帧上下文含初始化时已知历史，其余时效全部为预测。\n'
 (out/'RESULTS.md').write_text(md)
 print('RESULTS',json.dumps({k:{x:v for x,v in q.items() if x!='runs'} for k,q in result['comparisons'].items()}),flush=True)
if __name__=='__main__':main()
