"""Fixed 27-parameter no-intercept COT source comparison, C dates only."""
import csv,json,hashlib,sys,os,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments');H=ROOT/'hubei_forecast_cot_increment_20261022';O=ROOT/'hubei_cot_linear_increment_20261023'
sys.path.insert(0,str(H/'code_attempt1'))
from history_base import HistoryCOTGHIHead,sha256
SITES=['Niushou','Wangzhai','Mizhuang'];SEEDS=[42,43,44]

def features(cot,valid,leads,fit):
    values=cot[fit][valid[fit]];scale=max(float(np.sqrt(np.mean(values.astype(np.float64)**2))),1e-6)
    c=np.where(valid,cot/scale,0.);l=leads/240.;basis=np.stack((np.ones_like(l),l,l*l),1)
    return (c[:,:,None]*basis[:,None,:]).reshape(len(c),27).astype(np.float64),scale

def main():
    O.mkdir(exist_ok=True);out=O/'screen_attempt1';out.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    all_records=[];day_stats={};witness=[]
    for si,st in enumerate(SITES):
        p=H/'prepared_attempt1'/st;meta=json.loads((p/'meta.json').read_text());rows=list(csv.DictReader((p/'rows.csv').open()));roles=np.load(p/'roles.npy')
        # Determine permitted indices from metadata BEFORE loading tensors. Only C payload slices are materialized.
        take=np.flatnonzero(np.isin(roles,['C_fit','C_select']));rr=[rows[i] for i in take];role=roles[take];fit=role=='C_fit';select=role=='C_select'
        assert fit.any() and select.any() and not set(r['target_time_bjt'][:8] for r,v in zip(rr,fit) if v)&set(r['target_time_bjt'][:8] for r,v in zip(rr,select) if v)
        data={k:np.asarray(np.load(p/(k+'.npy'),mmap_mode='r')[take]) for k in ['future','history','fusion_features','ghi','clear']}
        assert all(np.isfinite(v).all() for v in data.values());leads=np.array([int(r['lead_minutes']) for r in rr],np.float64)
        ff=data['fusion_features'];histmask=ff[:,207:279].reshape(-1,8,9)[:,-1].astype(bool)
        # History log COT is restored from saved train-standardized features; invalid slots remain masked.
        bn=meta['base_norm'];hc=ff[:,135:207].reshape(-1,8,9)[:,-1]*bn['history_cot_log1p_std']+bn['history_cot_log1p_mean']
        fn=meta['future_cot_norm'];fc=ff[:,279:288]*fn['log1p_std']+fn['log1p_mean'];fv=ff[:,297:306].astype(bool)
        assert ff.shape[1]==307;assert np.all(hc[histmask]>=-1e-6) and np.all(fc[fv]>=-1e-6)
        ph,sh=features(hc,histmask,leads,fit);pf,sf=features(fc,fv,leads,fit);clear=data['clear'].astype(np.float64);truth=data['ghi'].astype(np.float64)
        for sj,seed in enumerate(SEEDS):
            ck=H/'training_attempt1/pairs'/f'{st}_seed{seed}'/'G0_best.pt';model=HistoryCOTGHIHead().cuda().eval();m=torch.load(ck,map_location='cuda',weights_only=False);model.load_state_dict(m['model']);base=[]
            with torch.inference_mode():
                for j in range(0,len(take),1536):
                    f=torch.from_numpy(np.ascontiguousarray(data['future'][j:j+1536])).cuda();h=torch.from_numpy(np.ascontiguousarray(data['history'][j:j+1536])).cuda();c=torch.zeros((len(f),8,1,3,3),device='cuda');base.append(model(f,h,c).cpu().numpy())
            base=np.concatenate(base).astype(np.float64)*clear;pred=[base];coefs=[]
            for design,valid,name in [(ph,histmask,'H'),(pf,fv,'F')]:
                x=design*clear[:,None]/1000.;y=(truth-base)/1000.;xf=x[fit];yf=y[fit]
                beta=np.linalg.solve(xf.T@xf/fit.sum()+.01*np.eye(27),xf.T@yf/fit.sum());q=np.maximum(0,base+x@beta*1000)
                assert np.array_equal(q[~valid.any(1)],base[~valid.any(1)])
                resid=float(np.max(abs((xf.T@xf/fit.sum()+.01*np.eye(27))@beta-xf.T@yf/fit.sum())))
                assert resid<1e-10;pred.append(q);coefs.append(beta);witness.append(dict(station=st,seed=seed,arm=name,normal_equation_max_abs_error=resid,all_invalid_rows=int((~valid.any(1)).sum())))
            pred=np.array(pred);dates=np.array([r['target_time_bjt'][:8] for r in rr]);vi=np.flatnonzero(select)
            np.savez_compressed(out/f'{st}_seed{seed}_Cselect.npz',prediction=pred[:,vi],truth=truth[vi],clear=clear[vi],date=dates[vi],init_time=np.array([rr[i]['init_time_bjt'] for i in vi]),target_time=np.array([rr[i]['target_time_bjt'] for i in vi]),lead=leads[vi],coefficients=np.array(coefs),cot_scale=np.array([sh,sf]))
            for ar,name in enumerate(['A','H','F']):
                e=pred[ar,select]-truth[select];hh=truth[select]>=800
                all_records.append(dict(station=st,seed=seed,arm=name,n=int(select.sum()),rmse=float(np.sqrt(np.mean(e*e))),mae=float(abs(e).mean()),bias=float(e.mean()),high_ghi_bias=float(e[hh].mean()) if hh.any() else None))
                for day in sorted(set(dates[select])):
                    if day not in day_stats:day_stats[day]=np.zeros((3,3,3,2),np.float64)
                    ee=(pred[ar]-truth)[select&(dates==day)];day_stats[day][si,sj,ar]=[float(ee@ee),len(ee)]
            print(st,seed,'DONE',flush=True)
    days=sorted(day_stats);daily=np.array([day_stats[d] for d in days]);total=daily.sum(0);points=np.sqrt(total[...,0]/total[...,1]).mean((0,1));w=np.random.default_rng(20261023).multinomial(len(days),np.ones(len(days))/len(days),5000)
    boot=np.sqrt((w@daily[...,0].reshape(len(days),-1))/(w@daily[...,1].reshape(len(days),-1))).reshape(5000,3,3,3).mean((1,2));comparisons={}
    for j,k in [(2,0),(2,1),(1,0)]:
        delta=points[j]-points[k];ci=np.quantile(boot[:,j]-boot[:,k],[.025,.975]);site=[];seeds=0;high=[]
        for st in SITES:
            vals=[np.mean([r['rmse'] for r in all_records if r['station']==st and r['arm']==a]) for a in ['A','H','F']];site.append(100*(vals[j]/vals[k]-1))
        for seed in SEEDS:
            vals=[np.mean([r['rmse'] for r in all_records if r['seed']==seed and r['arm']==a]) for a in ['A','H','F']];seeds+=vals[j]<vals[k]
        for st in SITES:
            for seed in SEEDS:
                vals=[next(r['high_ghi_bias'] for r in all_records if r['station']==st and r['seed']==seed and r['arm']==a) for a in ['A','H','F']]
                if vals[j] is not None and vals[k] is not None:high.append(abs(vals[j])-abs(vals[k]))
        comparisons[['A','H','F'][j]+'_minus_'+['A','H','F'][k]]=dict(delta_rmse=float(delta),ci95=ci.tolist(),site_change_percent=site,improving_seeds=int(seeds),mean_abs_high_bias_increase=float(np.mean(high)),passes=bool(delta<0 and ci[1]<0 and max(site)<=1 and seeds>=2 and np.mean(high)<=5))
    passed=comparisons['F_minus_A']['passes'] and comparisons['F_minus_H']['passes']
    report=dict(state='CSELECT_LINEAR_SCREEN_COMPLETE',arms=dict(zip(['A','H','F'],points.tolist())),comparisons=comparisons,passes_future_cot_screen=passed,decision='freeze_then_consider_development_validation' if passed else 'close_candidate_no_validation_scoring_no_hyperparameter_search',records=all_records,dates=days,witness=witness,official_validation_labels_read=0,new_period_labels_read=0,test_payloads_read=0,buffer_payloads_read=0,upstream_weights_updated=False,ridge=.01,parameters_per_arm=27,script_sha256=sha256(Path(__file__)))
    (out/'SCREEN_RESULT.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ['state','arms','comparisons','decision']}),flush=True)
if __name__=='__main__':main()
