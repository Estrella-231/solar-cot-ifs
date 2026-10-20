"""Frozen validation-input interventions; no training and no test access."""
import csv,json,hashlib,sys,importlib.util
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
import torch

ST={'Niushou':(8,6),'Wangzhai':(9,8),'Mizhuang':(7,10)}
def loadmod(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def met(y,p):
    e=np.asarray(p,np.float64)-np.asarray(y,np.float64)
    return dict(n=int(e.size),rmse=float(np.sqrt(np.mean(e*e))),mae=float(np.mean(abs(e))),bias=float(e.mean()))
def csvout(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    base=Path(sys.argv[1]);proj=base.parent.parent;out=base/'diagnosis_v1'/'output_attempt2';out.mkdir(exist_ok=False)
    inp=base/'diagnosis_v1/forecast/validation';run=base/'runtime_attempt2';pack=run/'forecast_pack_attempt1';rroot=proj/'experiments/hubei_cloud_probability_20261020'
    sys.path.insert(0,str(base/'code_attempt2'));head=loadmod('head',base/'code_attempt2/train_paired_forecast.py');rmod=loadmod('rmodel',base/'code_attempt2/train_cot_repaired.py')
    dev=torch.device('cuda');torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    keys=list(csv.DictReader((inp/'keys.csv').open()));assert len(keys)==407
    obs=np.load(inp/'true_agri_patch.npy',allow_pickle=False);pred=np.load(inp/'pred_agri_patch.npy',allow_pickle=False);geo=np.load(inp/'future_geometry_patch.npy',allow_pickle=False)
    assert obs.shape==pred.shape==(407,16,13,16,16) and np.isfinite(obs).all() and np.isfinite(pred).all()
    for k in keys:
        dt=datetime.strptime(k['init_time_bjt'],'%Y%m%d%H%M');assert k['target_times_bjt'].split('|')==[(dt+timedelta(minutes=15*i)).strftime('%Y%m%d%H%M') for i in range(1,17)]
    norm=json.loads((rroot/'R_pack_correctutc_attempt2/norm.json').read_text());mean=np.asarray(norm['mean'],np.float32)[None,:,None,None];std=np.asarray(norm['std'],np.float32)[None,:,None,None]
    rm=rmod.COTUNet().to(dev).eval();rm.load_state_dict(torch.load(rroot/'R_correctutc_attempt3/best.pt',map_location=dev)['model'])
    def invert(a):
        raw=np.concatenate((a,geo),axis=2).reshape(-1,16,16,16);ans=[]
        for j in range(0,len(raw),2048):
            x=raw[j:j+2048];day=x[:,15:16]>.5;z=(x-mean)/std;z[:,15:16]=np.where(day,z[:,15:16],0.)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):v=rm(torch.from_numpy(z).to(dev)).float()
            ans.append(torch.expm1(v).masked_fill(~torch.from_numpy(day).to(dev),0.).cpu().numpy())
        return np.concatenate(ans).reshape(407,16,1,16,16)
    co=invert(obs);cp=invert(pred);bank=np.load(pack/'cot_banks/validation.npy',allow_pickle=False)
    cot_parity=float(abs(cp-bank).max());assert cot_parity<.05,('COT replay',cot_parity)
    # Validate physical units and coordinates against separately packed real-image R inputs.
    rr=list(csv.DictReader((rroot/'R_pack_correctutc_attempt2/rows.csv').open()));rx=np.load(rroot/'R_pack_correctutc_attempt2/x_norm.npy',mmap_mode='r');rt=np.load(rroot/'R_pack_correctutc_attempt2/target.npy',mmap_mode='r');mask=np.load(rroot/'R_pack_correctutc_attempt2/mask.npy',mmap_mode='r')
    ri={r['time_bjt']:i for i,r in enumerate(rr) if r['split']=='val'};reference=np.full_like(co,np.nan);valid=np.zeros_like(co,bool);ag_checks=[]
    for i,k in enumerate(keys):
        for l,t in enumerate(k['target_times_bjt'].split('|')):
            if t not in ri:continue
            j=ri[t];back=np.asarray(rx[j])*std[0]+mean[0];ag_checks.append(float(abs(back[:13]-obs[i,l]).max()))
            reference[i,l]=100.0*rt[j];valid[i,l]=mask[j]
    assert ag_checks and max(ag_checks)<1e-3,('observed physical unit/grid mismatch',max(ag_checks))
    rows0=list(csv.DictReader((pack/'rows.csv').open()));ids=[i for i,r in enumerate(rows0) if r['split']=='val'];rows=[rows0[i] for i in ids];N=len(rows);assert N==6723
    lookup={k['init_time_bjt']:i for i,k in enumerate(keys)};seq=np.array([lookup[r['init_time_bjt']] for r in rows]);lead=np.array([int(r['lead_minutes'])//15-1 for r in rows]);Y=np.load(pack/'ghi_wm2.npy')[ids];clear=np.load(pack/'clear_sky_wm2.npy')[ids]
    agp=np.empty((N,13,3,3),np.float32);ago=np.empty_like(agp);g=np.empty((N,2,3,3),np.float32);cop=np.empty((N,1,3,3),np.float32);cpp=np.empty_like(cop)
    for j,r in enumerate(rows):
        y,x=ST[r['station']];s,l=seq[j],lead[j];sl=(slice(y-1,y+2),slice(x-1,x+2));agp[j]=pred[s,l][:,sl[0],sl[1]];ago[j]=obs[s,l][:,sl[0],sl[1]];g[j]=geo[s,l][:2,sl[0],sl[1]];cop[j]=co[s,l][:,sl[0],sl[1]];cpp[j]=cp[s,l][:,sl[0],sl[1]]
    assert np.max(abs(agp-np.load(pack/'agri_raw.npy')[ids]))<1e-6 and np.max(abs(g-np.load(pack/'geometry.npy')[ids]))<1e-6
    modes=('FA_pred','FAC_pred','FAC_observedCOT','FAC_zeroCOTslot','FAC_permutedCOT','FA_observedAGRI','FAC_observedAGRI_predCOT','FAC_observedAGRI_observedCOT')
    P={m:np.empty((3,N),np.float64) for m in modes};metrics=[];parity=[];hashes={};rng=np.random.default_rng(20261020)
    for st in ST:
        take=np.array([j for j,r in enumerate(rows) if r['station']==st]);perm=np.arange(len(take))
        for l in range(16):
            ix=np.flatnonzero(lead[take]==l);perm[ix]=rng.permutation(ix)
        for si,seed in enumerate((42,43,44)):
            for arm in ('FA','FAC'):
                path=run/f'paired_seed{seed}_attempt1'/st/f'{arm}_best.pt';ck=torch.load(path,map_location=dev);z=ck['station_normalization'];m=head.HubeiLocalGHIHead(arm).to(dev).eval();m.load_state_dict(ck['model']);hashes[str(path)]=sha(path)
                def get(a,c,zero=False):
                    x=(a-np.array(z['agri_mean'],np.float32)[None,:,None,None])/np.array(z['agri_std'],np.float32)[None,:,None,None]
                    x=np.concatenate((x,g[take]),axis=1).astype(np.float32);c=((np.log1p(c)-z['cot_log1p_mean'])/z['cot_log1p_std']).astype(np.float32)
                    if zero:c=np.zeros_like(c)
                    with torch.inference_mode():return np.concatenate([(m(torch.from_numpy(x[j:j+512]).to(dev),torch.from_numpy(c[j:j+512]).to(dev))*torch.from_numpy(clear[take][j:j+512]).to(dev)).cpu().numpy() for j in range(0,len(take),512)]).astype(np.float64)
                if arm=='FA':settings=[('FA_pred',agp[take],cpp[take],False),('FA_observedAGRI',ago[take],cop[take],False)]
                else:settings=[('FAC_pred',agp[take],cpp[take],False),('FAC_observedCOT',agp[take],cop[take],False),('FAC_zeroCOTslot',agp[take],cpp[take],True),('FAC_permutedCOT',agp[take],cpp[take][perm],False),('FAC_observedAGRI_predCOT',ago[take],cpp[take],False),('FAC_observedAGRI_observedCOT',ago[take],cop[take],False)]
                for name,a,c,zero in settings:
                    p=get(a,c,zero);P[name][si,take]=p;metrics.append(dict(station=st,seed=seed,mode=name,**met(Y[take],p)))
                    if name in ('FA_pred','FAC_pred'):
                        with np.load(path.parent/f'{arm}_validation_predictions_with_keys.npz',allow_pickle=False) as b:
                            assert np.array_equal(b['truth_ghi_wm2'],Y[take]);delta=float(abs(p-b['pred_ghi_wm2']).max());assert delta<.01,('GHI replay',delta);parity.append(delta)
    csvout(out/'head_interventions.csv',metrics)
    np.savez_compressed(out/'paired_predictions.npz',truth=Y,clear=clear,station=np.array([r['station'] for r in rows]),init=np.array([r['init_time_bjt'] for r in rows]),target=np.array([r['target_time_bjt'] for r in rows]),lead_minutes=(lead+1)*15,observed_cot=cop,predicted_cot=cpp,**P)
    csvout(out/'rows.csv',rows)
    cm=[];seqrec=[]
    for st,(y,x) in ST.items():
        daylight=geo[:,:,2,y,x]>.5;a=co[:,:,0,y,x];b=cp[:,:,0,y,x]
        for l in range(16):
            take=daylight[:,l];d=met(a[take,l],b[take,l]);cm.append(dict(station=st,lead_minutes=(l+1)*15,comparison='predR_vs_observedR_station',**d))
            v=valid[:,l,0,y,x]&daylight[:,l]
            if v.any():
                for name,c in [('predR_vs_CPP',b),('observedR_vs_CPP',a)]:cm.append(dict(station=st,lead_minutes=(l+1)*15,comparison=name,**met(reference[v,l,0,y,x],c[v,l])))
        for i,k in enumerate(keys):
            good=daylight[i];ls=[j for j,r in enumerate(rows) if r['station']==st and seq[j]==i]
            if good.sum()<12 or len(ls)<12:continue
            aa=a[i,good];bb=b[i,good];lag_scores=[]
            for lag in range(-4,5):
                t=np.arange(16);u=t+lag;ok=(u>=0)&(u<16);t=t[ok];u=u[ok];ok=good[t]&good[u]
                lag_scores.append((float(np.mean((np.log1p(a[i,t[ok]])-np.log1p(b[i,u[ok]]))**2)),lag))
            e0=float(np.mean((np.log1p(aa)-np.log1p(bb))**2));best,lag=min(lag_scores)
            seqrec.append(dict(station=st,init=k['init_time_bjt'],sequence_index=i,n=len(ls),cot_rmse=float(np.sqrt(np.mean((aa-bb)**2))),observed_range=float(np.ptp(aa)),observed_peak=float(aa.max()),predicted_peak=float(bb.max()),observed_std=float(aa.std()),predicted_std=float(bb.std()),oracle_lag_minutes=15*lag,log_mse=e0,oracle_lag_log_mse=best,FA_rmse=met(Y[ls],P['FA_pred'][:,ls].mean(0))['rmse'],FAC_rmse=met(Y[ls],P['FAC_pred'][:,ls].mean(0))['rmse'],FAC_observedCOT_rmse=met(Y[ls],P['FAC_observedCOT'][:,ls].mean(0))['rmse']))
    csvout(out/'cot_metrics.csv',cm);csvout(out/'sequence_diagnostics.csv',seqrec)
    # Fixed, auditable case selection: per station worst COT error and median-error active-cloud event; no GHI-gain selection.
    cases=[]
    for st in ST:
        candidates=[r for r in seqrec if r['station']==st];active=[r for r in candidates if r['observed_range']>=3]
        if not active:active=candidates
        worst=max(active,key=lambda r:(r['cot_rmse'],r['init']));cases.append(dict(category='largest_COT_error',**worst))
        other=[r for r in active if r['init'][:8]!=worst['init'][:8]] or active;other=sorted(other,key=lambda r:(r['cot_rmse'],r['init']));cases.append(dict(category='median_active_event',**other[len(other)//2]))
    (out/'selected_cases.json').write_text(json.dumps(cases,indent=2));chosen=sorted({r['sequence_index'] for r in cases})
    np.savez_compressed(out/'case_maps.npz',sequence_index=np.array(chosen),observed_agri=obs[chosen],predicted_agri=pred[chosen],observed_cot=co[chosen],predicted_cot=cp[chosen],geometry=geo[chosen],cpp=reference[chosen],cpp_valid=valid[chosen])
    dates=sorted({r['target_time_bjt'][:8] for r in rows});ci={};rng=np.random.default_rng(20261020)
    groups=[dates[j:j+2] for j in range(0,len(dates),2)];weights=np.zeros((5000,len(dates)))
    draw=rng.integers(0,len(groups),(5000,len(groups)))
    for j,gr in enumerate(groups):
        for d in gr:weights[:,dates.index(d)]=(draw==j).sum(1)
    D=np.array([r['target_time_bjt'][:8] for r in rows]);sc={}
    for name in modes:
        sn=[];ss=[]
        for st in ST:
            take=np.array([r['station']==st for r in rows]);sn.extend([[np.sum(take&(D==d)) for d in dates]]*3)
            for si in range(3):ss.append([np.sum((P[name][si,take&(D==d)]-Y[take&(D==d)])**2) for d in dates])
        counts=weights@np.array(sn).T;sc[name]=np.sqrt((weights@np.array(ss).T)/counts).mean(1)
    for name,baseline in [('FAC_observedCOT','FAC_pred'),('FAC_zeroCOTslot','FAC_pred'),('FAC_permutedCOT','FAC_pred'),('FAC_observedAGRI_observedCOT','FAC_pred'),('FA_observedAGRI','FA_pred')]:
        delta=sc[name]-sc[baseline];ci[name+'_minus_'+baseline]=np.quantile(delta,[.025,.975]).tolist()
    means={m:float(np.mean([r['rmse'] for r in metrics if r['mode']==m])) for m in modes}
    result=dict(state='FROZEN_INPUT_ATTRIBUTION_COMPLETE',scope='validation-only offline diagnosis, interventions not deployable forecasts; observedR is not independent COT truth',rows=N,sequences=407,target_dates=dates,equal_station_seed_rmse=means,paired_two_day_ci95=ci,cot_replay_max_abs=cot_parity,ghi_replay_max_abs=max(parity),observed_R_input_reconstruction_max_abs=max(ag_checks),checkpoint_sha256=hashes,true_agri_sha256=sha(inp/'true_agri_patch.npy'),pred_agri_sha256=sha(inp/'pred_agri_patch.npy'),test_payloads_read=0,buffer_payloads_read=0)
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
