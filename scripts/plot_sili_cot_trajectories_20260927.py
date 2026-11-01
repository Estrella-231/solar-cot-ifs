"""CPU diagnostics of saved validation COT fields; no training or test use."""
import csv,json,sys
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_regional_cot_ghi_pilot_20260927 import motion_shift,transport

root=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907')
exp=root/'experiments/regional_station_cot_20260925'
out=exp/'sili_trajectory_diagnostic_20260927_v2';out.mkdir(exist_ok=True)
cache=exp/'agri_cpp_pair_bank_v1'
records=[]
for sh in json.loads((cache/'index.json').read_text())['shards']:
    d=cache/sh['directory']
    records.extend({**r,'directory':d} for r in json.loads((d/'records.json').read_text()) if r['split']=='val' and r['included_for_R'])
lookup={r['utc']:i for i,r in enumerate(records)}
extract=exp/'r_pair_audit_20260927_v1/extracted_arrays/r256_full_exploratory_seed42_batch512_node22'
pred=np.load(extract/'pred_log1p_cot.npy',mmap_mode='r')
truth=np.load(extract/'reference_log1p_cot.npy',mmap_mode='r')
mask=np.load(extract/'valid_mask.npy',mmap_mode='r')
sc=json.loads((root/'configs/s_frozen_hunan_seed42.json').read_text())
rows=list(csv.DictReader(open(sc['manifest'])));rows=[r for r in rows if r['split']=='val']
bank=exp/'regional_ghi_pilot_bank_20260927_v1/val'
seq=np.load(bank/'sequence_indices.npy')
forecast=np.load(bank/'cot_local_forecast.npy',mmap_mode='r')
oracle=np.load(bank/'cot_local_oracle.npy',mmap_mode='r')
history=np.load(bank/'cot_local_history.npy',mmap_mode='r')
rr,cc=163,148
sl=(slice(rr-32,rr+32),slice(cc-32,cc+32))
names=['persistence','transport','simvp','oracle']
sums={n:np.zeros((16,3)) for n in names};point={n:np.zeros((16,3)) for n in names}
cases=[];drifts=[];torch.set_num_threads(4)
sys.path.insert(0,'/public/home/slfu/ttzhou/swc/irradiance_forecast/HunanSimVPFull_20260906')
from hunan_data import load_frame
from train_cot_repaired import COTUNet
retriever=COTUNet().eval()
retriever.load_state_dict(torch.load(exp/'r256_full_exploratory_seed42_batch512_node22/best.pt',map_location='cpu',weights_only=False)['model'])
rnorm=json.loads((cache/'norm.json').read_text())
rm=np.array(rnorm['mean'],np.float32)[:,None,None];rs=np.array(rnorm['std'],np.float32)[:,None,None]
data_root=Path('/public/home/slfu/ttzhou/Auxiliary_data/HuNan_AGRI_Preprocessed')
def raw(stamp):
    r=records[lookup[stamp]]
    return np.load(r['directory']/'x_raw.npy',mmap_mode='r')[r['array_row']]
for j,k in enumerate(seq):
    row=rows[int(k)];stamps=[Path(p).stem for p in row['data_relpaths'].split('|')]
    if any(t not in lookup for t in stamps[6:8]):continue
    observed=[];motion=[];sn=json.loads(Path(sc['normalization']).read_text());sm=np.array(sn['mean'],np.float32)[:,None,None];ss=np.array(sn['std'],np.float32)[:,None,None]
    for rel in row['data_relpaths'].split('|')[6:8]:
        ag,valid_ag,geo=load_frame(data_root,rel)
        observed.append(np.concatenate((np.where(valid_ag,ag,np.nan),geo)));motion.append(np.where(valid_ag,(ag-sm)/ss,0).astype(np.float32))
    shift=motion_shift(np.stack(motion))
    with torch.no_grad():full=retriever(torch.from_numpy(np.nan_to_num((observed[-1]-rm)/rs,nan=0.,posinf=0.,neginf=0.))[None])[0]
    drift=float(np.max(np.abs(full.numpy()[0][sl]-history[j,0,-1,0]))); print('HISTORY_EQUIVALENCE',j,drift,flush=True); drifts.append(drift); assert drift < .05
    fields={n:np.full((16,64,64),np.nan,np.float32) for n in names}
    reference=np.full((16,64,64),np.nan,np.float32);valid=np.zeros((16,64,64),bool)
    rgb=np.full((16,64,64,3),np.nan,np.float32)
    for ld,t in enumerate(stamps[8:]):
        adv,av=transport(full,shift,ld+1)
        fields['transport'][ld]=np.where(av.numpy()[sl],np.expm1(adv.numpy()[0][sl]),np.nan)
        fields['persistence'][ld]=np.expm1(full.numpy()[0][sl])
        fields['simvp'][ld]=np.expm1(forecast[j,0,ld,0])
        fields['oracle'][ld]=np.expm1(oracle[j,0,ld,0])
        if t not in lookup:continue
        ix=lookup[t];reference[ld]=np.expm1(truth[ix,0][sl]);valid[ld]=mask[ix,0][sl]&av.numpy()[sl]
        rgb[ld]=np.moveaxis(raw(t)[[2,1,0]][:,sl[0],sl[1]],0,-1)
        for n in names:
            e=(fields[n][ld]-reference[ld])[valid[ld]].astype(float)
            sums[n][ld]+=[len(e),np.abs(e).sum(),(e*e).sum()]
            if valid[ld,32,32]:
                z=float(fields[n][ld,32,32]-reference[ld,32,32]);point[n][ld]+=[1,abs(z),z*z]
    good=valid[:,32,32]
    if good.sum()<8:continue
    scores={n:float(np.sqrt(np.mean((fields[n][good,32,32]-reference[good,32,32])**2))) for n in names}
    cases.append(dict(j=j,sequence=int(k),init=(datetime.strptime(row['BJT_start'],'%Y-%m-%d %H:%M:%S')+timedelta(minutes=105)).isoformat(),shift=shift.tolist(),scores=scores,fields=fields,reference=reference,valid=valid,rgb=rgb))
    print('SEQUENCE',j,flush=True)
def metric(v):
    return [{'lead_minutes':15*(i+1),'n':int(a),'rmse':float(np.sqrt(c/a)) if a else None,'mae':float(b/a) if a else None} for i,(a,b,c) in enumerate(v)]
report={'test_used':False,'station':'Sili','transport_R_inference':'CPU FP32 recomputation of same R checkpoint and historical inputs; diagnostic numeric variant of GPU baseline','history_log1p_max_abs_drift':max(drifts),'station_rc':[rr,cc],'selected_sequences':len(seq),'usable_case_sequences':len(cases),'reference':'CPP retrieval reference; producer QA unresolved','mask':'target CPP validity intersect full-region transport in-domain; same for all methods','motion':'last two real C13 regional phase correlation; constant translation in physical COT','local64':{n:metric(sums[n]) for n in names},'station_pixel':{n:metric(point[n]) for n in names}}
(out/'summary.json').write_text(json.dumps(report,indent=2))
fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
colors=['#666666','#D55E00','#0072B2','#009E73']
for ax,key,title in zip(axs,['station_pixel','local64'],['Sili station pixel','Sili native 64 x 64 neighbourhood']):
    for n,c in zip(names,colors):ax.plot(np.arange(1,17)*15,[x['rmse'] for x in report[key][n]],label=n,color=c,marker='o',ms=3)
    ax.set(title=title,xlabel='Forecast lead (min)',ylabel='COT RMSE vs CPP');ax.legend();ax.grid(alpha=.2)
fig.savefig(out/'sili_lead_rmse.png',dpi=160);fig.savefig(out/'sili_lead_rmse.pdf');plt.close(fig)
rank=sorted(cases,key=lambda x:x['scores']['simvp']-x['scores']['transport'])
selected=[('simvp_better',rank[0]),('simvp_worse',rank[-1]),('typical',rank[len(rank)//2])]
metadata=[]
for label,d in selected:
    fields=d['fields'];ref=d['reference'];va=d['valid'];rgb=d['rgb'];leads=[0,3,7,11,15]
    finite=rgb[np.isfinite(rgb).all(-1)];lo=np.percentile(finite,2,axis=0);hi=np.percentile(finite,98,axis=0)
    rgb=np.clip((rgb-lo)/(hi-lo+1e-8),0,1)
    fig,axs=plt.subplots(6,5,figsize=(13,14),layout='constrained')
    for col,ld in enumerate(leads):
        panels=[rgb[ld],np.where(va[ld],ref[ld],np.nan)]+[fields[n][ld] for n in names]
        for ri,p in enumerate(panels):
            ax=axs[ri,col]
            if np.isfinite(p).any():im=ax.imshow(p,**({} if ri==0 else dict(vmin=0,vmax=60,cmap='magma')),interpolation='nearest')
            else:ax.text(.5,.5,'Missing reference image',ha='center',transform=ax.transAxes)
            ax.plot(32,32,'+',color='cyan',ms=9,mew=1.5);ax.set_xticks([]);ax.set_yticks([])
            if ri==0:ax.set_title(f'+{15*(ld+1)} min')
            if col==0:ax.set_ylabel(['Real AGRI RGB proxy','CPP reference','Persistence','Translation','SimVP -> R','Real AGRI -> R (oracle)'][ri])
    fig.colorbar(im,ax=axs[1:,:],label='COT (dimensionless)',extend='max',shrink=.5)
    fig.suptitle(f'Sili {d["init"]} BJT | {label}\nNative 64x64; shared real RGB 2-98% stretch; cyan cross = station')
    fig.savefig(out/(label+'_maps.png'),dpi=130);fig.savefig(out/(label+'_maps.pdf'));plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(12,7),layout='constrained')
    t=np.arange(1,17)*15
    axs[0].plot(t,np.where(va[:,32,32],ref[:,32,32],np.nan),'k-o',label='CPP reference')
    for n,c in zip(names,colors):axs[0].plot(t,fields[n][:,32,32],color=c,label=n,marker='.',linestyle='--' if n=='oracle' else '-')
    axs[0].set(xlabel='Lead (min)',ylabel='COT at station pixel',title=f'Sili {d["init"]} BJT: station COT trajectory');axs[0].legend(ncol=3);axs[0].grid(alpha=.2)
    # Horizontal station transect, one native-pixel row; no spatial averaging.
    axs[1].axis('off');plt.close(fig)
    fig,axs=plt.subplots(1,5,figsize=(15,4),layout='constrained')
    for ax,n in zip(axs,['CPP']+names):
        z=np.where(va[:,32,:],ref[:,32,:],np.nan) if n=='CPP' else fields[n][:,32,:]
        im=ax.imshow(z,origin='lower',aspect='auto',extent=[-32.5,31.5,7.5,247.5],vmin=0,vmax=60,cmap='magma',interpolation='nearest')
        ax.axvline(0,color='cyan',lw=1);ax.set(title=n,xlabel='Column offset from station (pixels)',ylabel='Forecast lead (min)')
    fig.colorbar(im,ax=axs,label='COT',extend='max');fig.suptitle('Sili station-row space-time transect (no spatial averaging)')
    fig.savefig(out/(label+'_spacetime.png'),dpi=160);fig.savefig(out/(label+'_spacetime.pdf'));plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    ax.plot(t,np.where(va[:,32,32],ref[:,32,32],np.nan),'k-o',label='CPP reference')
    for n,c in zip(names,colors):ax.plot(t,fields[n][:,32,32],color=c,label=n,marker='.',linestyle='--' if n=='oracle' else '-')
    ax.set(xlabel='Forecast lead (min)',ylabel='Station-pixel COT',title=f'Sili {d["init"]} BJT');ax.legend(ncol=3);ax.grid(alpha=.2)
    fig.savefig(out/(label+'_trajectory.png'),dpi=160);plt.close(fig)
    np.savez_compressed(out/(label+'_source.npz'),reference=ref,valid=va,rgb= d['rgb'],**fields)
    metadata.append({k:d[k] for k in ['sequence','init','shift','scores']}|{'selection':label,'rgb_stretch_low':lo.tolist(),'rgb_stretch_high':hi.tolist()})
(out/'cases.json').write_text(json.dumps(metadata,indent=2));print('COMPLETE',json.dumps(metadata),flush=True)






