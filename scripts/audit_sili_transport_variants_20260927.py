"""Compare causal AGRI motion estimators as COT transport baselines on fixed val bank."""
import csv,json,sys
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.nn import functional as F

ROOT=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907')
EXP=ROOT/'experiments/regional_station_cot_20260925'; OUT=EXP/'sili_transport_variants_20260927_v2';OUT.mkdir(exist_ok=True)
BANK=EXP/'regional_ghi_pilot_bank_20260927_v1/val'; CACHE=EXP/'agri_cpp_pair_bank_v1'
EXT=EXP/'r_pair_audit_20260927_v1/extracted_arrays/r256_full_exploratory_seed42_batch512_node22'
station=(163,148); rr,cc=station; crop=(slice(rr-32,rr+32),slice(cc-32,cc+32))
seq=np.load(BANK/'sequence_indices.npy'); history_cot=np.load(BANK/'cot_local_history.npy',mmap_mode='r')
future_simvp=np.load(BANK/'cot_local_forecast.npy',mmap_mode='r'); future_oracle=np.load(BANK/'cot_local_oracle.npy',mmap_mode='r')
truth=np.load(EXT/'reference_log1p_cot.npy',mmap_mode='r'); valid_ref=np.load(EXT/'valid_mask.npy',mmap_mode='r')
sc=json.loads((ROOT/'configs/s_frozen_hunan_seed42.json').read_text()); rows=[r for r in csv.DictReader(open(sc['manifest'])) if r['split']=='val']
lookup={}
for sh in json.loads((CACHE/'index.json').read_text())['shards']:
    d=CACHE/sh['directory']
    for r in json.loads((d/'records.json').read_text()):
        if r['split']=='val' and r['included_for_R']: lookup[r['utc']]=(d,int(r['array_row']))
sys.path.insert(0,'/public/home/slfu/ttzhou/swc/irradiance_forecast/HunanSimVPFull_20260906')
from hunan_data import load_frame
data=Path('/public/home/slfu/ttzhou/Auxiliary_data/HuNan_AGRI_Preprocessed')
norm=json.loads(Path(sc['normalization']).read_text()); means=np.array(norm['mean'],np.float32);stds=np.array(norm['std'],np.float32)
torch.set_num_threads(3)

def phase(a,b):
    """Windowed subpixel phase correlation; return row/col pixels per 15 min and peak quality."""
    a=np.nan_to_num(a.astype(np.float64));b=np.nan_to_num(b.astype(np.float64))
    a=(a-a.mean())*np.outer(np.hanning(a.shape[0]),np.hanning(a.shape[1]))
    b=(b-b.mean())*np.outer(np.hanning(b.shape[0]),np.hanning(b.shape[1]))
    cross=np.fft.fft2(b)*np.conj(np.fft.fft2(a));cross/=np.maximum(np.abs(cross),1e-12)
    corr=np.fft.ifft2(cross).real;pr,pc=np.unravel_index(corr.argmax(),corr.shape)
    def sub(x0,xm,xp):
        den=xm-2*x0+xp
        return .5*(xm-xp)/den if abs(den)>1e-12 else 0.
    dy=pr+sub(corr[pr,pc],corr[(pr-1)%a.shape[0],pc],corr[(pr+1)%a.shape[0],pc])
    dx=pc+sub(corr[pr,pc],corr[pr,(pc-1)%a.shape[1]],corr[pr,(pc+1)%a.shape[1]])
    if dy>a.shape[0]/2:dy-=a.shape[0]
    if dx>a.shape[1]/2:dx-=a.shape[1]
    peak=float(corr[pr,pc]/(np.sum(np.abs(corr))+1e-12))
    return np.array([dy,dx]),peak

def transport_crop(field,shift,lead):
    h,w=field.shape
    yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
    sy=yy-lead*float(shift[0]);sx=xx-lead*float(shift[1])
    grid=torch.stack((2*sx/(w-1)-1,2*sy/(h-1)-1),-1).float()[None]
    moved=F.grid_sample(torch.expm1(field[None,None]),grid,mode='bilinear',padding_mode='zeros',align_corners=True)[0,0]
    inside=(sy>=0)&(sy<=h-1)&(sx>=0)&(sx<=w-1)
    return moved.numpy(),inside.numpy()

def get_history(rels):
    imgs=[]
    for rel in rels:
        cache_file=Path('/tmp/slfu_hunan_triad_frames_'+sc['manifest_sha256'][:16])/(rel+'.npz')
        if cache_file.exists():
            with np.load(cache_file) as z:a,v=z['a'],z['v']
        else:a,v,_=load_frame(data,rel)
        c=a[10] # C13 is channel index 10 in the 13 AGRI channels.
        imgs.append(np.where(v[10] if v.ndim==3 else v,(c-means[10])/stds[10],0).astype(np.float32))
    return np.stack(imgs)

names=['persistence','global_last1','global_med4','global_med7','local64_med7','local32_med7','local16_med7','simvp']
stats={n:np.zeros((16,4),np.float64) for n in names}; site={n:np.zeros((16,4),np.float64) for n in names};common_coverage=np.zeros((16,2),np.float64)
motions=[];cases=[]
for j,k in enumerate(seq):
    row=rows[int(k)];rels=row['data_relpaths'].split('|'); stamps=[Path(x).stem for x in rels]
    hist=get_history(rels[:8]);pair_global=[];pair_local={64:[],32:[],16:[]};qualities=[]
    for t in range(1,8):
        d,q=phase(hist[t-1],hist[t]);pair_global.append(d);qualities.append(q)
        for size in pair_local:
            half=size//2;sl=(slice(rr-half,rr+half),slice(cc-half,cc+half))
            dl,ql=phase(hist[t-1][sl],hist[t][sl]);pair_local[size].append(dl)
    pg=np.asarray(pair_global); motions.append({'sequence':int(k),'init':(datetime.strptime(row['BJT_start'],'%Y-%m-%d %H:%M:%S')+timedelta(minutes=105)).isoformat(),'last1':pg[-1].tolist(),'med4':np.median(pg[-4:],axis=0).tolist(),'med7':np.median(pg,axis=0).tolist(),'local64':np.median(pair_local[64],axis=0).tolist(),'local32':np.median(pair_local[32],axis=0).tolist(),'local16':np.median(pair_local[16],axis=0).tolist(),'integer_zero_last1':bool(np.all(np.rint(pg[-1])==0)),'float_zero_last1':bool(np.linalg.norm(pg[-1])<.1),'mean_phase_peak':float(np.mean(qualities))})
    vectors={'global_last1':pg[-1],'global_med4':np.median(pg[-4:],axis=0),'global_med7':np.median(pg,axis=0),'local64_med7':np.median(pair_local[64],axis=0),'local32_med7':np.median(pair_local[32],axis=0),'local16_med7':np.median(pair_local[16],axis=0)}
    base=np.array(history_cot[j,0,-1,0]); pred={n:np.repeat(np.expm1(base)[None],16,axis=0) for n in names if n.startswith('global_') or n.startswith('local') or n=='persistence'}
    support={n:np.ones((16,64,64),bool) for n in pred}
    for n,d in vectors.items():
        for lead in range(16):
            moved,inside=transport_crop(torch.tensor(base),d,lead+1);pred[n][lead]=moved;support[n][lead]=inside
    pred['persistence']=np.repeat(np.expm1(base)[None],16,axis=0)
    pred['simvp']=np.expm1(future_simvp[j,0,:,0])
    oracle=np.expm1(future_oracle[j,0,:,0])
    ref=np.full((16,64,64),np.nan,np.float32);mask=np.zeros((16,64,64),bool)
    for ld,t in enumerate(stamps[8:]):
        if t not in lookup:continue
        dd,ar=lookup[t];ref[ld]=np.load(dd/'cot.npy',mmap_mode='r')[ar][crop];mask[ld]=np.load(dd/'mask.npy',mmap_mode='r')[ar][crop]
    # Each transport method is scored against persistence on its identical in-domain target pixels.
    for ld in range(16):
        common=mask[ld].copy()
        for supported in support.values():common &= supported[ld]
        common_coverage[ld]+=[int(common.sum()),int(mask[ld].sum())]
        for n in names:
            if n=='oracle':continue
            if common.any():
                e=(pred[n][ld]-ref[ld])[common].astype(np.float64);stats[n][ld]+=[e.size,np.abs(e).sum(),np.square(e).sum(),common.sum()]
            if common[32,32]:
                z=float(pred[n][ld,32,32]-ref[ld,32,32]);site[n][ld]+=[1,abs(z),z*z,1]
        # Oracle reference is only a retrieval ceiling diagnostic on CPP-valid site pixel.
        if mask[ld,32,32]:
            z=float(oracle[ld,32,32]-ref[ld,32,32]);site.setdefault('oracle',np.zeros((16,4)))[ld]+=[1,abs(z),z*z,1]
    # Add consistent event catalog from the saved SimVP and oracle station tracks.
    sv=mask[:,32,32];rv=ref[:,32,32];sp=pred['simvp'][:,32,32];op=oracle[:,32,32]
    if sv.any():
        peak=int(np.nanargmax(np.where(sv,rv,np.nan))); thick=sv&(rv>=30)
        cases.append({'bank_index':j,'sequence':int(k),'init':motions[-1]['init'],'motion':motions[-1],'valid_leads':int(sv.sum()),'thick_leads':int(thick.sum()),'thick_miss_leads':int((thick&(sp<10)).sum()),'cpp_peak_lead_min':15*(peak+1),'cpp_peak':float(rv[peak]),'simvp_at_cpp_peak':float(sp[peak]),'oracle_at_cpp_peak':float(op[peak]),'simvp_peak':float(np.max(sp[sv])),'site_rmse_simvp':float(np.sqrt(np.mean((sp[sv]-rv[sv])**2)))})
    print('SEQUENCE',j,flush=True)

def pack(arr):
    out=[]
    for ld,(n,mae,sq,_) in enumerate(arr):out.append({'lead_min':15*(ld+1),'n':int(n),'mae':float(mae/n) if n else None,'rmse':float(np.sqrt(sq/n)) if n else None})
    return out

coverage_records=[{'lead_min':15*(i+1),'common_pixels':int(a),'reference_valid_pixels':int(b),'fraction':float(a/b) if b else None} for i,(a,b) in enumerate(common_coverage)]
out_summary={'test_used':False,'station':'Sili','station_pixel_rc':[rr,cc],'cohort':'fixed BJT-month stratified 128-sequence validation bank; CPP availability not used in selection','sequences_evaluated':len(motions),'motion_zero_rates':{'last_pair_integer_round_zero':float(np.mean([m['integer_zero_last1'] for m in motions])),'last_pair_subpixel_norm_lt_0.1':float(np.mean([m['float_zero_last1'] for m in motions])),'last_pair_axis_median_abs':np.median(np.abs([m['last1'] for m in motions]),axis=0).tolist(),'global_med7_zero_vector_fraction':float(np.mean([np.linalg.norm(m['med7'])<.1 for m in motions])),'local64_med7_zero_vector_fraction':float(np.mean([np.linalg.norm(m['local64'])<.1 for m in motions]))},'mean_phase_peak':float(np.mean([m['mean_phase_peak'] for m in motions])),'common_support_coverage':coverage_records,'station_pixel_metrics':{n:pack(v) for n,v in site.items()},'neighborhood_metrics':{n:pack(v) for n,v in stats.items()},'sequence_event_counts':{'with_cpp':len(cases),'thick_reference_leads':sum(x['thick_leads'] for x in cases),'severe_thick_miss_leads':sum(x['thick_miss_leads'] for x in cases),'nonzero_global_med7':int(sum(np.linalg.norm(m['med7'])>=.1 for m in motions)),'nonzero_local64_med7':int(sum(np.linalg.norm(m['local64'])>=.1 for m in motions))},'motion_methods':{'global_last1':'full-region C13 phase correlation on latest pair','global_med4':'median of latest four adjacent 15min displacement estimates','global_med7':'median of seven adjacent history pair estimates','local64_med7':'median seven pairwise phase estimates in station-centred 64x64','local32_med7':'ditto 32x32','local16_med7':'ditto 16x16'},'limitations':'Only 128 fixed pilot sequences; reference is CPP retrieval; frames within sequence are dependent; motion is constant-velocity translation; no training/GHI causal claim. All methods share the same CPP and intersection of in-domain support.'}
(OUT/'summary.json').write_text(json.dumps(out_summary,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));(OUT/'motion_vectors.json').write_text(json.dumps(motions,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));(OUT/'sequence_events.json').write_text(json.dumps(cases,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
fig,axs=plt.subplots(1,3,figsize=(15,4),layout='constrained')
for ax,key,title in zip(axs,['last1','med7','local64'],['Last pair, full region','7-pair median, full region','7-pair median, Sili 64x64']):
    z=np.array([m[key] for m in motions]);ax.scatter(z[:,1],z[:,0],s=16,alpha=.55);ax.axhline(0,color='gray',lw=.7);ax.axvline(0,color='gray',lw=.7);ax.set(title=title,xlabel='Column displacement / 15 min (pixels)',ylabel='Row displacement / 15 min (pixels)')
fig.savefig(OUT/'motion_zero_audit.png',dpi=160);fig.savefig(OUT/'motion_zero_audit.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
for n,color in zip(['persistence','global_last1','global_med4','global_med7','local64_med7','local32_med7','local16_med7','simvp'],['#666666','#D55E00','#CC79A7','#E69F00','#009E73','#56B4E9','#0072B2','#000000']):
    z=out_summary['station_pixel_metrics'].get(n)
    if z:ax.plot([r['lead_min'] for r in z],[r['rmse'] for r in z],label=n,marker='.',color=color)
ax.set(xlabel='Lead (min)',ylabel='Station-pixel COT RMSE vs CPP',title='Sili: multi-frame and local motion transport');ax.legend(ncol=2);ax.grid(alpha=.2)
fig.savefig(OUT/'transport_rmse.png',dpi=160);fig.savefig(OUT/'transport_rmse.pdf');plt.close(fig)
print('COMPLETE',json.dumps(out_summary,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),flush=True)
