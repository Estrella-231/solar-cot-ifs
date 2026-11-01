"""Validation-only thick-cloud miss and motion audit; frozen saved predictions."""
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
out=exp/'sili_thick_motion_20260927_v1';out.mkdir(exist_ok=True)
cache=exp/'agri_cpp_pair_bank_v1';records=[]
for sh in json.loads((cache/'index.json').read_text())['shards']:
    d=cache/sh['directory']
    records.extend({**r,'directory':d} for r in json.loads((d/'records.json').read_text()) if r['split']=='val' and r['included_for_R'])
lookup={r['utc']:i for i,r in enumerate(records)}
ext=exp/'r_pair_audit_20260927_v1/extracted_arrays/r256_full_exploratory_seed42_batch512_node22'
truth=np.load(ext/'reference_log1p_cot.npy',mmap_mode='r');mask=np.load(ext/'valid_mask.npy',mmap_mode='r')
sc=json.loads((root/'configs/s_frozen_hunan_seed42.json').read_text())
rows=[r for r in csv.DictReader(open(sc['manifest'])) if r['split']=='val']
bank=exp/'regional_ghi_pilot_bank_20260927_v1/val'
seq=np.load(bank/'sequence_indices.npy');fc=np.load(bank/'cot_local_forecast.npy',mmap_mode='r');oc=np.load(bank/'cot_local_oracle.npy',mmap_mode='r');hist=np.load(bank/'cot_local_history.npy',mmap_mode='r')
sys.path.insert(0,'/public/home/slfu/ttzhou/swc/irradiance_forecast/HunanSimVPFull_20260906')
from hunan_data import load_frame
data=Path('/public/home/slfu/ttzhou/Auxiliary_data/HuNan_AGRI_Preprocessed')
sn=json.loads(Path(sc['normalization']).read_text());sm=np.array(sn['mean'],np.float32)[:,None,None];ss=np.array(sn['std'],np.float32)[:,None,None]
rr,cc=163,148;sl=(slice(rr-32,rr+32),slice(cc-32,cc+32))
torch.set_num_threads(4)
catalog=[];counts={k:0 for k in ['valid_station_targets','thick_reference_targets','simvp_thick_misses','oracle_thick_misses','far_thick_targets','far_simvp_thick_misses']}
def historical(rel):
    p=Path('/tmp/slfu_hunan_triad_frames_'+sc['manifest_sha256'][:16])/(rel+'.npz')
    if p.exists():
        with np.load(p) as z:return z['a'],z['v'],z['g']
    return load_frame(data,rel)
def motion_inputs(rels):
    values=[historical(t) for t in rels[-2:]]
    h=np.stack([np.where(v,(a-sm)/ss,0).astype(np.float32) for a,v,g in values])
    return motion_shift(h),values
for j,k in enumerate(seq):
    row=rows[int(k)];rels=row['data_relpaths'].split('|');stamps=[Path(t).stem for t in rels]
    shift,values=motion_inputs(rels[:8]);ref=np.full((16,64,64),np.nan,np.float32);valid=np.zeros((16,64,64),bool)
    for ld,t in enumerate(stamps[8:]):
        if t in lookup:
            ix=lookup[t];ref[ld]=np.expm1(truth[ix,0][sl]);valid[ld]=mask[ix,0][sl]
    f=np.expm1(fc[j,0,:,0]);o=np.expm1(oc[j,0,:,0]);v=valid[:,32,32];r=ref[:,32,32]
    thick=v&(r>=30);miss=thick&(f[:,32,32]<10);omiss=thick&(o[:,32,32]<10);far=np.arange(16)>=7
    for key,value in [('valid_station_targets',v.sum()),('thick_reference_targets',thick.sum()),('simvp_thick_misses',miss.sum()),('oracle_thick_misses',omiss.sum()),('far_thick_targets',(thick&far).sum()),('far_simvp_thick_misses',(miss&far).sum())]:counts[key]+=int(value)
    if not v.any():continue
    peak=int(np.nanargmax(np.where(v,r,np.nan)));fp=int(np.argmax(np.where(v,f[:,32,32],-np.inf)))
    catalog.append({'bank_index':j,'sequence':int(k),'init':(datetime.strptime(row['BJT_start'],'%Y-%m-%d %H:%M:%S')+timedelta(minutes=105)).isoformat(),'shift_row_col_per_15min':shift.tolist(),'valid_targets':int(v.sum()),'thick_targets':int(thick.sum()),'miss_targets':int(miss.sum()),'reference_peak':float(r[peak]),'simvp_at_reference_peak':float(f[peak,32,32]),'oracle_at_reference_peak':float(o[peak,32,32]),'peak_lead_min':15*(peak+1),'simvp_peak_lead_min':15*(fp+1),'simvp_peak_amplitude':float(f[fp,32,32]),'peak_underprediction':float(r[peak]-f[peak,32,32])})
    print('AUDIT',j,shift.tolist(),flush=True)
nonzero=[d for d in catalog if any(d['shift_row_col_per_15min'])]
cloudy=[d for d in nonzero if d['valid_targets']>=8 and d['reference_peak']>=30]
summary={'test_used':False,'cohort':'128 existing regional GHI validation pilot sequences; no GHI magnitude selection','counts':counts,'sequences_with_station_reference':len(catalog),'nonzero_motion_sequences':len(nonzero),'nonzero_motion_cloudy_sequences':len(cloudy),'thresholds':{'reference_thick_cot':30,'severe_miss_prediction_below':10,'far_lead_start_min':120},'scope':'CPP retrieval reference, not independent physical truth; frames are dependent; peak-time differences unreliable when cloud peak is absent; no GHI attribution'}
(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'sequence_catalog.json').write_text(json.dumps(catalog,indent=2))
selected=[('largest_thick_peak_miss',max([d for d in catalog if d['valid_targets']>=8],key=lambda d:d['peak_underprediction']))]
if cloudy:
    selected.append(('nonzero_motion_cloud',max(cloudy,key=lambda d:d['peak_underprediction'])))
    best=min(cloudy,key=lambda d:abs(d['peak_underprediction']))
    if best['sequence']!=selected[-1][1]['sequence']:selected.append(('nonzero_motion_control',best))
elif nonzero:selected.append(('nonzero_motion_no_thick_cloud',max(nonzero,key=lambda d:d['reference_peak'])))
from train_cot_repaired import COTUNet
rmodel=COTUNet().eval();rmodel.load_state_dict(torch.load(exp/'r256_full_exploratory_seed42_batch512_node22/best.pt',map_location='cpu',weights_only=False)['model'])
rn=json.loads((cache/'norm.json').read_text());rm=np.array(rn['mean'],np.float32)[:,None,None];rs=np.array(rn['std'],np.float32)[:,None,None]
for label,d in selected:
    j=d['bank_index'];rels=rows[d['sequence']]['data_relpaths'].split('|');stamps=[Path(p).stem for p in rels];shift,vals=motion_inputs(rels[:8]);a,v,g=vals[-1]
    raw=np.concatenate((np.where(v,a,np.nan),g))
    with torch.no_grad():full=rmodel(torch.from_numpy(np.nan_to_num((raw-rm)/rs,nan=0.,posinf=0.,neginf=0.))[None])[0]
    d['cpu_gpu_history_max_abs_log1p_difference']=float(np.max(abs(full.numpy()[0][sl]-hist[j,0,-1,0])))
    fields={'simvp':np.expm1(fc[j,0,:,0]),'oracle':np.expm1(oc[j,0,:,0]),'persistence':np.repeat(np.expm1(full.numpy()[0][sl])[None],16,axis=0),'transport':np.full((16,64,64),np.nan,np.float32)}
    ref=np.full((16,64,64),np.nan,np.float32);rgb=np.full((16,64,64,3),np.nan,np.float32);valid=np.zeros((16,64,64),bool)
    for ld,t in enumerate(stamps[8:]):
        moved,support=transport(full,shift,ld+1);fields['transport'][ld]=np.where(support.numpy()[sl],np.expm1(moved.numpy()[0][sl]),np.nan)
        if t not in lookup:continue
        ix=lookup[t];ref[ld]=np.expm1(truth[ix,0][sl]);valid[ld]=mask[ix,0][sl]
        rec=records[ix];x=np.load(rec['directory']/'x_raw.npy',mmap_mode='r')[rec['array_row']]
        rgb[ld]=np.moveaxis(x[[2,1,0]][:,sl[0],sl[1]],0,-1)
    good=rgb[np.isfinite(rgb).all(-1)];lo=np.percentile(good,2,axis=0);hi=np.percentile(good,98,axis=0);display=np.clip((rgb-lo)/(hi-lo+1e-8),0,1)
    d['rgb_percentile_low']=lo.tolist();d['rgb_percentile_high']=hi.tolist()
    np.savez_compressed(out/(label+'_source.npz'),reference=ref,valid=valid,rgb=rgb,**fields)
    leads=sorted(set([0,3,7,11,15,d['peak_lead_min']//15-1]))
    fig,axs=plt.subplots(6,len(leads),figsize=(3*len(leads),14),layout='constrained')
    labels=['Real AGRI RGB proxy','CPP reference','Persistence','Constant translation','SimVP -> R','Real AGRI -> R (oracle)']
    for ci,ld in enumerate(leads):
        panels=[display[ld],np.where(valid[ld],ref[ld],np.nan)]+[fields[n][ld] for n in ['persistence','transport','simvp','oracle']]
        for ri,p in enumerate(panels):
            ax=axs[ri,ci]
            if np.isfinite(p).any():im=ax.imshow(p,interpolation='nearest',**({} if ri==0 else dict(cmap='magma',vmin=0,vmax=70)))
            else:ax.text(.5,.5,'Missing',transform=ax.transAxes,ha='center')
            ax.plot(32,32,'+',color='cyan',ms=9);ax.set_xticks([]);ax.set_yticks([])
            if ci==0:ax.set_ylabel(labels[ri])
            if ri==0:ax.set_title(f'+{15*(ld+1)} min')
    fig.colorbar(im,ax=axs[1:,:],extend='max',label='COT',shrink=.5)
    fig.suptitle(f'Sili {d["init"]} BJT | shift (row,col)={shift.tolist()} pixels/15 min\n{label}; real RGB shared 2-98% stretch; native64; cyan station cross')
    fig.savefig(out/(label+'_maps.png'),dpi=130);fig.savefig(out/(label+'_maps.pdf'));plt.close(fig)
    fig,axs=plt.subplots(1,5,figsize=(16,4),layout='constrained')
    for ax,n in zip(axs,['CPP','persistence','transport','simvp','oracle']):
        z=np.where(valid[:,32,:],ref[:,32,:],np.nan) if n=='CPP' else fields[n][:,32,:]
        im=ax.imshow(z,origin='lower',aspect='auto',extent=[-32.5,31.5,7.5,247.5],vmin=0,vmax=70,cmap='magma',interpolation='nearest');ax.axvline(0,color='cyan',lw=1);ax.set(title=n,xlabel='Station column offset (pixels)',ylabel='Lead (min)')
    fig.colorbar(im,ax=axs,label='COT',extend='max');fig.suptitle('Sili station-row space-time transect: '+label)
    fig.savefig(out/(label+'_spacetime.png'),dpi=160);fig.savefig(out/(label+'_spacetime.pdf'));plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4),layout='constrained');t=np.arange(1,17)*15
    ax.plot(t,np.where(valid[:,32,32],ref[:,32,32],np.nan),'k-o',label='CPP reference')
    for n,c in zip(['persistence','transport','simvp','oracle'],['#666666','#D55E00','#0072B2','#009E73']):ax.plot(t,fields[n][:,32,32],label=n,color=c,marker='.',linestyle='--' if n=='oracle' else '-')
    ax.axhline(30,color='gray',ls=':',lw=1);ax.set(xlabel='Lead (min)',ylabel='Station-pixel COT',title=f'Sili {d["init"]} BJT | shift={shift.tolist()}');ax.legend(ncol=3);ax.grid(alpha=.2)
    fig.savefig(out/(label+'_trajectory.png'),dpi=160);plt.close(fig)
(out/'selected_cases.json').write_text(json.dumps([{'label':l,**d} for l,d in selected],indent=2))
print('COMPLETE',json.dumps(summary),flush=True)
