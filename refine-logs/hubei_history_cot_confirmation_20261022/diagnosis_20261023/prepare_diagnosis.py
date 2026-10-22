"""Read-only alignment, fixed-head zero-COT diagnostic, and physical case export."""
import json,csv,sys,hashlib,time
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
import torch
ROOT=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments')
E=ROOT/'hubei_history_cot_confirmation_20261022'; B=ROOT/'hubei_solar_geometry_repair_20261021'; H=ROOT/'hubei_forecast_cot_increment_20261022'
O=E/'diagnosis_20261023'; O.mkdir(exist_ok=True)
sys.path.insert(0,str(E/'code_attempt3'))
from infer_new_period import geom,stamp,AGRI,sha
from history_base import HistoryCOTGHIHead,CENTERS
sel=json.loads((E/'INPUT_SELECTION.json').read_text()); frozen=json.loads((E/'MODEL_FREEZE.json').read_text())
grid=np.load(E/'grid_static256.npz'); times=[datetime.fromisoformat(t) for t in sel['history_times_BJT']]; ti={t:i for i,t in enumerate(times)}
D=Path('/dev/shm/hubei_cot_confirmation_215731/data'); assert D.exists()
features=[];rawvis=[];maxgeom=0.
for i,t in enumerate(times):
    raw=np.load(D/f'{stamp(t)}.npy',mmap_mode='r'); g=geom(grid,t)
    stored=np.stack((np.cos(np.deg2rad(raw[18])),np.cos(np.deg2rad(raw[17]-raw[15])),raw[19]))
    maxgeom=max(maxgeom,float(abs(stored-g).max()))
    features.append(np.concatenate((raw[AGRI,103:119,118:134],g[:,103:119,118:134])))
    if i%500==0:print('frames',i,flush=True)
features=np.array(features,np.float32); valid=(features[:,15:16]>.5)&(features[:,13:14]>np.cos(np.deg2rad(80.)))
cot=np.load(E/'predictions_attempt3/history_cot.npy'); assert cot.shape==(len(times),1,16,16)
assert np.all(cot[~valid]==0) and np.isfinite(cot).all() and maxgeom<1e-6
inits=[datetime.fromisoformat(t) for t in sel['initializations_BJT']]; ii={t:i for i,t in enumerate(inits)}
pred=np.empty((len(inits),16,13,16,16),np.float32);geometry=np.empty((len(inits),16,3,16,16),np.float32);seen=set()
for f in sorted((E/'inference_attempt3').glob('shard*/COMPLETE.json')):
    d=json.loads(f.read_text()); ids=d['sequence_ids'];assert not seen.intersection(ids);seen.update(ids)
    assert sha(f.parent/'pred.npy')==d['pred_sha256'] and sha(f.parent/'geometry.npy')==d['geometry_sha256']
    pred[ids]=np.load(f.parent/'pred.npy');geometry[ids]=np.load(f.parent/'geometry.npy')
assert seen==set(range(len(inits)))
station_checks={};torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True; maxreplay=0.;allvalid=[]
for st,(cy,cx) in CENTERS.items():
    ys=slice(cy-1,cy+2);xs=slice(cx-1,cx+2)
    with np.load(E/'predictions_attempt3'/f'{st}_seed42_G0.npz') as z: rr={k:z[k] for k in z.files}
    its=[datetime.fromisoformat(s) for s in rr['init_time_BJT']];targets=[datetime.fromisoformat(s) for s in rr['target_time_BJT']]
    si=np.array([ii[t] for t in its]);li=rr['lead_minutes']//15-1
    assert all(b-a==timedelta(minutes=int(l)) for a,b,l in zip(its,targets,rr['lead_minutes']))
    hi=np.array([[ti[t-timedelta(minutes=o)] for o in range(105,-1,-15)] for t in its])
    future=np.concatenate((pred[si,li,:,ys,xs],geometry[si,li,:2,ys,xs]),axis=1)
    history=features[hi,:15,ys,xs];c=cot[hi,:,ys,xs];v=valid[hi,:,ys,xs]
    cinit=cot[hi[:,-1],0,cy,cx];ctarget=np.array([cot[ti[t],0,cy,cx] if t in ti and valid[ti[t],0,cy,cx] else np.nan for t in targets])
    lastvalid=np.array([valid[j,0,cy,cx] for j in hi[:,-1]]);histvalid=v.mean(axis=(1,2,3,4))
    initcos=features[hi[:,-1],13,cy,cx];targetcos=geometry[si,li,0,cy,cx]
    np.savez_compressed(O/f'{st}_physical.npz',init_cot=cinit,target_cot=ctarget,last_valid=lastvalid,history_valid_fraction=histvalid,
        init_cosSOZ=initcos,target_cosSOZ=targetcos,init_time_BJT=rr['init_time_BJT'],target_time_BJT=rr['target_time_BJT'],lead_minutes=rr['lead_minutes'])
    station_checks[st]=dict(center_rc=[cy,cx],full_rc=[103+cy,118+cx],grid_lat=float(grid['lat'][103+cy,118+cx]),grid_lon=float(grid['lon'][103+cy,118+cx]),rows=len(si),max_geometry_error=0.)
    for sample in sorted(set(np.linspace(0,len(si)-1,15,dtype=int))):
        gp=geom(grid,targets[sample])[:2,103+cy-1:103+cy+2,118+cx-1:118+cx+2]
        err=float(abs(gp-future[sample,13:]).max());assert err<3e-7
        station_checks[st]['max_geometry_error']=max(station_checks[st]['max_geometry_error'],err)
    for seed in [42,43,44]:
        spec=next(s for s in frozen['GHI_heads'] if s['station']==st and s['seed']==seed and s['arm']=='G1')
        norm=json.loads(Path(spec['norm_source']).read_text())['data_meta']['base_norm'];assert norm['station_center_rc_in_patch']==[cy,cx]
        f=future.copy();h=history.copy();f[:,:13]=(f[:,:13]-np.array(norm['future_agri_mean'],np.float32)[None,:,None,None])/np.array(norm['future_agri_std'],np.float32)[None,:,None,None]
        h[:,:,:13]=(h[:,:,:13]-np.array(norm['history_agri_mean'],np.float32)[None,None,:,None,None])/np.array(norm['history_agri_std'],np.float32)[None,None,:,None,None]
        cc=np.where(v,(np.log1p(c)-np.float32(norm['history_cot_log1p_mean']))/np.float32(norm['history_cot_log1p_std']),0).astype(np.float32)
        model=HistoryCOTGHIHead().cuda().eval();model.load_state_dict(torch.load(spec['path'],map_location='cpu',weights_only=False)['model'])
        outputs=[];zeros=[]
        with torch.inference_mode():
            for j in range(0,len(si),1536):
                xx=[torch.from_numpy(np.ascontiguousarray(a[j:j+1536])).cuda() for a in [f,h,cc]]
                outputs.extend(model(*xx).cpu().numpy().tolist());xx[2].zero_();zeros.extend(model(*xx).cpu().numpy().tolist())
        orig=np.load(E/'predictions_attempt3'/f'{st}_seed{seed}_G1.npz');got=np.array(outputs)*rr['clear'].astype(np.float64)
        err=float(abs(got-orig['pred_ghi']).max());assert err<.002,err;maxreplay=max(maxreplay,err)
        np.savez_compressed(O/f'{st}_seed{seed}_counterfactual.npz',g1_zero_cot=np.array(zeros)*rr['clear'].astype(np.float64),g1_replay=got)
        print('head',st,seed,'replay',err,flush=True)
    # Input distribution relative to fitted TRAIN statistics: simple per-channel station-pixel summaries.
    pnorm=json.loads(Path(spec['norm_source']).read_text())['data_meta']['base_norm']
    station_checks[st]['new_history_cot_normalized_center_quantiles']=np.quantile(cc[:,-1,0,1,1],[0,.1,.5,.9,1]).tolist()
    station_checks[st]['new_history_valid_fraction_mean']=float(histvalid.mean())
# Export cases selected only for diagnosis, with missing future physical retrieval shown as missing.
cases=json.loads((O/'cases.json').read_text());case_rows=[]
for case in cases:
    t=datetime.fromisoformat(case['init']);idx=ii[t]; offs=list(range(-105,241,15));nf=len(offs)
    obs=np.full((nf,13,16,16),np.nan,np.float32);co=np.full((nf,16,16),np.nan,np.float32);vm=np.zeros((nf,16,16),bool)
    for j,off in enumerate(offs):
        tt=t+timedelta(minutes=off)
        if tt in ti:
            k=ti[tt];obs[j]=features[k,:13];co[j]=cot[k,0];vm[j]=valid[k,0]
    name=t.strftime('%Y%m%d_%H%M');np.savez_compressed(O/f'case_{name}.npz',offset_minutes=offs,observed_agri=obs,observed_R_cot=co,cot_valid=vm,forecast_agri=pred[idx],forecast_geometry=geometry[idx])
    case_rows.append(dict(**case,file=f'case_{name}.npz',available_future_frames=int(np.isfinite(obs[8:,0,0,0]).sum())))
report=dict(state='PASS_NEW_INPUT_ALIGNMENT_AND_GPU_HEAD_REPLAY',station_checks=station_checks,full_frame_geometry_max_abs_error=maxgeom,head_replay_max_abs_error_wm2=maxreplay,
    chronological_history_minutes=list(range(-105,1,15)),agri_channels=['C01','C02','C03','C04','C05','C06','C09','C10','C11','C12','C13','C14','C15'],target_interval='(target-15min,target]',past_only=True,
    counterfactual='G1 fixed head with standardized-zero COT; off-manifold diagnostic, not deployable forecast/model result',cases=case_rows,
    observed_future_R='retrieved from already available actual new-date AGRI, offline diagnostic only; not CPP truth',old_test_payloads_read=0,old_buffer_payloads_read=0)
(O/'ALIGNMENT_AUDIT.json').write_text(json.dumps(report,indent=2));print('PASS',flush=True)
