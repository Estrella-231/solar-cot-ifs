"""Frozen-model attribution on train-internal select only, with raw error sums."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
import torch
from trajectory_model import TrajectorySource
from train_trajectory import sha

P=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
assert not a.out.exists()
selection=json.loads((P/'fits_attempt1/ARM_SELECTION.json').read_text());spec=next(r for r in selection['runs'] if Path(r['run'])==a.run)
assert sha(a.run/'best.pt')==spec['best_sha256']
ck=torch.load(a.run/'best.pt',map_location='cpu');assert ck['binding']['model_source_sha256']==sha(P/'trajectory_model.py')
pack=Path(json.loads((P/'PAIR_PACK_COMPLETE.json').read_text())['pack']);assert sha(pack/'SHA256.json')==ck['binding']['pack_hash_file_sha256']
allrows=list(csv.DictReader((pack/'pairs.csv').open()));target=np.load(pack/'target_cot.npy',mmap_mode='r');valid=np.load(pack/'target_valid.npy',mmap_mode='r')
supported=np.asarray(valid).reshape(len(valid),-1).any(1)
rows=[r for r in allrows if r['role']=='select' and supported[int(r['target_index'])]]
assert len(rows)==2744 and all(r['role']=='select' for r in rows)
history=np.load(pack/'history_log.npy',mmap_mode='r');hvalid=np.load(pack/'history_valid.npy',mmap_mode='r');geometry=np.load(pack/'target_geometry.npy',mmap_mode='r')
torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
model=TrajectorySource().cuda().eval();model.load_state_dict(ck['model'],strict=True)
fields=['count','sse_persistence','sse_transport','sse_learned','se_persistence','se_transport','se_learned','sum_learned_minus_transport']
strata=['all','near_zero','thin','moderate','thick'];scopes=['region_target_valid','region_common','station_common']
totals=np.zeros((3,5,len(fields)),np.float64);perframe=[];small=[];step0=[];last=[];refsmall=[];masksmall=[];common_small=[]
started=time.monotonic();objective_num=0.;objective_count=0
with torch.inference_mode():
    for start in range(0,len(rows),32):
        rr=rows[start:start+32];hi=np.array([[int(i) for i in r['history_indices'].split('|')] for r in rr]);ti=np.array([int(r['target_index']) for r in rr])
        arrays=[np.asarray(history[hi]),np.asarray(hvalid[hi]),np.array([[float(r['motion_dy']),float(r['motion_dx'])] for r in rr],np.float32),
            np.array([int(r['lead']) for r in rr],np.float32),np.asarray(geometry[ti])]
        args=[torch.from_numpy(np.ascontiguousarray(v)).cuda() for v in arrays]
        with torch.autocast('cuda',dtype=torch.float16):pred=model(*args,arm=spec['arm'])
        _,base,good=model.transport_features(*args,arm=spec['arm'])
        truth=np.asarray(target[ti])[:,0].astype(np.float64);target_mask=np.asarray(valid[ti])[:,0].astype(bool)
        physical=np.expm1(pred[:,0].float().cpu().numpy().astype(np.float64));transport=np.expm1(base[:,0].cpu().numpy().astype(np.float64))
        persistence=np.expm1(arrays[0][:,-1].astype(np.float64));common=target_mask & arrays[1][:,-1].astype(bool) & good[:,-1].cpu().numpy()
        assert np.isfinite(physical).all() and np.isfinite(transport).all()
        error=np.where(target_mask,physical-truth,0.);objective_num+=float(np.square(error).sum());objective_count+=int(target_mask.sum())
        # In region_target_valid, unavailable persistence remains excluded from the
        # baseline sums. Only learned/full-domain objective is compared to its receipt.
        station_mask=np.zeros_like(common)
        for y,x in [(111,124),(112,126),(110,128)]:station_mask[:,y,x]=True
        changes=physical-transport;ep=persistence-truth;et=transport-truth;el=physical-truth
        masks=[target_mask,common,common & station_mask]
        class_masks=[np.ones_like(common),truth<1,(truth>=1)&(truth<10),(truth>=10)&(truth<30),truth>=30]
        sums=np.zeros((len(rr),3,5,len(fields)),np.float64)
        for j,support in enumerate(masks):
            for k,cloud in enumerate(class_masks):
                m=support & cloud
                series=[m.sum(axis=(1,2))]
                # Main comparisons always use region_common or station_common.
                for e in [ep,et,el]:series.append(np.where(m,e*e,0.).sum(axis=(1,2)))
                for e in [ep,et,el]:series.append(np.where(m,e,0.).sum(axis=(1,2)))
                series.append(np.where(m,changes,0.).sum(axis=(1,2)))
                sums[:,j,k]=np.stack(series,axis=1)
        totals+=sums.sum(0);perframe.append(sums)
        cut=lambda v:np.stack([v[:,y-1:y+2,x-1:x+2] for y,x in [(111,124),(112,126),(110,128)]],1)
        small.append(cut(physical));step0.append(cut(transport));last.append(cut(persistence));refsmall.append(cut(truth));masksmall.append(cut(target_mask));common_small.append(cut(common))
        if start%512==0:print(json.dumps(dict(done=start+len(rr),total=len(rows),seconds=time.monotonic()-started)),flush=True)
observed=objective_num/objective_count/10000.;difference=abs(observed-spec['best_select_mse_normalized'])
assert difference<1e-7,(observed,spec['best_select_mse_normalized'],difference)
a.out.mkdir(parents=True)
np.savez(a.out/'predictions.npz',init=[r['init'] for r in rows],target=[r['target'] for r in rows],lead=[int(r['lead']) for r in rows],
    learned=np.concatenate(small),transport=np.concatenate(step0),persistence=np.concatenate(last),reference=np.concatenate(refsmall),
    target_valid=np.concatenate(masksmall),common=np.concatenate(common_small),frame_sums=np.concatenate(perframe),fields=fields,strata=strata,scopes=scopes)
report=dict(state='TRAIN_INTERNAL_FROZEN_REPLAY_COMPLETE',arm=spec['arm'],seed=spec['seed'],rows=len(rows),fields=fields,strata=strata,scopes=scopes,
    sums=totals.tolist(),full_domain_objective_normalized=observed,checkpoint_objective_normalized=spec['best_select_mse_normalized'],objective_abs_difference=difference,
    checkpoint_sha256=sha(a.run/'best.pt'),predictions_sha256=sha(a.out/'predictions.npz'),seconds=time.monotonic()-started,
    outer_validation_payloads_read=0,test_buffer_payloads_read=0,GHI_labels_read=0,new_training_runs=0,
    caveat='region_target_valid baseline input missing values are placeholders; use region_common/station_common for method comparisons')
(a.out/'COMPLETE.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
