"""Frozen inference on selected existing validation cases; no model training."""
import csv,json,sys,hashlib
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
import torch
S=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907')
E=S/'experiments/regional_station_cot_20260925'
O=S/'experiments/history_cot_forecast_ghi_20261015_v1/diagnosis_20261015'
sys.path[:0]=[str(S/'scripts'),'/public/home/slfu/ttzhou/swc/irradiance_forecast/HunanSimVPFull_20260906']
from hunan_data import load_frame
from train_cot_repaired import COTUNet
from train_sili_transport_thickness_pilot_20260927 import warp_batch
from train_sili_history_repeat_control_20260928 import HistoryControlNet,history_input
from build_sili_cot_dynamics_bank_20260927 import local32_median_motion
torch.set_num_threads(4)
config=json.loads((S/'configs/s_frozen_hunan_seed42.json').read_text())
norm=json.loads(Path(config['normalization']).read_text())
manifest=list(csv.DictReader(Path(config['manifest']).open()))
val=[r for r in manifest if r['split']=='val']
assert hashlib.sha256(Path(config['manifest']).read_bytes()).hexdigest()=='9b6f5ff452a377a202db3d7dc305cebd74230cb084b6accf21df745b4b47dcf8'
daylist=list(csv.DictReader((O.parent/'day_assignment.csv').open()))
allowed={r['date_BJT'] for r in daylist if r['split'] in ('train','val')}
with np.load(O.parent/'results_attempt2/paired_pack.npz') as z: pack={k:z[k] for k in ('val_ids','val_seq','val_rows','val_lead','val_cot_transport','val_cot_hybrid','val_x')}
bank=E/'sili_cot_dynamics_bank_train512_val512_20260927_v1/val'
ids=np.load(bank/'sequence_indices.npy');lookup={int(k):i for i,k in enumerate(ids)}
hbank=np.load(bank/'cot_local_history.npy',mmap_mode='r');fbank=np.load(bank/'cot_local_forecast.npy',mmap_mode='r')
cpp=np.load(bank/'cot_local_cpp.npy',mmap_mode='r');cmask=np.load(bank/'cot_local_cpp_mask.npy',mmap_mode='r');vel=np.load(bank/'motion_local32.npy')
ib=E/'regional_ghi_pilot_bank_val512_20260927_v1/val';imids=np.load(ib/'sequence_indices.npy');imlookup={int(k):i for i,k in enumerate(imids)};im=np.load(ib/'image_local.npy',mmap_mode='r')
rm=json.loads((E/'agri_cpp_pair_bank_v1/norm.json').read_text());mean=torch.tensor(rm['mean'],device='cuda')[None,:,None,None];std=torch.tensor(rm['std'],device='cuda')[None,:,None,None]
r=COTUNet().cuda().eval().requires_grad_(False);state=torch.load(E/'r256_full_exploratory_seed42_batch512_node22/best.pt',map_location='cpu',weights_only=False);r.load_state_dict(state['model'])
hnet=HistoryControlNet().cuda().eval().requires_grad_(False);state=torch.load(E/'sili_full_vs_repeat_last_seed42_20260928_v5/full_history/best_learned.pt',map_location='cpu',weights_only=False);assert state['update']==96;hnet.load_state_dict(state['model'])
audits=[]
with torch.no_grad():
 for case in json.loads((O/'selected_cases.json').read_text()):
    seq=case['sequence'];row=val[seq];rels=row['data_relpaths'].split('|');assert len(rels)==24
    init=datetime.strptime(row['BJT_start'],'%Y-%m-%d %H:%M:%S')+timedelta(minutes=105)
    assert init.isoformat()+'+08:00'==case['init']
    assert all((datetime.strptime(Path(x).stem,'%Y%m%d%H%M')+timedelta(hours=8)).strftime('%Y-%m-%d') in allowed for x in rels)
    actual=[];valids=[];geometry=[];C=[]
    for ti,rel in enumerate(rels):
        a,v,g=load_frame('/public/home/slfu/ttzhou/Auxiliary_data/HuNan_AGRI_Preprocessed',rel)
        utc=datetime.strptime(Path(rel).stem,'%Y%m%d%H%M');assert utc+timedelta(hours=8)==init+timedelta(minutes=(ti-7)*15)
        actual.append(a);valids.append(v);geometry.append(g)
    actual=np.stack(actual);valids=np.stack(valids);geometry=np.stack(geometry)
    raw=np.concatenate((np.where(valids,actual,np.nan),geometry),axis=1)
    for lo in range(0,24,8):
        x=torch.tensor(raw[lo:lo+8],device='cuda');z=torch.nan_to_num((x-mean)/std,nan=0.,posinf=0.,neginf=0.)
        C.append(r(z).squeeze(1).cpu().numpy())
    oracle=np.concatenate(C)[:,131:195,116:180]
    j=lookup[seq];h=torch.tensor(np.asarray(hbank[j:j+1,0]),device='cuda');f=torch.tensor(np.asarray(fbank[j:j+1,0]),device='cuda');v=torch.tensor(vel[j:j+1],device='cuda')
    adv,support=warp_batch(h[:,-1],v);hy=hnet(history_input(h,f,adv,v,'full_history'),adv)
    image=np.asarray(im[imlookup[seq],0]);gerr=float(abs(image[:,13:]-geometry[8:,:,131:195,116:180]).max())
    assert gerr<1e-6,'Prediction geometry differs from raw target-time aux'
    hh=np.where(valids[:8],(actual[:8]-np.array(norm['mean'],np.float32)[None,:,None,None])/np.array(norm['std'],np.float32)[None,:,None,None],0)
    motion=local32_median_motion(hh);merr=float(abs(motion-vel[j]).max());assert merr<1e-3
    q=pack['val_ids'][pack['val_seq']]==seq;rows=np.flatnonzero(q);ld=pack['val_lead'][q]
    assert len(rows)==16 and np.array_equal(ld,np.arange(16))
    tr=adv[0,0].cpu().numpy();hy=hy[0,0].cpu().numpy()
    terr=float(abs(tr[:,31:34,31:34]-pack['val_cot_transport'][rows,8+ld]).max());herr=float(abs(hy[:,31:34,31:34]-pack['val_cot_hybrid'][rows,8+ld]).max());assert max(terr,herr)<1e-4
    image_err=float(abs(image[np.arange(16),:,31:34,31:34]-pack['val_x'][q]).max());assert image_err==0
    rgb=np.moveaxis(actual[8:,[2,1,0],131:195,116:180],1,-1)
    frgb=np.moveaxis(image[:,[2,1,0]]*np.array(norm['std'])[[2,1,0]][None,:,None,None]+np.array(norm['mean'])[[2,1,0]][None,:,None,None],1,-1)
    np.savez_compressed(O/f'case_{seq}.npz',rgb=rgb,rgb_valid=np.moveaxis(valids[8:,[2,1,0],131:195,116:180],1,-1),forecast_rgb=frgb,oracle_cot=np.expm1(oracle[8:]),oracle_history=np.expm1(oracle[:8]),transport=np.expm1(tr),hybrid=np.expm1(hy),transport_support=support[0,0].cpu().numpy(),cpp=np.expm1(np.asarray(cpp[j,0,:,0])),cpp_valid=np.asarray(cmask[j,0,:,0])>.5,history=np.expm1(np.asarray(hbank[j,0,:,0])),row_ids=pack['val_rows'][q],geometry=geometry[8:,:,131:195,116:180],relpaths=np.array(rels),motion=vel[j])
    audits.append(dict(sequence=seq,init=case['init'],station_global_rc=[163,148],crop64_top_left=[131,116],station_local_rc=[32,32],crop3_slice='31:34',target_geometry_max_abs_diff=gerr,history_motion_max_abs_diff=merr,transport_pack_max_log_diff=terr,hybrid_pack_max_log_diff=herr,image_pack_max_abs_diff=image_err,history_R_replay_max_log_diff=float(abs(oracle[:8]-np.asarray(hbank[j,0,:,0])).max()),input_future_observation=False,actual_future_used_for_diagnostic_only=True,test_payload_read=False))
    print('CASE_COMPLETE',seq,flush=True)
(O/'alignment_audit.json').write_text(json.dumps(audits,indent=2));print('CASE_EXTRACTION_COMPLETE',flush=True)
