"""Frozen same-fold R on actual train frames, for targets/diagnosis only."""
import argparse,csv,gc,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent;B=P.parent/'hubei_solar_geometry_repair_20261021';CODE=B/'code_pipeline_attempt12';sys.path.insert(0,str(CODE))
from train_cot_repaired import COTUNet
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
ap=argparse.ArgumentParser();ap.add_argument('--roles',required=True);a=ap.parse_args()
torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
rawroot=P/'raw_attempt1';assert json.loads((rawroot/'COMPLETE.json').read_text())['state']=='MATCHED_FOLD_R_RAW_TRAIN_PATCHES_READY'
frames=list(csv.DictReader((rawroot/'frames.csv').open()));idx={r['bjt']:int(r['frame_index']) for r in frames};rf=list(csv.DictReader((rawroot/'role_frames.csv').open()));pairs=list(csv.DictReader((rawroot/'pairs.csv').open()));raw=np.load(rawroot/'raw16.npy',mmap_mode='r')
source=json.loads((B/'GHI_pack_attempt12/pack_report.json').read_text())['source_correctutc_Rs'];source={r['role']:r for r in source}
for role in a.roles.split(','):
    assert role in [f'fold_{k}' for k in range(5)]
    out=P/'reference_attempt1'/role;out.mkdir(parents=True,exist_ok=False)
    ck=B/'R_runs_attempt12'/role/'best.pt';normpath=B/'prepared_attempt12/R_packs'/role/'norm.json'
    assert sha(ck)==source[role]['R_checkpoint_sha256'] and sha(normpath)==source[role]['R_normalization_sha256']
    trained=json.loads((ck.parent/'run_metadata.json').read_text());assert trained['model_source_sha256']==sha(CODE/'train_cot_repaired.py')
    norm=json.loads(normpath.read_text());mu=np.array(norm['mean'],np.float32)[None,:,None,None];sd=np.array(norm['std'],np.float32)[None,:,None,None];assert (sd>0).all()
    model=COTUNet().cuda().eval().requires_grad_(False);model.load_state_dict(torch.load(ck,map_location='cuda',weights_only=False)['model'])
    def infer(x):
        valid=(x[:,15:16]>.5)&(x[:,13:14]>np.cos(np.deg2rad(80.)))
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):y=torch.expm1(model(torch.from_numpy(np.ascontiguousarray((x-mu)/sd)).cuda()).float())
        y=y.masked_fill(~torch.from_numpy(valid).cuda(),0.).cpu().numpy();assert np.isfinite(y).all() and (y>=0).all();return y,valid
    witness=next(r for r in pairs if r['source_role']==role and r['all_frame_metadata_available']=='True');seq=int(witness['source_sequence_index']);lead=int(witness['lead'])//15-1;bank=B/'S_banks_attempt12'/role/'val'
    keys=list(csv.DictReader((bank/'keys.csv').open()));assert keys[seq]['init_time_bjt']==witness['init_time'] and keys[seq]['target_times_bjt'].split('|')[lead]==witness['target_time']
    pred=np.load(bank/'pred_agri_patch.npy',mmap_mode='r')[seq,lead];geo=np.load(bank/'future_geometry_patch.npy',mmap_mode='r')[seq,lead];x=np.concatenate((pred,geo))[None].astype('float32');q,_=infer(x)
    expected=np.load(B/'COT_banks_attempt12'/f'{role}.npy',mmap_mode='r')[seq,lead][None]
    difference=q.astype('float64')-expected;replay=dict(rmse=float(np.sqrt(np.mean(difference*difference))),max_abs=float(abs(difference).max()))
    assert replay['rmse']<=.05 and replay['max_abs']<=.25,replay
    times=sorted(r['bjt'] for r in rf if r['role']==role);ids=np.array([idx[t] for t in times]);values=[];valids=[]
    for j in range(0,len(ids),128):
        q,v=infer(np.asarray(raw[ids[j:j+128]],np.float32));values.append(q);valids.append(v)
    np.savez_compressed(out/'reference.npz',cot=np.concatenate(values),valid=np.concatenate(valids),bjt=np.array(times),raw_frame_indices=ids)
    (out/'COMPLETE.json').write_text(json.dumps(dict(state='MATCHED_FOLD_ACTUAL_AGRI_R_REFERENCE_COMPLETE',role=role,frames=len(ids),R_sha256=sha(ck),normalization_sha256=sha(normpath),model_source_sha256=sha(CODE/'train_cot_repaired.py'),forecast_replay=replay,reference_sha256=sha(out/'reference.npz'),test_outerbuffer_val_new_period_payloads_read=0,upstream_training=False,usage='Train target/offline reference only; not deployable input or independent COT truth'),indent=2));print(role,len(ids),replay,flush=True)
    del model;gc.collect();torch.cuda.empty_cache()
