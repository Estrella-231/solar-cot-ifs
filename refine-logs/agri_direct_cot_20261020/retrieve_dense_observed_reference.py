"""Frozen U-only R on observed V frames: offline reference, never a forecast input."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from train_r_u import COTUNet


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def read_csv(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();base=a.base;pack=base/'data/dense_observed_v_reference_patches_v1'
    receipt=json.loads((pack/'complete.json').read_text());assert receipt['frames']==4025 and not receipt['smoke']
    assert receipt['state']=='COMPLETE_OBSERVED_V_REFERENCE_PATCHES'
    for k in ('ghi_labels_read','cpp_read','training_started','test_payload_opened','buffer_payload_opened','future_observations_used_by_deployable_inference'):assert not receipt[k],k
    for n,h in receipt['artifact_sha256'].items():assert sha(pack/n)==h,n
    gate=json.loads((base/'data/dense_observed_reference_gate_20261021.json').read_text())
    manifest=json.loads((base/'data/cpp_score_inputs_v1/inference/inference_manifest.json').read_text())
    rdir=base/'data/r_u_frozen_model_20261021';rmeta=json.loads((rdir/'run_metadata.json').read_text())
    assert sha(rdir/'epoch_050.pt')==manifest['contract']['r_checkpoint_sha256']
    assert sha(rdir/'normalization.json')==manifest['contract']['r_norm_sha256']
    assert sha(base/'train_r_u.py')==rmeta['trainer_source_sha256']
    assert sha(base/'solar_geometry_corrected.py')==manifest['contract']['geometry_source_sha256']==gate['sha256']['geometry']
    assert sha(base/'data/sequence_support_v1/v_sequences.csv')==manifest['contract']['v_support_sha256']==gate['sha256']['v_support']
    assert rmeta['h_rows_used']==rmeta['outer_validation_rows_used']==0
    rows=read_csv(pack/'rows.csv');assert len(rows)==4025 and [int(r['frame_index']) for r in rows]==list(range(4025))
    x=np.load(pack/'x_raw.npy',mmap_mode='r');mask=np.load(pack/'mask.npy')
    assert x.shape==(4025,2,16,16,16) and mask.shape==(4025,2,1,16,16) and mask.dtype==bool
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    norm=json.loads((rdir/'normalization.json').read_text());mean=np.asarray(norm['mean'],np.float32)[:,None,None];std=np.asarray(norm['std'],np.float32)[:,None,None]
    assert np.isfinite(mean).all() and np.isfinite(std).all() and (std>0).all()
    flatmask=mask.reshape(-1,1,16,16);eligible=np.flatnonzero(flatmask.any((1,2,3)))
    pred=np.full((8050,1,16,16),np.nan,np.float32)
    torch.set_num_threads(4);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=COTUNet().to(device).float().eval().requires_grad_(False)
    ck=torch.load(rdir/'epoch_050.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model'],strict=True)
    started=time.time()
    with torch.inference_mode():
        for lo in range(0,len(eligible),128):
            ids=eligible[lo:lo+128];inp=np.asarray(x[ids//2,ids%2],np.float32)
            inp=(inp-mean)/std;inp[~np.isfinite(inp)]=0
            result=torch.expm1(model(torch.from_numpy(inp).to(device)).float()).cpu().numpy()
            mm=flatmask[ids]
            assert np.isfinite(result[mm]).all() and (result[mm]>=0).all()
            pred[ids]=np.where(mm,result,np.nan)
    pred=pred.reshape(4025,2,1,16,16)
    assert np.isnan(pred[~mask]).all() and np.isfinite(pred[mask]).all()
    np.save(a.output/'cot.npy',pred);np.save(a.output/'mask.npy',mask)
    (a.output/'rows.csv').write_bytes((pack/'rows.csv').read_bytes())
    previous=base/'results/teacher_forecast_error_decomposition_v1_20261021';oldrows=read_csv(previous/'reference_rows.csv')
    oldp=np.load(previous/'observed_agri_r_u_cot.npy')[:,0];oldm=np.load(previous/'corrected_common_reference_mask.npy')[:,0]
    bytime={r['target_time_utc']:i for i,r in enumerate(rows)};drifts=[];delta=[]
    for old in oldrows:
        if old['target_time_utc'] not in bytime:continue
        i=bytime[old['target_time_utc']];j=('sili','zhujia').index(old['station']);oi=int(old['reference_index'])
        common=oldm[oi]&mask[i,j,0]
        if not common.any():continue
        d=pred[i,j,0][common].astype(np.float64)-oldp[oi][common]
        drifts.append({'station':old['station'],'target_time_utc':old['target_time_utc'],'n':int(common.sum()),
            'rmse':float(np.sqrt(np.mean(d*d))),'max_abs_difference':float(np.abs(d).max())})
        delta.append(d)
    with (a.output/'sparse_reference_drift.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(drifts[0]));w.writeheader();w.writerows(drifts)
    dd=np.concatenate(delta)
    summary={'state':'COMPLETE_DENSE_OBSERVED_V_R_REFERENCE','frames':4025,'station_patches':8050,
        'eligible_station_patches':len(eligible),'unsupported_station_patches':8050-len(eligible),
        'valid_pixels':int(mask.sum()),'valid_cot_min':float(pred[mask].min()),'valid_cot_max':float(pred[mask].max()),
        'elapsed_seconds':time.time()-started,'device':str(device),'batch':128,'tensor_dtype':'torch.float32','autocast':False,
        'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
        'unsupported_output':'NaN with mask=false; never clear-sky zero',
        'sparse_comparison':{'matching_station_patches':len(drifts),'n':len(dd),'rmse':float(np.sqrt(np.mean(dd*dd))),'max_abs_difference':float(np.abs(dd).max())},
        'source':{'cpu_receipt_sha256':sha(pack/'complete.json'),'input_arrays_sha256':receipt['artifact_sha256'],
            'r_checkpoint_sha256':sha(rdir/'epoch_050.pt'),'r_norm_sha256':sha(rdir/'normalization.json'),
            'v_support_sha256':sha(base/'data/sequence_support_v1/v_sequences.csv'),
            'inference_manifest_sha256':sha(base/'data/cpp_score_inputs_v1/inference/inference_manifest.json'),
            'script_sha256':sha(Path(__file__)),'protocol_sha256':sha(base/'DENSE_REFERENCE_INFERENCE_PROTOCOL_20261021.md')},
        'artifact_sha256':{n:sha(a.output/n) for n in ('cot.npy','mask.npy','rows.csv','sparse_reference_drift.csv')},
        'future_observed_agri_read_for_reference':True,'future_observations_used_by_deployable_inference':False,
        'training_started':False,'ghi_labels_read':False,'test_payload_opened':False,'buffer_payload_opened':False,
        'scope':'seen development V; fidelity to learned R teacher only; not CPP truth or GHI improvement'}
    (a.output/'complete.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
