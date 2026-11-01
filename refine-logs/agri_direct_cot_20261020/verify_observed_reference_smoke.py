"""Bind remote extracted V inputs to archived real-image retrieval inputs."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from train_r_u import COTUNet

BASE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvread(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def main():
    p=BASE/'results/dense_observed_v_smoke_20261021_v1'
    s=json.loads((p/'complete.json').read_text());assert s['smoke'] and s['frames']==3
    for n,h in s['artifact_sha256'].items():assert sha(p/n)==h,n
    previous=BASE/'results/teacher_forecast_error_decomposition_v1_20261021'
    summary=json.loads((previous/'summary.json').read_text())
    oldrows=csvread(previous/'reference_rows.csv');bykey={(r['station'],r['target_time_utc']):r for r in oldrows}
    original=BASE.parents[1]/'experiments/monthly_R_station16_20261013_v1/data/pack_trainval_v1/x_raw.npy'
    assert sha(original)==summary['source']['raw_agri_sha256']
    oldx=np.load(original,mmap_mode='r'); oldmask=np.load(previous/'corrected_common_reference_mask.npy')[:,0]
    oldcot=np.load(previous/'observed_agri_r_u_cot.npy')[:,0]
    x=np.load(p/'x_raw.npy');inputs=[];matches=[]
    rdir=BASE/'data/r_u_frozen_model_20261021'
    assert sha(rdir/'epoch_050.pt')==summary['source']['r_checkpoint_sha256']
    assert sha(rdir/'normalization.json')==summary['source']['r_normalization_sha256']
    norm=json.loads((rdir/'normalization.json').read_text())
    for i,r in enumerate(csvread(p/'rows.csv')):
        for j,st in enumerate(('sili','zhujia')):
            old=bykey[(st,r['target_time_utc'])];refi=int(old['reference_index']);oi=int(old['array_index'])
            m=oldmask[refi];d=float(np.max(np.abs(x[i,j,:13][:,m]-oldx[oi,:13][:,m])))
            assert d<=1e-6,(st,r['target_time_utc'],d)
            inp=(x[i,j]-np.asarray(norm['mean'],np.float32)[:,None,None])/np.asarray(norm['std'],np.float32)[:,None,None]
            inp[~np.isfinite(inp)]=0;inputs.append(inp)
            matches.append({'station':st,'target_time_utc':r['target_time_utc'],'reference_index':refi,'agri_max_abs_difference':d})
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    model=COTUNet().to(device).eval();ck=torch.load(rdir/'epoch_050.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(ck['model'],strict=True)
    with torch.inference_mode():pred=torch.expm1(model(torch.from_numpy(np.stack(inputs)).to(device)).float()).cpu().numpy()[:,0]
    for i,m in enumerate(matches):
        valid=oldmask[m['reference_index']];d=pred[i][valid]-oldcot[m['reference_index']][valid]
        m['cot_reference_max_abs_difference']=float(np.abs(d).max());m['cot_reference_rmse']=float(np.sqrt(np.mean(d.astype(np.float64)**2)))
    result={'state':'PASS_THREE_FRAME_SIX_STATION_PATCH_SOURCE_RELOAD','matches':matches,'device':device,
        'source_and_archive_inputs_match':True,'prediction_is_offline_reference_only':True,'training_started':False,
        'test_payload_opened':False,'buffer_payload_opened':False,'script_sha256':sha(Path(__file__)),
        'smoke_receipt_sha256':sha(p/'complete.json')}
    (p/'source_reload_check.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
