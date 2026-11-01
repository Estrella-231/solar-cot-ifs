"""Hunan epoch-150 EMA: complete frozen validation forecasts and station patches."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code_hunan_v1'))
from data import BASE, PREPARED, CHANNELS
from dataset import CachedDataset
from model import AGRIDyDiff, ModelConfig
from diffusion import DynamicalDiffusion

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()

def write(path,value):
    path=Path(path);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--indices-file',type=Path)
    a=ap.parse_args()
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    local=int(os.environ.get('LOCAL_RANK',0));torch.cuda.set_device(local)
    device=torch.device('cuda',local)
    if world>1:dist.init_process_group('nccl')
    a.output.mkdir(parents=True,exist_ok=True)
    run=BASE/'runs/seed42_150ep_hunan_v1';checkpoint_path=run/'last.pt'
    complete=json.loads((run/'training_complete.json').read_text())
    metadata=json.loads((run/'run_metadata.json').read_text())
    assert complete['status']=='PASS' and complete['epochs']==150 and complete['global_step']==66300
    assert complete['test_payloads_read']==metadata['test_payloads_read']==0
    assert metadata['region']=='Hunan_FULL' and metadata['old_subset_weights_reused'] is False
    data=CachedDataset('val')
    assert len(data)==3140 and len(CHANNELS)==13
    selected=list(range(len(data)))
    if a.indices_file:
        selected=json.loads(a.indices_file.read_text())['indices']
        assert len(selected)==16 and len(set(selected))==16 and all(0<=i<3140 for i in selected)
    record={'scope':'Hunan independently trained epoch150 EMA; frozen validation; not independent test',
            'indices':selected,'rows':[data.rows[i] for i in selected],
            'selection':'all frozen 3140 validation rows' if not a.indices_file else 'existing fixed16 pipeline smoke only',
            'hubei_training_day_overlap_indices':[],
            'overlap_note':'Hubei training dates do not define this Hunan-trained model evaluation',
            'checkpoint_epoch':150,'weights':'ema','seed_rule':'20261022+1000000+frozen_val_index',
            'manifest_sha256':data.manifest_sha256,'normalization_sha256':data.normalization_sha256,
            'test_payloads_read':0}
    if rank==0:
        if (a.output/'selection.json').exists():assert json.loads((a.output/'selection.json').read_text())==record
        else:write(a.output/'selection.json',record)
        provenance={'checkpoint':str(checkpoint_path),'checkpoint_sha256':sha(checkpoint_path),
                    'checkpoint_epoch':150,'weights':'ema','normalization_sha256':data.normalization_sha256,
                    'manifest_sha256':data.manifest_sha256,'script_sha256':sha(__file__),
                    'future_reference_in_sampling':False,'precision':'bf16 as training validation',
                    'sampling_steps':25,'members':1,'test_payloads_read':0}
        if (a.output/'provenance.json').exists():assert json.loads((a.output/'provenance.json').read_text())==provenance
        else:write(a.output/'provenance.json',provenance)
    if world>1:dist.barrier()
    provenance=json.loads((a.output/'provenance.json').read_text())
    model=AGRIDyDiff(ModelConfig()).to(device).eval().requires_grad_(False)
    payload=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    assert payload['epoch']==150 and payload['global_step']==66300
    model.load_state_dict(payload['ema'],strict=True);del payload
    diffusion=DynamicalDiffusion().to(device)
    mean=torch.tensor(data.mean,device=device)[None,None]
    std=torch.tensor(data.std,device=device)[None,None]
    start=time.monotonic();new=0
    for idx in selected[rank::world]:
        name=f'val_{idx:05d}';done=a.output/(name+'_complete.json')
        if done.exists():
            rec=json.loads(done.read_text())
            assert rec['index']==idx and rec['checkpoint_sha256']==provenance['checkpoint_sha256']
            assert (a.output/(name+'_pred.npy')).exists()
            continue
        sample=data[idx]
        batch={k:torch.from_numpy(v)[None].to(device) for k,v in sample.items()}
        seed=20261022+1000000+idx
        # Future images/QA occur only in the metrics below; sampler receives
        # historical images/QA and deterministic geometry, as during training.
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            normalized=diffusion.sample(model,batch['history'],batch['geometry'],batch['history_valid'],steps=25,seed=[seed]).float()
        assert normalized.shape==(1,16,13,256,256) and torch.isfinite(normalized).all()
        physical=normalized*std+mean
        coords=((163,148),(73,150))
        patch=np.stack([physical[0,:,:,rr-8:rr+8,cc-8:cc+8].cpu().numpy() for rr,cc in coords],axis=1)
        tmp=a.output/(name+'_pred.tmp')
        with tmp.open('wb') as f:np.save(f,patch)
        tmp.replace(a.output/(name+'_pred.npy'))
        y=batch['future'];mask=batch['future_valid'].bool()
        error=torch.where(mask,normalized-y,0.0)
        sse=error.square().sum((0,3,4),dtype=torch.float64).cpu().numpy()
        count=mask.sum((0,3,4)).cpu().numpy()
        rec={'index':idx,'seed':seed,'shape':list(patch.shape),'checkpoint_sha256':provenance['checkpoint_sha256'],
             'normalized_sse_lead_channel':sse.tolist(),'count_lead_channel':count.tolist(),
             'test_payloads_read':0}
        write(done,rec);new+=1
        event={'event':'hunan_forecast_window_complete','rank':rank,'index':idx,'new_windows':new,'elapsed_seconds':time.monotonic()-start}
        print(json.dumps(event),flush=True);write(a.output/f'progress_rank{rank}.json',event)
    if world>1:dist.barrier()
    if rank==0:
        records=[json.loads((a.output/f'val_{i:05d}_complete.json').read_text()) for i in selected]
        assert all(r['checkpoint_sha256']==provenance['checkpoint_sha256'] and r['test_payloads_read']==0 for r in records)
        sse=np.sum([r['normalized_sse_lead_channel'] for r in records],axis=0,dtype=np.float64)
        count=np.sum([r['count_lead_channel'] for r in records],axis=0,dtype=np.float64)
        result={'status':'PASS','scope':record['scope'],'windows':len(selected),'sampling_steps':25,'members':1,
                'checkpoint_epoch':150,'weights':'ema','checkpoint_sha256':provenance['checkpoint_sha256'],
                'normalization_sha256':data.normalization_sha256,'manifest_sha256':data.manifest_sha256,
                'patch_shape':[16,2,13,16,16],'seed_rule':record['seed_rule'],
                'forecast_normalized_rmse':float(np.sqrt(sse.sum()/count.sum())),
                'rmse_lead_channel':np.sqrt(sse/np.maximum(count,1)).tolist(),
                'test_payloads_read':0,'elapsed_seconds':time.monotonic()-start}
        write(a.output/'completion.json',result)
        print('HUNAN_FORECAST_COMPLETE',json.dumps(result),flush=True)
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()
