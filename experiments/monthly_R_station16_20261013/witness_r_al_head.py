#!/usr/bin/env python3
"""One-batch target-runtime AL shape/gradient witness using real train rows."""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<22),b''):h.update(b)
    return h.hexdigest()

def rows(p):
    with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def main():
    p=argparse.ArgumentParser()
    for k in ('feature-bank','pack','r-checkpoint','r-source','output'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--station',choices=('sili','zhujia'),default='sili');p.add_argument('--batch-size',type=int,default=16)
    a=p.parse_args()
    fm=json.loads((a.feature_bank/'feature_manifest.json').read_text())
    pm=json.loads((a.pack/'pack_manifest.json').read_text())
    if fm.get('state')!='COMPLETE' or pm.get('state')!='COMPLETE':raise ValueError('feature/GHI pack incomplete')
    if fm.get('test_payload_opened') is not False or pm.get('test_payload_opened') is not False:raise ValueError('test gate failed')
    if sha(a.r_checkpoint)!=fm.get('R_checkpoint_sha256') or sha(a.r_source)!=fm.get('R_source_sha256'):raise ValueError('R provenance mismatch')
    sys.path.insert(0,str(a.r_source.parent));sys.path.insert(0,str(Path(__file__).resolve().parent))
    from train_station16_R import COTUNet
    from head_candidate_v0 import LocalGHIHead
    from head_r_al_candidate import LocalGHIHeadAL,parameter_count
    ck=torch.load(a.r_checkpoint,map_location='cpu',weights_only=False)
    r=COTUNet().eval().requires_grad_(False);r.load_state_dict(ck['model'],strict=True)
    if any(x.requires_grad for x in r.parameters()):raise RuntimeError('R is not frozen')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');r.to(device)
    feat=np.load(a.feature_bank/'dec1_station_3x3.npy',mmap_mode='r',allow_pickle=False)
    fk=rows(a.feature_bank/'frame_keys.csv');fidx={q['target_time_utc']:int(q['index']) for q in fk}
    if feat.shape!=(len(fk),2,32,3,3):raise ValueError(f'feature bank shape mismatch {feat.shape}')
    gr=rows(a.pack/'rows.csv');si=('sili','zhujia').index(a.station)
    selected=[]
    for i,q in enumerate(gr):
        if q['split']=='train' and q['station']==a.station and q['target_time_utc'] in fidx:selected.append((i,q))
    if len(selected)!=sum(q['split']=='train' and q['station']==a.station for q in gr):raise ValueError('not all station train keys in feature bank')
    selected=selected[:a.batch_size]
    if len(selected)<2:raise ValueError('insufficient batch')
    indices=np.asarray([i for i,_ in selected]);keys=[q['target_time_utc'] for _,q in selected]
    fi=np.asarray([fidx[t] for t in keys],np.int64)
    xraw=np.load(a.pack/'agri_raw.npy',mmap_mode='r',allow_pickle=False)[indices]
    valid=np.load(a.pack/'agri_valid.npy',mmap_mode='r',allow_pickle=False)[indices].astype(bool)
    geo=np.asarray(np.load(a.pack/'geometry.npy',mmap_mode='r',allow_pickle=False)[indices],np.float32)
    fs=np.asarray(feat[fi,si],np.float32)
    fn=np.load(a.feature_bank/'train_feature_norm.npz',allow_pickle=False);fmv=np.asarray(fn['mean'],np.float32)[None,:,None,None];fsv=np.asarray(fn['std'],np.float32)[None,:,None,None]
    av=np.asarray(pm['AGRI_norm_mean'],np.float32)[None,:,None,None];ast=np.asarray(pm['AGRI_norm_std'],np.float32)[None,:,None,None]
    x=np.where(valid,(xraw-av)/ast,0).astype(np.float32);fs=(fs-fmv)/fsv
    ghi=torch.tensor([float(q['ghi']) for _,q in selected],dtype=torch.float32,device=device)
    clear=torch.tensor([float(q['clear']) for _,q in selected],dtype=torch.float32,device=device)
    xt=torch.tensor(x,dtype=torch.float32,device=device);gt=torch.tensor(geo,dtype=torch.float32,device=device)
    ft=torch.tensor(fs,dtype=torch.float32,device=device)
    torch.manual_seed(42);base=LocalGHIHead('A',2).to(device)
    torch.manual_seed(42);al=LocalGHIHeadAL(2).to(device)
    with torch.no_grad():
        al.agri_encoder.load_state_dict(base.agri_encoder.state_dict())
        al.readout[0].weight.zero_();al.readout[0].weight[:,:base.readout[0].in_features].copy_(base.readout[0].weight)
        al.readout[0].bias.copy_(base.readout[0].bias);al.readout[2:].load_state_dict(base.readout[2:].state_dict())
    ap=LocalGHIHead('AP',2)
    if parameter_count()!=sum(x.numel() for x in ap.parameters()):raise RuntimeError('AL/AP capacity mismatch')
    # The matched initialization zeros the new auxiliary columns in the first
    # readout layer so AL initially reproduces A. The first update therefore
    # teaches those readout columns; verify encoder gradients on the next batch.
    opt=torch.optim.AdamW(al.parameters(),lr=1e-3,weight_decay=1e-4)
    pred=al(xt,gt,ft);loss=(((pred*clear-ghi)/1000.)**2).mean();loss.backward()
    aux_cols=al.readout[0].weight.grad[:,32*3*3:]
    if aux_cols is None or not torch.isfinite(aux_cols).all() or float(aux_cols.abs().sum())<=0:
        raise RuntimeError('AL auxiliary readout columns have no finite nonzero first-step gradient')
    opt.step();opt.zero_grad(set_to_none=True)
    pred2=al(xt,gt,ft);loss2=(((pred2*clear-ghi)/1000.)**2).mean();loss2.backward()
    ag=[p.grad for p in al.aux_encoder.parameters()]
    if any(g is None or not torch.isfinite(g).all() for g in ag) or sum(float(g.abs().sum()) for g in ag)<=0:raise RuntimeError('AL auxiliary encoder has no finite nonzero gradient after matched warm-start update')
    if any(p.grad is not None for p in r.parameters()):raise RuntimeError('gradient leaked to frozen R')
    if ft.requires_grad or ft.grad is not None:raise RuntimeError('cached R feature is not detached')
    result={'state':'PASS_REAL_TRAIN_BATCH_SHAPE_GRADIENT','station':a.station,'batch':len(indices),'feature_shape':list(fs.shape),
            'AL_parameter_count':sum(p.numel() for p in al.parameters()),'AP_parameter_count':sum(p.numel() for p in ap.parameters()),
            'first_update_aux_readout_gradient_l1':float(aux_cols.abs().sum()),
            'AL_aux_encoder_gradient_l1_after_first_update':sum(float(g.abs().sum()) for g in ag),
            'R_all_grad_none':all(p.grad is None for p in r.parameters()),
            'feature_detached':not ft.requires_grad,'loss_after_first_update':float(loss2.detach()),'feature_manifest_sha256':sha(a.feature_bank/'feature_manifest.json'),
            'GHI_pack_manifest_sha256':sha(a.pack/'pack_manifest.json'),'test_payload_opened':False}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
