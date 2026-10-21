#!/usr/bin/env python3
"""Infer date-held-out historical COT for one frozen Hubei R fold."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, time
from pathlib import Path
from collections import Counter
import numpy as np
import torch

AGRI_CH = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]
CHANNELS = ["C01","C02","C03","C04","C05","C06","C09","C10","C11","C12","C13","C14","C15","cosSOZ","cosRAA","day_mask"]
STAGED_SHA = "8b1f95c2a2cf1d66cabd11539b975fe3dccd486146bf3bec17fb18e537f72183"
FOLD_SHA = "c86bd6713d5afea1d6864cc4ef6048a93b596051ea4358e7954f39c32b054122"
MODEL_SHA = "8af2d0e101b492bf2096223fe788b7f3a8a9b5ece2544db45903b95397620e05"
CKPT_SHA = [
 "4ffb7c841b8f278fcdcd6143062d5af8e9ce515d7139044c02946229795e5a68",
 "8525ab7e4a0a1a57db38221f53545f316af6bea84990e4d944e661c31208d558",
 "8029465a066bd5651272a2297102605fa5a36a6857248507dc74563fd212f2f3",
 "b4f8c035511e4f12d6867adb24b93250ba94be68d0646ea669587984390bf5aa",
 "91a98712546aba04f45685467c133aa74e728fba7d5b45311f307f2df82a6f52",
]

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def rows(path: Path):
    with path.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def staged_raw(path: Path) -> np.ndarray:
    with np.load(path,allow_pickle=False) as z:
        if not {'patch','cosSOZ','cosRAA','day_mask'}.issubset(z.files): raise ValueError(f'bad input schema: {path}')
        p=np.asarray(z['patch'],dtype=np.float32)
        if p.ndim!=3 or p.shape[0]<15 or p.shape[1:]!=(16,16): raise ValueError(f'bad AGRI patch {p.shape}')
        g=[np.asarray(z[k],dtype=np.float32) for k in ('cosSOZ','cosRAA','day_mask')]
        if any(x.shape!=(16,16) for x in g): raise ValueError(f'bad geometry {path}')
        x=np.concatenate((p[AGRI_CH],*(q[None] for q in g)),axis=0)
    if x.shape!=(16,16,16) or not np.isfinite(x[13:]).all(): raise ValueError(f'bad input tensor {path}')
    return x

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fold',type=int,required=True); p.add_argument('--staged-rows',type=Path,required=True)
    p.add_argument('--fold-map',type=Path,required=True); p.add_argument('--agri-root',type=Path,required=True)
    p.add_argument('--fold-pack',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--model-source',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch',type=int,default=256); a=p.parse_args()
    f=a.fold
    if f not in range(5): raise ValueError('fold must be 0..4')
    if a.output.exists(): raise FileExistsError(a.output)
    if not torch.cuda.is_available(): raise RuntimeError('allocated CUDA device required')
    if sha(a.staged_rows)!=STAGED_SHA or sha(a.fold_map)!=FOLD_SHA: raise ValueError('frozen key/fold manifest hash mismatch')
    if sha(a.model_source)!=MODEL_SHA or sha(a.checkpoint)!=CKPT_SHA[f]: raise ValueError('R source/checkpoint hash mismatch')
    status=json.loads((a.fold_pack/'pack_status.json').read_text())
    if status.get('state')!='HUBEI_R_TRAIN_VAL_CORRECT_UTC_PACK_BUILT_TEST_CLOSED' or status.get('test_payloads_read')!=0 or status.get('buffer_payloads_read')!=0 or status.get('heldout_train_date_overlap')!=0:
        raise ValueError(f'fold pack status failed: {status}')
    norm=json.loads((a.fold_pack/'norm.json').read_text())
    if norm.get('input_channels')!=CHANNELS or norm.get('fit_split')!='train only' or norm.get('test_used') is not False: raise ValueError('fold normalization contract mismatch')
    if json.loads((a.fold_pack/'SHA256.json').read_text()).get('norm.json')!=sha(a.fold_pack/'norm.json'): raise ValueError('fold norm hash mismatch')
    fold_assign={r['date_BJT'].replace('-',''):int(r['heldout_fold']) for r in rows(a.fold_map)}
    data=rows(a.staged_rows)
    if len(data)!=22061 or sum(r['split']=='train' for r in data)!=20206 or sum(r['split']=='val' for r in data)!=1855: raise ValueError('staged manifest counts changed')
    selected=[]
    for r in data:
        d=r['time_bjt'][:8]
        if r['split']=='train' and fold_assign.get(d)==f: selected.append(r)
        if r['split']=='val' and d in fold_assign: raise ValueError(f'val date in train fold map: {d}')
    expected=[4174,4052,4008,4031,3941][f]
    if len(selected)!=expected or len({r['time_bjt'] for r in selected})!=expected: raise ValueError(f'fold key support mismatch {len(selected)} != {expected}')
    root=a.agri_root.resolve()
    paths=[]
    for r in selected:
        q=Path(r['path_agri']).resolve()
        if not q.is_relative_to(root) or not q.is_file(): raise ValueError(f'input outside/missing staged root: {q}')
        paths.append(q)
    spec=importlib.util.spec_from_file_location('hubei_fold_r',a.model_source)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    model=mod.COTUNet().cuda().eval()
    ckpt=torch.load(a.checkpoint,map_location='cuda:0',weights_only=False)
    if 'model' not in ckpt: raise ValueError('checkpoint has no model state')
    model.load_state_dict(ckpt['model'],strict=True)
    mu=np.asarray(norm['mean'],dtype=np.float32).reshape(16,1,1)
    sd=np.asarray(norm['std'],dtype=np.float32).reshape(16,1,1)
    pred=np.empty((len(selected),16,16),dtype=np.float32); day=np.empty((len(selected),16,16),dtype=np.bool_)
    torch.set_num_threads(1); torch.backends.cudnn.benchmark=False
    start=time.time(); n=0
    with torch.inference_mode():
        for s in range(0,len(selected),a.batch):
            end=min(s+a.batch,len(selected)); raw=np.stack([staged_raw(q) for q in paths[s:end]])
            solar_valid=raw[:,15:16]>0.5
            day_b=solar_valid[:,0]
            x=(raw-mu[None])/sd[None]
            # The train-fitted day_mask has std=1e-6 because supervised train
            # pixels are all daytime. Match the approved full-bank inference:
            # feed a neutral normalized value at night instead of -1e6.
            x[:,15:16]=np.where(solar_valid,x[:,15:16],0.0)
            x[~np.isfinite(x)]=0.0
            x_tensor=torch.from_numpy(x).cuda()
            with torch.autocast('cuda',dtype=torch.float16): out=model(x_tensor).float()
            if not torch.isfinite(out).all():
                raise RuntimeError(f'nonfinite log1p(COT) output fold={f} rows={s}:{end} input_absmax={float(x_tensor.abs().max())} channel_absmax={x_tensor.abs().amax(dim=(0,2,3)).detach().cpu().tolist()}')
            cot_t=torch.expm1(out)
            cot_t=cot_t.masked_fill(~torch.from_numpy(solar_valid).cuda(),0.0)
            if not torch.isfinite(cot_t).all() or (cot_t<0).any():
                finite_log=out[torch.isfinite(out)]
                raise RuntimeError(f'invalid COT output fold={f} rows={s}:{end} log_min={float(finite_log.min()) if finite_log.numel() else "NA"} log_max={float(finite_log.max()) if finite_log.numel() else "NA"} cot_min={float(torch.nan_to_num(cot_t).min())} cot_max={float(torch.nan_to_num(cot_t).max())}')
            cot=cot_t.cpu().numpy()[:,0]
            pred[s:end]=cot; day[s:end]=day_b; n+=end-s
            print(f'FOLD={f} ROWS={n}/{len(selected)}',flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.mkdir(parents=False,exist_ok=False)
    np.save(a.output/'cot_oof.npy',pred,allow_pickle=False); np.save(a.output/'day_mask.npy',day,allow_pickle=False)
    with (a.output/'rows.csv').open('w',newline='',encoding='utf-8') as fp:
        w=csv.writer(fp); w.writerow(['index','time_bjt','split','heldout_fold','path_agri'])
        w.writerows([[i,r['time_bjt'],r['split'],f,r['path_agri']] for i,r in enumerate(selected)])
    report={'state':'COMPLETE_TRAIN_DATE_HELDOUT_HISTORICAL_COT_INFERENCE','purpose':'input-origin distribution diagnostic only; no COT labels or GHI scored',
      'fold':f,'rows':len(selected),'test_payloads_read':0,'buffer_payloads_read':0,'cpp_target_or_quality_payloads_read':0,'ghi_labels_read':0,
      'staged_rows_sha256':sha(a.staged_rows),'fold_map_sha256':sha(a.fold_map),'fold_norm_sha256':sha(a.fold_pack/'norm.json'),
      'fold_checkpoint_sha256':sha(a.checkpoint),'model_source_sha256':sha(a.model_source),'prediction_sha256':sha(a.output/'cot_oof.npy'),
      'day_mask_sha256':sha(a.output/'day_mask.npy'),'batch':a.batch,'elapsed_seconds':time.time()-start,
      'rows_per_second':len(selected)/(time.time()-start),'gpu':torch.cuda.get_device_name(0),'visible_device':torch.cuda.current_device(),
      'night_day_mask_normalization':'neutral zero in normalized channel, matching full-bank inference',
      'out_of_range_day_pixels_gt100':int(((pred>100)&day).sum()),'valid_day_pixels':int(day.sum()),
      'note':'Date-held-out R prediction from the audited AGRI/geometry inputs. COT values are model inputs/retrieval estimates, not independent truth; values were not clipped.'}
    (a.output/'fold_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
if __name__=='__main__': main()
