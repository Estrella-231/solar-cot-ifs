"""Independently recompute validation metrics and bind saved predictions to best R."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from train_cot_repaired import COTUNet

root=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907')
run=root/'runs/cot_repaired_seed42_v1'
pack=root/'data/cot_repaired_pack_20260907_v1'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
status=json.loads((run/'status.json').read_text())
assert status['state']=='COMPLETE' and status['test_used'] is False
hashes=json.loads((run/'artifact_sha256.json').read_text())
for n,d in hashes.items():assert sha(run/n)==d,n
with (pack/'rows.csv').open(newline='') as f:rows=list(csv.DictReader(f))
with np.load(run/'validation_predictions.npz') as z:
    ids=z['row_indices'];pred=z['pred_log1p_cot'].astype(np.float64)
    truth=z['reference_cot'];mask=z['mask']
expected=np.array([i for i,r in enumerate(rows) if r['split']=='validation'])
assert np.array_equal(ids,expected) and len(ids)==6536
assert np.isfinite(pred).all() and (pred>=0).all()
targets=np.load(pack/'target.npy',mmap_mode='r')
masks=np.load(pack/'mask.npy',mmap_mode='r')
assert np.array_equal(truth,targets[ids]*100) and np.array_equal(mask,masks[ids])
error=(np.expm1(pred)-truth)[mask]
log_error=(pred-np.log1p(truth.astype(np.float64)))[mask]
abs_log=np.abs(log_error)
huber=float(np.where(abs_log<=0.1,0.5*log_error**2,0.1*(abs_log-0.05)).mean())
rmse=float(np.sqrt((error**2).mean()));mae=float(np.abs(error).mean())
reported=json.loads((run/'validation_metrics.json').read_text())
assert abs(rmse-reported['cot_rmse'])<1e-10
assert abs(mae-reported['cot_mae'])<1e-10
assert abs(huber-reported['huber_log1p'])<1e-7
torch.set_num_threads(1)
checkpoint=torch.load(run/'best.pt',map_location='cpu')
assert checkpoint['epoch']==status['best_epoch']
assert checkpoint['metadata']['test_used'] is False
model=COTUNet();model.load_state_dict(checkpoint['model']);model.eval()
norm=checkpoint['metadata']['norm']
assert norm==json.loads((pack/'norm.json').read_text())
sampled=[]
for station in ('sili','zhujia'):
    positions=[i for i,idx in enumerate(ids) if rows[int(idx)]['station_id']==station]
    sampled.extend(positions[i] for i in sorted({0,len(positions)//2,len(positions)-1}))
x=np.load(pack/'x_raw.npy',mmap_mode='r')[ids[sampled]].copy()
x=(x-np.array(norm['mean'],np.float32)[None,:,None,None])/np.array(norm['std'],np.float32)[None,:,None,None]
x[~np.isfinite(x)]=0
with torch.no_grad():actual=model(torch.from_numpy(x)).numpy()
max_error=float(np.abs(actual-pred[sampled]).max())
assert max_error<0.01,('CPU FP32 vs saved GPU FP16 mismatch',max_error)
station_metrics={}
for station in ('sili','zhujia'):
    selected=np.array([rows[int(i)]['station_id']==station for i in ids])
    residual=(np.expm1(pred[selected])-truth[selected])[mask[selected]]
    station_metrics[station]=dict(samples=int(selected.sum()),pixels=len(residual),
             rmse=float(np.sqrt((residual**2).mean())),mae=float(np.abs(residual).mean()))
result=dict(status='PASS',best_checkpoint=str(run/'best.pt'),best_sha256=hashes['best.pt'],
            validation_samples=len(ids),valid_pixels=int(mask.sum()),cot_rmse=rmse,cot_mae=mae,
            huber_log1p=huber,best_epoch_0based=checkpoint['epoch'],station_metrics=station_metrics,
            cpu_reload_samples=len(sampled),cpu_vs_saved_max_log1p_error=max_error,
            test_used=False,model_use='frozen R candidate for downstream A/AC; forecast-domain and GHI gains untested',
            script_sha256=sha(Path(__file__)))
(run/'independent_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
