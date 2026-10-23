"""Frozen-upstream 34-channel COT residual pilot. F roles only."""
import argparse,copy,csv,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
P=Path(__file__).resolve().parent
class Corrector(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(34,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,1,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x,base,available,direction):
        delta=self.net(x)
        delta=torch.where(direction>0,delta.clamp_min(0),torch.where(direction<0,delta.clamp_max(0),torch.zeros_like(delta)))
        y=torch.expm1(torch.relu(torch.log1p(base)+delta))
        y=torch.where(direction==0,base,y)
        return torch.where(available,y,torch.zeros_like(y))
def loss_fn(pred,target,valid):
    error=torch.where(valid,pred-target,torch.zeros_like(pred))
    return error.square().sum()/valid.sum().clamp_min(1)/10000
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def seed(s):
    torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def optimizer(model):
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=600)
    return opt,sched
def loadpack():
    manifest=json.loads((P/'PACK_COMPLETE.json').read_text());root=Path(manifest['path'])
    assert manifest['state']=='F_ONLY_CAUSAL_COT_CORRECTION_PACK_READY'
    arrays={name:np.load(root/(name+'.npy'),mmap_mode='r') for name in ['x','base','target','persistence','mask','forecast_mask','score_mask','fit','select']}
    for name in arrays:assert sha(root/(name+'.npy'))==manifest['hashes'][name+'.npy']
    fields=list(csv.DictReader((root/'fields.csv').open()));members=list(csv.DictReader((root/'members.csv').open()))
    assert len(fields)==manifest['fields'] and len(members)==manifest['station_members']
    tensors={name:torch.from_numpy(np.array(arrays[name])).cuda() for name in ['x','base','target','persistence','mask','forecast_mask','score_mask']}
    gate=json.loads((P/'GATE_COMPLETE.json').read_text());assert gate['state']=='PASS_CAUSAL_OOF_DIRECTION_MAPS_AND_ALL_STATION_REPLAYS'
    assert sha(gate['path'])==gate['sha256']
    direction=np.load(gate['path'],mmap_mode='r');assert direction.shape==arrays['base'].shape
    tensors['direction']=torch.from_numpy(np.array(direction)).cuda()
    fit=np.flatnonzero(arrays['fit']);sel=np.flatnonzero(arrays['select'])
    fit=fit[np.asarray(arrays['mask'][fit]).reshape(len(fit),-1).any(1)]
    return manifest,root,arrays,fields,members,tensors,fit,sel
def trainstep(model,opt,sched,t,ids):
    model.train();opt.zero_grad(set_to_none=True)
    pred=model(t['x'][ids],t['base'][ids],t['forecast_mask'][ids],t['direction'][ids]);loss=loss_fn(pred,t['target'][ids],t['mask'][ids])
    assert torch.isfinite(loss);loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    opt.step();sched.step();return float(loss.detach())
def profile(t,fit):
    seed(42);model=Corrector().cuda();opt,sched=optimizer(model);active=t['direction'][torch.tensor(fit,device='cuda')].ne(0).flatten(1).any(1).cpu().numpy();ids=torch.tensor(fit[active][:128],device='cuda');assert len(ids)>0
    with torch.no_grad():
        out=model(t['x'][ids],t['base'][ids],t['forecast_mask'][ids],t['direction'][ids]);err=float(abs(out-t['base'][ids])[t['forecast_mask'][ids]].max())
    assert err<1e-4,err
    # Force both residual signs before fitting; no label-based choice of direction.
    with torch.no_grad():
        for forced in [-.15,.15]:
            model.net[-1].bias.fill_(forced);q=model(t['x'][ids],t['base'][ids],t['forecast_mask'][ids],t['direction'][ids]);b=t['base'][ids];d=t['direction'][ids];av=t['forecast_mask'][ids]
            assert torch.equal(q[(d==0)&av],b[(d==0)&av])
            assert (q[(d>0)&av]>=b[(d>0)&av]-1e-4).all() and (q[(d<0)&av]<=b[(d<0)&av]+1e-4).all()
            assert (q[~av]==0).all()
        model.net[-1].bias.zero_()
    nan=torch.tensor([float('nan'),2.],device='cuda').reshape(2,1,1,1);truth=torch.tensor([float('nan'),1.],device='cuda').reshape_as(nan);vm=torch.tensor([False,True],device='cuda').reshape_as(nan)
    assert abs(float(loss_fn(nan,truth,vm))-.0001)<1e-10
    value=trainstep(model,opt,sched,t,ids);grad=max(float(p.grad.abs().max()) for p in model.parameters() if p.grad is not None);assert grad>0
    snapshot={'model':copy.deepcopy(model.state_dict()),'optimizer':copy.deepcopy(opt.state_dict()),'scheduler':copy.deepcopy(sched.state_dict()),'step':1}
    torch.save(snapshot,P/'training_attempt1'/'profile_resume.pt')
    for _ in range(2):trainstep(model,opt,sched,t,ids)
    seed(987);restored=Corrector().cuda();oo,ss=optimizer(restored);saved=torch.load(P/'training_attempt1'/'profile_resume.pt',map_location='cuda',weights_only=False)
    restored.load_state_dict(saved['model']);oo.load_state_dict(saved['optimizer']);ss.load_state_dict(saved['scheduler'])
    for _ in range(2):trainstep(restored,oo,ss,t,ids)
    diff=max(float(abs(a-b).max()) for a,b in zip(model.parameters(),restored.parameters()));assert diff==0.,diff
    assert opt.param_groups[0]['lr']==oo.param_groups[0]['lr'] and sched.state_dict()==ss.state_dict()
    torch.cuda.synchronize();start=time.perf_counter()
    for _ in range(50):trainstep(model,opt,sched,t,ids)
    torch.cuda.synchronize();secs=time.perf_counter()-start
    report=dict(state='PASS_CORRECTOR_STEP0_GRADIENT_MASK_RESUME_PROFILE',step0_max_abs=err,nonzero_gradient=grad,resume_weight_difference=diff,forced_residual_signs_checked=[-.15,.15],stable_exact_base=True,unavailable_zero=True,direction_respected=True,profile_steps=50,seconds=secs,samples_per_second=50*128/secs,peak_memory_bytes=torch.cuda.max_memory_allocated(),parameters=sum(p.numel() for p in model.parameters()),GPU=torch.cuda.get_device_name(),budget_updates=600,global_batch=128)
    (P/'training_attempt1/PROFILE.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['profile','train'],required=True);ap.add_argument('--seed',type=int,default=42);a=ap.parse_args()
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest,root,arrays,fields,members,t,fit,select=loadpack()
    if a.mode=='profile':profile(t,fit);return
    assert json.loads((P/'training_attempt1/PROFILE.json').read_text())['state'].startswith('PASS_CORRECTOR_')
    out=P/'training_attempt1'/f'seed{a.seed}';out.mkdir(exist_ok=False)
    seed(a.seed);model=Corrector().cuda();opt,sched=optimizer(model)
    rng=np.random.default_rng(20261023);schedule=[]
    while sum(len(s) for s in schedule)<600*128:schedule.append(rng.permutation(fit))
    schedule=np.concatenate(schedule)[:600*128].reshape(600,128);schedule_hash=hashlib.sha256(schedule.tobytes()).hexdigest()
    init_hash=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    metadata=dict(seed=a.seed,parameters=sum(p.numel() for p in model.parameters()),global_batch=128,updates=600,lr=1e-3,weight_decay=1e-4,initialization_sha256=init_hash,schedule_sha256=schedule_hash,fit_fields=len(fit),select_fields=len(select),pack_manifest_sha256=sha(P/'PACK_COMPLETE.json'),source_sha256=sha(__file__),upstream_training=False,GHI_values_read=0,outer_payloads_read=0)
    control=P.parent/'hubei_forecast_domain_cot_20261023'/'training_attempt1'/f'seed{a.seed}'/'run_metadata.json'
    control_meta=json.loads(control.read_text());assert init_hash==control_meta['initialization_sha256'] and schedule_hash==control_meta['schedule_sha256']
    assert len(fit)==control_meta['fit_fields']
    metadata.update(direction_gate_sha256=json.loads((P/'GATE_COMPLETE.json').read_text())['sha256'],control_metadata_sha256=sha(control),method='fixed direction-constrained COT residual; residual7249 plus frozen linear direction branch921')
    (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
    selset=set(select.tolist());centers={'Niushou':(8,6),'Wangzhai':(9,8),'Mizhuang':(7,10)};groups=[]
    for st,(cy,cx) in centers.items():
        for lead in range(15,241,15):
            ids=[int(r['field_index']) for r in members if r['station']==st and int(r['field_index']) in selset and int(fields[int(r['field_index'])]['lead'])==lead]
            ids=np.array(ids,dtype='int64');valid=np.asarray(arrays['score_mask'][ids,0,cy,cx]);ids=ids[valid];assert len(ids)>0,(st,lead)
            groups.append((torch.tensor(ids,device='cuda'),cy,cx))
    def evaluate():
        model.eval();sse=[0.]*len(groups)
        # Compact predictions stay on GPU; no real future input is passed into model.
        predictions=torch.zeros_like(t['base'])
        with torch.no_grad():
            for j in range(0,len(select),256):
                ids=torch.tensor(select[j:j+256],device='cuda');predictions[ids]=model(t['x'][ids],t['base'][ids],t['forecast_mask'][ids],t['direction'][ids])
            scores=[torch.sqrt(((predictions[ids,0,cy,cx].double()-t['target'][ids,0,cy,cx].double())**2).mean()).item() for ids,cy,cx in groups]
        assert np.isfinite(scores).all();return float(np.mean(scores)),predictions[torch.tensor(select,device='cuda')].cpu().numpy()
    best=float('inf');history=[];start=time.perf_counter()
    for step in range(601):
        if step:
            ids=torch.tensor(schedule[step-1],device='cuda');lastloss=trainstep(model,opt,sched,t,ids)
        if step%50==0:
            score,pred=evaluate();history.append(dict(step=step,select_equal_station_lead_rmse=score,train_loss=None if step==0 else lastloss,elapsed_seconds=time.perf_counter()-start))
            checkpoint=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),step=step,seed=a.seed,metadata=metadata,select_score=score)
            if score<best:
                best=score;torch.save(checkpoint,out/'best.pt');np.savez_compressed(out/'best_select_predictions.npz',prediction=pred,field_index=select)
            torch.save(checkpoint,out/'last.pt');(out/'curve.json').write_text(json.dumps(history,indent=2));print(a.seed,step,score,flush=True)
    final=dict(state='COT_DOMAIN_CORRECTOR_F_TRAINING_COMPLETE',seed=a.seed,updates=600,best_step=min(history,key=lambda r:r['select_equal_station_lead_rmse'])['step'],best_select_rmse=best,seconds=time.perf_counter()-start,best_checkpoint_sha256=sha(out/'best.pt'),prediction_sha256=sha(out/'best_select_predictions.npz'),upstream_training=False,outer_payloads_read=0)
    (out/'COMPLETE.json').write_text(json.dumps(final,indent=2));print(json.dumps(final),flush=True)
if __name__=='__main__':main()
