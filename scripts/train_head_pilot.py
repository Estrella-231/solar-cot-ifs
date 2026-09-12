"""Fair frozen-feature A/AC seed42 validation pilot; no test or upstream gradients."""
import argparse,gc,hashlib,json,math,os,time
from pathlib import Path
import numpy as np
import torch
from torch import nn

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def atomic(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False));temp.replace(path)

def save(path,obj):
    temp=path.with_suffix('.tmp');torch.save(obj,temp);temp.replace(path)

class Head(nn.Module):
    def __init__(self,use_cot):
        super().__init__();self.use_cot=use_cot
        self.cnn=nn.Sequential(nn.Conv2d(13,16,3,padding=1),nn.SiLU(),
            nn.Conv2d(16,32,3,padding=1),nn.SiLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.lead=nn.Embedding(16,8);self.station=nn.Embedding(2,4)
        self.mlp=nn.Sequential(nn.Linear(52,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,1),nn.Softplus())
    def forward(self,x,c,g,station,lead):
        a=self.cnn(x)
        if not self.use_cot:c=torch.zeros_like(c)
        n=torch.zeros((len(x),1),device=x.device,dtype=x.dtype)
        return self.mlp(torch.cat((a,c,n,g,self.lead(lead),self.station(station)),1)).squeeze(1)

def seed(value=42):
    np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)

def prediction(model,data,ids):
    with torch.autocast('cuda',dtype=torch.bfloat16):
        return model(*(data[k][ids] for k in ('x','c','g','station','lead'))).float()

def update(model,opt,data,ids):
    opt.zero_grad(set_to_none=True)
    p=prediction(model,data,ids)
    # FP32 loss avoids FP16 overflow for the retained dawn kt outliers (~478).
    loss=((p-data['kt'][ids].float()).square()*data['weight'][ids]).mean()
    if not torch.isfinite(loss):raise RuntimeError('nonfinite weighted kt loss')
    loss.backward()
    # No added clipping or robust loss: preserve preregistered weighted kt MSE.
    if not torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all():
        raise RuntimeError('nonfinite head gradients')
    opt.step();return float(loss.detach())

@torch.no_grad()
def evaluate(model,data,ids,batch,keep=False):
    model.eval();sse=torch.zeros(2,device='cuda',dtype=torch.float64);counts=sse.clone();outputs=[]
    for ix in ids.split(batch):
        p=prediction(model,data,ix)
        if not torch.isfinite(p).all():raise RuntimeError('nonfinite validation predictions')
        residual=p.double()*data['clear'][ix]-data['ghi'][ix]
        for station in (0,1):
            mask=data['station'][ix]==station
            sse[station]+=residual[mask].square().sum();counts[station]+=mask.sum()
        if keep:outputs.append(p.cpu().numpy())
    assert (counts>0).all()
    rmse=torch.sqrt(sse/counts)
    return dict(station_equal_ghi_rmse=float(rmse.mean()),station_ghi_rmse=rmse.cpu().tolist(),
                station_counts=counts.long().cpu().tolist()),(np.concatenate(outputs) if keep else None)

@torch.no_grad()
def weighted_loss(model,data,ids):
    model.eval();p=prediction(model,data,ids)
    return float(((p-data['kt'][ids].float()).square()*data['weight'][ids]).mean())

def profile(data,train,out):
    path=out/'profile.json'
    if path.exists():
        c=json.loads(path.read_text());assert c['trials']
        top=max(t['samples_per_second'] for t in c['trials'])
        assert c['batch']==min(t['batch'] for t in c['trials'] if t['samples_per_second']>=0.95*top)
        assert c['batch'] in (64,128,256,512,1024,2048)
        return c['batch']
    trials=[]
    for batch in (64,128,256,512,1024,2048):
        model=opt=None
        try:
            seed();model=Head(True).cuda();opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
            ix=train[:batch];torch.cuda.reset_peak_memory_stats()
            for _ in range(3):update(model,opt,data,ix)
            torch.cuda.synchronize();start=time.monotonic()
            for _ in range(10):update(model,opt,data,ix)
            torch.cuda.synchronize();seconds=time.monotonic()-start
            peak=torch.cuda.max_memory_allocated()/2**30
            trial=dict(batch=batch,samples_per_second=batch*10/seconds,peak_GiB=peak)
            if peak<0.80*torch.cuda.get_device_properties(0).total_memory/2**30:trials.append(trial)
            else:break
            print('PROFILE',json.dumps(trial),flush=True)
        except torch.cuda.OutOfMemoryError:break
        finally:
            del model,opt;gc.collect();torch.cuda.empty_cache()
    assert trials
    top=max(t['samples_per_second'] for t in trials)
    batch=min(t['batch'] for t in trials if t['samples_per_second']>=0.95*top)
    atomic(path,dict(batch=batch,trials=trials,rule='smallest batch within 5 percent of best speed, cap2048, 20 percent memory reserve'))
    return batch

def overfit(data,train,out):
    path=out/'overfit8.json'
    if path.exists():
        record=json.loads(path.read_text());assert record['passed'] and len(record['sequences'])==8
        for r in record['groups']:
            assert r['passed'] and r['final']<=0.5*r['initial']
            assert sha(out/f"overfit8_{r['group']}.pt")==r['checkpoint_sha256']
            assert sha(out/f"overfit8_{r['group']}_predictions.npz")==r['predictions_sha256']
        return
    gen=torch.Generator(device='cuda').manual_seed(42)
    sequences=torch.unique(data['sequence'][train],sorted=True)
    selected=sequences[torch.randperm(len(sequences),generator=gen,device='cuda')[:8]]
    ids=train[torch.isin(data['sequence'][train],selected)]
    assert len(selected)==8 and len(ids)>0
    reports=[]
    for use_cot in (False,True):
        seed();model=Head(use_cot).cuda();opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        initial=weighted_loss(model,data,ids)
        for step in range(800):
            model.train();update(model,opt,data,ids)
        final=weighted_loss(model,data,ids)
        group='AC' if use_cot else 'A';weight_path=out/f'overfit8_{group}.pt'
        save(weight_path,dict(model=model.state_dict(),sequence_ids=selected.cpu(),rows=ids.cpu(),steps=800))
        reloaded=Head(use_cot).cuda().eval()
        reloaded.load_state_dict(torch.load(weight_path,map_location='cpu')['model'],strict=True)
        with torch.no_grad():
            pred=prediction(model,data,ids);restored=prediction(reloaded,data,ids)
        reload_error=float((pred-restored).abs().max());assert reload_error<1e-6
        predpath=out/f'overfit8_{group}_predictions.npz'
        with predpath.open('wb') as f:np.savez(f,pack_row=ids.cpu().numpy(),pred_kt=pred.cpu().numpy(),
            target_kt=data['kt'][ids].cpu().numpy(),weight=data['weight'][ids].cpu().numpy())
        report=dict(group=group,initial=initial,final=final,passed=final<=0.5*initial,
            checkpoint_sha256=sha(weight_path),predictions_sha256=sha(predpath),reload_max_error=reload_error)
        print('OVERFIT',json.dumps(report),flush=True);reports.append(report)
        del model,opt,reloaded
    result=dict(passed=all(r['passed'] for r in reports),sequences=selected.cpu().tolist(),rows=len(ids),
        steps=800,selection='seed42 uniformly selected training sequences, no outcome-based selection',
        gate='final weighted kt MSE <= half initial for both A and AC',groups=reports)
    atomic(path,result)
    assert result['passed'],'8-sequence overfit failed; no pilot launch'

def fit(data,train,val,out,group,lr,batch,metadata):
    run=out/f'{group}_lr{lr:g}_seed42';run.mkdir(parents=True,exist_ok=True)
    if (run/'complete.json').exists():
        report=json.loads((run/'complete.json').read_text())
        assert report['state']=='COMPLETE_VALIDATION_PILOT'
        assert sha(run/'best.pt')==report['checkpoint_sha256']
        assert sha(run/'validation_predictions.npz')==report['predictions_sha256']
        ck=torch.load(run/'best.pt',map_location='cpu')
        assert ck['metadata']==metadata and ck['group']==group and ck['lr']==lr and ck['batch']==batch
        return
    seed();model=Head(group=='AC').cuda();opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    best=float('inf');wait=0;epoch0=0
    if (run/'last.pt').exists():
        ck=torch.load(run/'last.pt',map_location='cpu');assert ck['metadata']==metadata and ck['group']==group and ck['lr']==lr
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        assert ck['batch']==batch
        best=ck['best'];wait=ck['wait'];epoch0=ck['next_epoch']
    for epoch in range(epoch0,50):
        if wait>=8:break
        gen=torch.Generator(device='cuda').manual_seed(42+epoch)
        ids=train[torch.randperm(len(train),generator=gen,device='cuda')]
        model.train();start=time.monotonic();loss_sum=0
        for ix in ids.split(batch):loss_sum+=update(model,opt,data,ix)*len(ix)
        result,_=evaluate(model,data,val,batch)
        score=result['station_equal_ghi_rmse'];improved=score<best
        if improved:best=score;wait=0
        else:wait+=1
        ck=dict(model=model.state_dict(),optimizer=opt.state_dict(),metadata=metadata,group=group,lr=lr,
                best=best,wait=wait,next_epoch=epoch+1,batch=batch,seed=42)
        if improved:save(run/'best.pt',ck)
        save(run/'last.pt',ck)
        report=dict(epoch=epoch,train_weighted_kt_mse=loss_sum/len(train),**result,best=best,wait=wait,seconds=time.monotonic()-start)
        atomic(run/f'epoch_{epoch:03d}.json',report);print(group,lr,json.dumps(report),flush=True)
    ck=torch.load(run/'best.pt',map_location='cpu');model.load_state_dict(ck['model']);model.cuda()
    report,pred=evaluate(model,data,val,batch,True)
    path=run/'validation_predictions.npz';temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:np.savez(f,pack_row=val.cpu().numpy(),pred_kt=pred,
        observed_ghi=data['ghi'][val].cpu().numpy(),clear_sky_ghi=data['clear'][val].cpu().numpy(),
        station=data['station'][val].cpu().numpy(),lead=data['lead'][val].cpu().numpy())
    temp.replace(path)
    # Independently aggregate saved outputs with NumPy double precision.
    with np.load(path) as z:
        residual=z['pred_kt'].astype('float64')*z['clear_sky_ghi']-z['observed_ghi']
        rmse=[float(np.sqrt(np.mean(residual[z['station']==s]**2))) for s in (0,1)]
    assert abs(np.mean(rmse)-report['station_equal_ghi_rmse'])<1e-8
    atomic(run/'complete.json',dict(state='COMPLETE_VALIDATION_PILOT',**report,best_epoch_0based=ck['next_epoch']-1,
        checkpoint_sha256=sha(run/'best.pt'),predictions_sha256=sha(path),test_used=False,
        prediction_recompute='PASS numpy float64 from saved outputs',label_pack_sha256=metadata['pack_audit_sha256']))
    del model,opt;gc.collect();torch.cuda.empty_cache()

def main():
    p=argparse.ArgumentParser();p.add_argument('--pack',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=('smoke','pilot'),default='smoke')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    audit=json.loads((a.pack/'audit.json').read_text());assert audit['state']=='COMPLETE'
    # The pack audit contains all immutable output hashes; reject missing or altered files.
    hashes={v['name']:v['sha256'] for v in audit['arrays'].values()}
    hashes.update({'rows.csv':audit['rows_csv_sha256'],'norm.json':audit['norm_sha256'],
                  'source_shards.json':audit['source_shards_sha256']})
    for name,digest in hashes.items():assert sha(a.pack/name)==digest,name
    data={}
    for key in ('x','c','g','kt','ghi','clear','station','lead','split','sequence','weight'):
        array=np.load(a.pack/f'{key}.npy',allow_pickle=False)
        assert np.isfinite(array).all(),key
        data[key]=torch.from_numpy(array).cuda()
    assert torch.cuda.is_bf16_supported()
    train=torch.where(data['split']==0)[0];val=torch.where(data['split']==1)[0]
    assert len(train)==538053 and len(val)==99849 and len(train)+len(val)==len(data['kt'])
    assert torch.isfinite(data['kt'].float()).all() and (data['kt']>=0).all()
    assert torch.max(torch.abs(data['kt']*data['clear']-data['ghi']))<1e-8
    assert data['x'].shape==(637902,13,16,16)
    metadata=dict(pack=str(a.pack),pack_audit_sha256=sha(a.pack/'audit.json'),code_sha256=sha(__file__),
        groups=['A','AC'],learning_rates=[0.0003,0.001],epochs_max=50,patience=8,seed=42,
        loss='train inverse-frequency weighted kt MSE, equal station/lead groups',
        selection='station-equal validation GHI RMSE; same LR budget',test_used=False,
        frozen_upstream='cached S and R; no upstream models in optimizer',
        parameters=sum(p.numel() for p in Head(True).parameters()),
        embedding_dimensions=dict(lead=8,station=4),common_geometry='patch mean cosSOZ/cosRAA/day_mask',
        mixed_precision='BF16 model, FP32 loss, FP64 GHI evaluation',kt_clipped=False)
    contract=a.output/'contract.json'
    if contract.exists():assert json.loads(contract.read_text())==metadata
    else:atomic(contract,metadata)
    print('PACK_LOADED',json.dumps(metadata),flush=True)
    torch.backends.cudnn.benchmark=True
    batch=profile(data,train,a.output)
    execution=dict(metadata,batch=batch,profile_sha256=sha(a.output/'profile.json'))
    exec_path=a.output/'execution_contract.json'
    if exec_path.exists():assert json.loads(exec_path.read_text())==execution
    else:atomic(exec_path,execution)
    overfit(data,train,a.output)
    smoke_receipt=a.output/'overfit_receipt.json'
    smoke_hash=sha(a.output/'overfit8.json')
    if smoke_receipt.exists():assert json.loads(smoke_receipt.read_text())['sha256']==smoke_hash
    else:atomic(smoke_receipt,dict(sha256=smoke_hash))
    if a.stage=='smoke':
        atomic(a.output/'smoke_complete.json',dict(state='COMPLETE_PROFILE_AND_OVERFIT',batch=batch,
            test_used=False,pilot_started=False,next='source dawn-label audit, then A/AC validation pilot'))
        print('SMOKE_COMPLETE',flush=True);return
    for group in ('A','AC'):
        for lr in (0.0003,0.001):fit(data,train,val,a.output,group,lr,batch,execution)
    atomic(a.output/'complete.json',dict(state='COMPLETE_A_AC_VALIDATION_PILOT',test_used=False,
        next='independent prediction audit and go/no-go; no automatic test or formal claim'))

if __name__=='__main__':main()
