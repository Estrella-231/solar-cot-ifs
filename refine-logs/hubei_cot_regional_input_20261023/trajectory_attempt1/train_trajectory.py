"""Fixed-budget paired trajectory fit; no outer-validation payload access."""
import argparse,csv,hashlib,json,math,os,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from trajectory_model import TrajectorySource,loss_parts,ddp_loss

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
def save(path,obj):
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pack',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--arm',choices=['full','repeat_last'],required=True);ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--resume',type=Path);ap.add_argument('--profile-only',action='store_true');ap.add_argument('--witness-only',action='store_true')
    ap.add_argument('--witness-init',type=Path)
    a=ap.parse_args();assert not(a.profile_only and a.witness_only)
    torch.set_num_threads(1);dist.init_process_group('nccl');rank=dist.get_rank();world=dist.get_world_size()
    assert world in (2,4) and 128%world==0
    local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);device=torch.device('cuda',local)
    if rank==0:
        a.out.mkdir(exist_ok=False,parents=True)
        assert json.loads((a.pack/'REPORT.json').read_text())['state']=='TRAIN_ONLY_TRAJECTORY_PAIR_PACK_COMPLETE_NOT_TRAINED'
        for name,digest in json.loads((a.pack/'SHA256.json').read_text()).items():assert sha(a.pack/name)==digest,name
    dist.barrier()
    rows=list(csv.DictReader((a.pack/'pairs.csv').open()))
    hlog=np.load(a.pack/'history_log.npy',mmap_mode='r');hvalid=np.load(a.pack/'history_valid.npy',mmap_mode='r')
    target=np.load(a.pack/'target_cot.npy',mmap_mode='r');mask=np.load(a.pack/'target_valid.npy',mmap_mode='r')
    geometry=np.load(a.pack/'target_geometry.npy',mmap_mode='r')
    hi=np.asarray([[int(v) for v in r['history_indices'].split('|')] for r in rows],np.int64)
    ti=np.asarray([int(r['target_index']) for r in rows],np.int64)
    motion=np.asarray([[float(r['motion_dy']),float(r['motion_dx'])] for r in rows],np.float32)
    lead=np.asarray([int(r['lead']) for r in rows],np.float32)
    support=np.asarray(mask).reshape(len(mask),-1).any(1)
    indices={role:np.asarray([i for i,r in enumerate(rows) if r['role']==role and support[ti[i]]],np.int64) for role in ['fit','select']}
    batches=len(indices['fit'])//128;assert batches>0 and len(indices['select'])>world
    torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);np.random.seed(a.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    model=TrajectorySource().to(device);ddp=DDP(model,device_ids=[local])
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=2000)
    scaler=torch.cuda.amp.GradScaler(enabled=not a.witness_only)
    binding=dict(pack_hash_file_sha256=sha(a.pack/'SHA256.json'),model_source_sha256=sha(Path(__file__).with_name('trajectory_model.py')),
       trainer_source_sha256=sha(__file__),arm=a.arm,seed=a.seed,global_batch=128,max_updates=2000,batches_per_epoch=batches,
       optimizer=dict(name='AdamW',lr=1e-3,weight_decay=1e-4),loss='global valid physical COT MSE /10000',clip=5.)
    def tensors(ids):
        values=[np.asarray(hlog[hi[ids]]),np.asarray(hvalid[hi[ids]]),motion[ids],lead[ids],
                np.asarray(geometry[ti[ids]]),np.asarray(target[ti[ids]]),np.asarray(mask[ti[ids]])]
        return [torch.from_numpy(np.ascontiguousarray(v)).to(device) for v in values]
    def update(ids):
        args=tensors(ids);opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16,enabled=not a.witness_only):q=ddp(*args[:5],arm=a.arm)
        assert torch.isfinite(q).all()
        num,count=loss_parts(q,args[5],args[6]);global_count=count.detach().clone();dist.all_reduce(global_count)
        loss=ddp_loss(num,global_count,world);scaler.scale(loss).backward();scaler.unscale_(opt)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
        scaler.step(opt);scaler.update();scheduler.step()
        return float(num.detach()),float(count)
    @torch.inference_mode()
    def evaluate():
        model.eval();totals=torch.zeros(2,device=device,dtype=torch.float64)
        ids=indices['select'][rank::world]
        for start in range(0,len(ids),32):
            args=tensors(ids[start:start+32])
            with torch.autocast('cuda',dtype=torch.float16):q=model(*args[:5],arm=a.arm)
            assert torch.isfinite(q).all();num,count=loss_parts(q,args[5],args[6]);totals+=torch.stack([num.double(),count])
        dist.all_reduce(totals);assert totals[1]>0
        return float(totals[0]/totals[1])
    def checkpoint(epoch,step,best,stale,best_epoch,best_model):
        return dict(binding=binding,model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
             scaler=scaler.state_dict(),epoch=epoch,global_step=step,best=best,stale=stale,best_epoch=best_epoch,best_model=best_model)
    start_epoch=0;step=0;best_epoch=-1;stale=0;best_model=None;best=float('inf')
    if a.resume:
        ck=torch.load(a.resume,map_location='cpu');assert ck['binding']==binding
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);scaler.load_state_dict(ck['scaler'])
        start_epoch=ck['epoch']+1;step=ck['global_step'];best=ck['best'];stale=ck['stale'];best_epoch=ck['best_epoch'];best_model=ck['best_model']
    ids=np.resize(indices['fit'],128);localids=ids[rank*128//world:(rank+1)*128//world]
    if a.profile_only or a.witness_only:
        assert a.resume is None
        shared_state_sha256=None
        if a.witness_init:
            assert a.witness_only
            ck=torch.load(a.witness_init,map_location='cpu')
            for key in ['pack_hash_file_sha256','model_source_sha256','arm','seed','global_batch']:
                assert ck['binding'][key]==binding[key],key
            model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
            scheduler.load_state_dict(ck['scheduler']);scaler.load_state_dict(ck['scaler'])
            shared_state_sha256=sha(a.witness_init)
        ddp.train()
        for _ in range(0 if a.witness_init else 3):update(localids)
        torch.cuda.synchronize();dist.barrier();started=time.monotonic();torch.cuda.reset_peak_memory_stats()
        timed_steps=1 if a.witness_init else 10
        for _ in range(timed_steps):num,count=update(localids)
        torch.cuda.synchronize();seconds=torch.tensor(time.monotonic()-started,device=device);dist.all_reduce(seconds,op=dist.ReduceOp.MAX)
        totals=torch.tensor([num,count],device=device,dtype=torch.float64);dist.all_reduce(totals)
        if rank==0:
            save(a.out/'checkpoint.pt',checkpoint(-1,scheduler.last_epoch,0,0,-1,None))
            if a.witness_only:save(a.out/'gradients.pt',{k:p.grad.detach().cpu().clone() for k,p in model.named_parameters()})
        dist.barrier();update(localids);expected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        ck=torch.load(a.out/'checkpoint.pt',map_location='cpu');model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler']);scaler.load_state_dict(ck['scaler']);assert scheduler.last_epoch==ck['global_step']
        update(localids);err=max(float((v.detach().cpu()-expected[k]).abs().max()) for k,v in model.state_dict().items())
        error=torch.tensor(err,device=device);dist.all_reduce(error,op=dist.ReduceOp.MAX);assert error<1e-7,float(error)
        memory=torch.tensor(torch.cuda.max_memory_allocated(),device=device);dist.all_reduce(memory,op=dist.ReduceOp.MAX)
        if rank==0:write(a.out/'PROFILE.json',dict(state='PASS_PROFILE_AND_FULL_STATE_REPLAY',world=world,global_batch=128,
            frames_per_second=128*timed_steps/float(seconds),seconds=float(seconds),memory_bytes=int(memory),resume_max_abs=float(error),
            global_loss=float(totals[0]/totals[1]),global_count=int(totals[1]),precision='fp32' if a.witness_only else 'fp16 AMP',
            outer_val_read=False,training_started=False,shared_state_sha256=shared_state_sha256))
        dist.destroy_process_group();return
    if best_model is None:
        best=evaluate();best_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if rank==0:
            save(a.out/'best.pt',dict(binding=binding,model=best_model,epoch=-1,global_step=0))
            write(a.out/'STEP0.json',dict(select_mse_normalized=best,baseline='fixed transport with persistence fallback',global_step=0))
    started=time.monotonic()
    for epoch in range(start_epoch,math.ceil(2000/batches)):
        ddp.train();order=np.random.default_rng(a.seed+epoch).permutation(indices['fit'])[:batches*128]
        total=torch.zeros(2,device=device,dtype=torch.float64)
        for chunk in order.reshape(batches,128):
            if step>=2000:break
            num,count=update(chunk[rank*128//world:(rank+1)*128//world]);total+=torch.tensor([num,count],device=device,dtype=torch.float64);step+=1
        dist.all_reduce(total);selected=evaluate();improved=selected<best
        if improved:best=selected;stale=0;best_epoch=epoch;best_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        if rank==0:
            record=dict(epoch=epoch,global_step=step,train_mse_normalized=float(total[0]/total[1]),select_mse_normalized=selected,
              best_epoch=best_epoch,best=best,seconds=time.monotonic()-started,world=world,dropped_rows=len(indices['fit'])-batches*128)
            write(a.out/f'epoch_{epoch:03d}.json',record);print(json.dumps(record),flush=True)
            save(a.out/'last.pt',checkpoint(epoch,step,best,stale,best_epoch,best_model))
            if improved:save(a.out/'best.pt',dict(binding=binding,model=best_model,epoch=epoch,global_step=step))
        dist.barrier()
        if stale>=5 or step>=2000:break
    if rank==0:write(a.out/'COMPLETE.json',dict(state='TRAJECTORY_FIT_INTERNAL_SELECTION_COMPLETE_NO_OUTER_VAL',arm=a.arm,seed=a.seed,
        world=world,global_step=step,final_epoch=epoch,best_epoch=best_epoch,best_select_mse_normalized=best,
        seconds=time.monotonic()-started,fit_rows=len(indices['fit']),select_rows=len(indices['select']),test_payloads_read=0,
        outer_validation_payloads_read=0,GHI_labels_read=0))
    dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
