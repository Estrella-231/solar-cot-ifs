"""DDP regional R training; monthly val is scored only after selection.

Run through torchrun on allocated GPUs. Profiles are train-only and write to
separate directories; a profile never becomes a training checkpoint.
"""
import argparse, csv, hashlib, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from regional_model import COTUNet, loss_parts, ddp_loss


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj, indent=2)); tmp.replace(path)


def save(path, obj):
    tmp = path.with_suffix('.tmp'); torch.save(obj, tmp); tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', type=Path, required=True); ap.add_argument('--roles', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--resume', type=Path)
    ap.add_argument('--profile-only', action='store_true'); ap.add_argument('--profile-steps', type=int, default=10)
    ap.add_argument('--witness-only', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(1)
    dist.init_process_group('nccl'); rank = dist.get_rank(); world = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    assert world in (1, 2, 4, 8) and 128 % world == 0
    if rank == 0:
        a.out.mkdir(parents=True, exist_ok=False)
        assert json.loads((a.pack/'REPORT.json').read_text())['state'] == 'REGIONAL_R_PACK_BUILT_NOT_TRAINED'
        for name, digest in json.loads((a.pack/'SHA256.json').read_text()).items():
            assert sha(a.pack/name) == digest, name
    dist.barrier()
    rows = list(csv.DictReader((a.pack/'rows.csv').open()))
    roles = json.loads(a.roles.read_text()); rm = {r['time_bjt']: r for r in roles['roles']}
    assert len(rm) == len(rows) and set(rm) == {r['time_bjt'] for r in rows}
    x = np.load(a.pack/'x_norm.npy', mmap_mode='r'); y = np.load(a.pack/'target_log1p.npy', mmap_mode='r')
    mask = np.load(a.pack/'mask.npy', mmap_mode='r')
    assert x.shape == (len(rows),16,256,256) and y.shape == mask.shape == (len(rows),1,256,256)
    assert mask.dtype == np.bool_
    supported = np.asarray(mask).reshape(len(rows), -1).any(1)
    indices = {k: np.asarray([i for i,r in enumerate(rows) if rm[r['time_bjt']]['role'] == k and supported[i]], dtype=np.int64)
               for k in ('R_fit','R_select','monthly_val')}
    assert all(len(v)>0 for v in indices.values())
    for r in rows:
        assert r['split'] == rm[r['time_bjt']]['outer_split']
    batches = len(indices['R_fit'])//128
    assert batches >= 1
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    model = nn.SyncBatchNorm.convert_sync_batchnorm(COTUNet()).to(device)
    ddp = DDP(model, device_ids=[local_rank])
    opt = torch.optim.AdamW(ddp.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50*batches)
    assert not (a.profile_only and a.witness_only)
    scaler = torch.cuda.amp.GradScaler(enabled=not a.witness_only)
    binding = dict(pack_hash_file_sha256=sha(a.pack/'SHA256.json'), roles_sha256=sha(a.roles),
                   model_source_sha256=sha(Path(__file__).with_name('regional_model.py')),
                   global_batch=128, batches_per_epoch=batches, max_epochs=50, seed=42)
    start_epoch = 0; step = 0; best = float('inf'); stale = 0; best_epoch = -1
    best_model = None
    if a.resume:
        ck = torch.load(a.resume, map_location='cpu')
        assert ck['binding'] == binding and not (a.profile_only or a.witness_only)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler']); scaler.load_state_dict(ck['scaler'])
        start_epoch=ck['epoch']+1; step=ck['global_step']; best=ck['best']; stale=ck['stale']; best_epoch=ck['best_epoch']
        best_model=ck['best_model']
        assert scheduler.last_epoch == step
    if rank == 0:
        write_json(a.out/'run_metadata.json', dict(**binding, world=world, gpu=torch.cuda.get_device_name(),
                   pbs_job_id=os.environ.get('PBS_JOBID'), architecture='UNet base16, two downs, SyncBN, softplus log1p COT',
                   optimizer='AdamW lr1e-3 wd1e-4', selection='lowest R_select valid-pixel Huber(delta .1), patience8',
                   precision='fp32 witness' if a.witness_only else 'fp16 AMP, finite unscaled gradients required', resume=str(a.resume),
                   supported_rows={k:len(v) for k,v in indices.items()}, reference='CPP retrieval, not independent truth',
                   test_payloads_read=0, GHI_labels_read=0, profile_only=a.profile_only, witness_only=a.witness_only))

    def tensors(ids):
        return tuple(torch.from_numpy(np.array(v[ids], copy=True)).to(device) for v in (x, y, mask))

    def update(ids):
        xx, yy, mm = tensors(ids); opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16, enabled=not a.witness_only): pred = ddp(xx)
        num, count = loss_parts(pred, yy, mm); dist.all_reduce(count)
        assert count > 0 and torch.isfinite(num)
        scaler.scale(ddp_loss(num,count,world)).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(ddp.parameters(),5.,error_if_nonfinite=True)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ddp.parameters())
        scaler.step(opt); scaler.update(); scheduler.step()
        return float(num.detach()), int(mm.sum())

    @torch.no_grad()
    def evaluate(ids, save_output=False):
        model.eval(); local_ids=ids[rank::world]
        totals=torch.zeros(2,device=device,dtype=torch.float64)
        out = None
        if save_output:
            out=np.lib.format.open_memmap(a.out/f'val_pred_log_rank{rank}.npy',mode='w+',dtype='float32',shape=(len(local_ids),1,256,256))
            np.save(a.out/f'val_indices_rank{rank}.npy',local_ids)
        for start in range(0,len(local_ids),32):
            ii=local_ids[start:start+32]; xx,yy,mm=tensors(ii)
            with torch.autocast('cuda',dtype=torch.float16): pred=model(xx).float()
            assert torch.isfinite(pred).all()
            num,count=loss_parts(pred,yy,mm); totals+=torch.stack((num.double(),count))
            if out is not None: out[start:start+len(ii)]=pred.cpu().numpy()
        if out is not None: out.flush()
        dist.all_reduce(totals); assert totals[1]>0
        return float(totals[0]/totals[1])

    if a.witness_only:
        ddp.train(); ids=np.resize(indices['R_fit'],128)
        local=ids[rank*128//world:(rank+1)*128//world]
        num,count=update(local)
        totals=torch.tensor([num,count],device=device,dtype=torch.float64); dist.all_reduce(totals)
        if rank==0:
            save(a.out/'witness.pt',dict(binding=binding,model=model.state_dict(),optimizer=opt.state_dict(),
                 scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),epoch=0,global_step=1,
                 gradients={k:v.grad.detach().cpu().clone() for k,v in model.named_parameters()}))
        dist.barrier()
        update(local)
        expected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        ck=torch.load(a.out/'witness.pt',map_location='cpu')
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler']); scaler.load_state_dict(ck['scaler'])
        assert ck['global_step']==scheduler.last_epoch==1
        update(local)
        err=max(float((v.detach().cpu()-expected[k]).abs().max()) for k,v in model.state_dict().items())
        assert err<1e-7, err
        if rank==0:
            write_json(a.out/'WITNESS.json',dict(state='PASS_GPU_GLOBAL_STEP_AND_CHECKPOINT_REPLAY',world=world,
                       global_batch=128,global_valid_pixels=int(totals[1]),global_huber=float(totals[0]/totals[1]),
                       resume_second_step_max_abs_difference=err,precision='fp32 witness',monthly_val_scored=False))
        dist.barrier(); dist.destroy_process_group(); return
    if a.profile_only:
        ddp.train(); ids=np.resize(indices['R_fit'],128)
        for _ in range(3): update(ids[rank*128//world:(rank+1)*128//world])
        torch.cuda.synchronize(); dist.barrier(); start=time.monotonic(); torch.cuda.reset_peak_memory_stats()
        for _ in range(a.profile_steps): update(ids[rank*128//world:(rank+1)*128//world])
        torch.cuda.synchronize(); seconds=torch.tensor(time.monotonic()-start,device=device); dist.all_reduce(seconds,op=dist.ReduceOp.MAX)
        memory=torch.tensor(torch.cuda.max_memory_allocated(),device=device); dist.all_reduce(memory,op=dist.ReduceOp.MAX)
        if rank==0:
            save(a.out/'amp_resume.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
                 scaler=scaler.state_dict(),global_step=scheduler.last_epoch))
        dist.barrier()
        local=ids[rank*128//world:(rank+1)*128//world]
        update(local); expected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        ck=torch.load(a.out/'amp_resume.pt',map_location='cpu')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler']);scaler.load_state_dict(ck['scaler'])
        assert scheduler.last_epoch==ck['global_step']
        update(local)
        resume_error=max(float((v.detach().cpu()-expected[k]).abs().max()) for k,v in model.state_dict().items())
        error=torch.tensor(resume_error,device=device);dist.all_reduce(error,op=dist.ReduceOp.MAX)
        assert error<1e-7,float(error)
        if rank==0: write_json(a.out/'PROFILE.json',dict(world=world,global_batch=128,steps=a.profile_steps,seconds=float(seconds),
                                frames_per_second=128*a.profile_steps/float(seconds),max_allocated_bytes=int(memory),
                                amp_checkpoint_replay_max_abs_difference=float(error),
                                reset_before_formal_training_required=True, val_scored=False))
        dist.destroy_process_group(); return
    start=time.monotonic()
    for epoch in range(start_epoch,50):
        ddp.train(); order=np.random.default_rng(42+epoch).permutation(indices['R_fit'])[:batches*128]
        total=torch.zeros(2,device=device,dtype=torch.float64)
        for chunk in order.reshape(batches,128):
            num,count=update(chunk[rank*128//world:(rank+1)*128//world]); total+=torch.tensor([num,count],device=device,dtype=torch.float64); step+=1
        dist.all_reduce(total); select=evaluate(indices['R_select'])
        improved=select<best
        if improved:
            best=select; stale=0; best_epoch=epoch
            best_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: stale+=1
        if rank==0:
            result=dict(epoch=epoch,global_step=step,train_huber=float(total[0]/total[1]),select_huber=select,
                        best_epoch=best_epoch,seconds=time.monotonic()-start,dropped_rows=len(indices['R_fit'])-batches*128)
            write_json(a.out/f'epoch_{epoch:03d}.json',result); print(json.dumps(result),flush=True)
            ck=dict(binding=binding,model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
                    epoch=epoch,global_step=step,best=best,stale=stale,best_epoch=best_epoch,best_model=best_model)
            save(a.out/'last.pt',ck)
            if improved: save(a.out/'best.pt',dict(model=best_model,binding=binding,epoch=epoch,global_step=step))
        dist.barrier()
        if stale>=8: break
    assert best_model is not None
    model.load_state_dict(best_model)
    val=evaluate(indices['monthly_val'],save_output=True)
    if rank==0:
        write_json(a.out/'training_complete.json',dict(state='R_REGION_TRAIN_COMPLETE_VAL_PREDICTIONS_SAVED',
                   selected_epoch=best_epoch,final_epoch=epoch,global_step=step,monthly_val_huber=val,
                   world=world,seconds=time.monotonic()-start,test_payloads_read=0,
                   pending='independent physical-COT and station/thickness/date metrics; no forecast/GHI gain established'))
    dist.barrier(); dist.destroy_process_group()


if __name__=='__main__':
    main()
