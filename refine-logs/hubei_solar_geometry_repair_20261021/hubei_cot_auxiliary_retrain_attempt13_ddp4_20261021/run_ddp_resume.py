"""Resume the same attempt13 COT arm on four GPUs with global batch 16."""
import argparse
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import sys
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from ddp_runtime import GlobalBatchShard, digest, patch_method, reduction_witness


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--mode',choices=['probe','full'],required=True)
    a=ap.parse_args()
    rank=int(os.environ['RANK']); local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    dist.init_process_group('nccl')
    assert dist.get_world_size()==4
    torch.set_num_threads(1)
    witnesses=reduction_witness()
    arm=a.root/('probe_arm' if a.mode=='probe' else 'runs/cot_aux')
    expected=json.loads((arm/'RESUME_INPUT.json').read_text())
    src=Path(expected['source_attempt']); base=src.parent
    prep=json.loads((src/'prepared_inputs.json').read_text())
    for path in (base/'simvp_code_snapshot',base/'code_pipeline_attempt13',src):
        sys.path.insert(0,str(path))
    from pytorch_lightning import Trainer, LightningDataModule, Callback, seed_everything
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
    from pytorch_lightning.loggers import CSVLogger
    from pytorch_lightning.strategies import DDPStrategy
    from openstl.methods.simvp import SimVP
    from openstl.datasets.dataloader_hbagri import HBAGRIDataset
    from openstl.utils.callbacks import BestCheckpointCallback, EpochEndCallback
    from train_cot_repaired import COTUNet
    import cot_aux_patch_attempt13 as old
    seed_everything(42,workers=True)
    old.patch_dataset_index(HBAGRIDataset)
    norm=np.load(prep['S_norm'])
    def stats(ds):
        for name in ('mean','std','input_mean','input_std'):
            k={'mean':'agri_mean','std':'agri_std'}.get(name,name)
            setattr(ds,name,np.asarray(norm[k],dtype=np.float32).copy())
    HBAGRIDataset._compute_stats=stats
    datasets={s:HBAGRIDataset(data_root=prep['S_cache'],split=s,index_csv=prep['index']) for s in ('train','val')}
    stats(datasets['val'])
    assert len(datasets['train'])==4086 and len(datasets['val'])==407
    samplers={s:GlobalBatchShard(len(ds),rank,4,s=='train') for s,ds in datasets.items()}
    loaders={s:DataLoader(ds,batch_sampler=samplers[s],num_workers=2,pin_memory=True) for s,ds in datasets.items()}
    assert len(loaders['train'])==256 and len(loaders['val'])==26
    class Data(LightningDataModule):
        def train_dataloader(self): return loaders['train']
        def val_dataloader(self): return loaders['val']
    cfg={k:v for k,v in runpy.run_path(str(arm/'source_config.py')).items() if not k.startswith('__')}
    cfg.update(batch_size=4,val_batch_size=4,gpus=[0,1,2,3],dist=True,
               steps_per_epoch=256,test_mean=datasets['train'].mean,test_std=datasets['train'].std,
               save_dir=str(arm))
    patch_method(SimVP)
    method=SimVP(**cfg).to(torch.device('cuda',local))
    r=COTUNet(); r.load_state_dict(torch.load(prep['r_checkpoint'],map_location='cpu')['model'])
    old.attach_state(method,r_model=r,s_norm_path=Path(prep['S_norm']),r_norm_path=Path(prep['r_norm']),
                     sidecar_dir=Path(prep['sidecar']),mode='cot_aux',lambda_path=src/'lambda/train_only_lambda.json',
                     r_checkpoint=Path(prep['r_checkpoint']))
    old.initialize_lambda_or_load(method,datasets['train'],src/'lambda/train_only_lambda.json')
    class Audit(Callback):
        def __init__(self): self.times=[]; self.intervals=[]; self.last_end=None
        def on_train_start(self,tr,mod):
            actual={'model':digest(mod.state_dict()),'optimizer':digest(tr.optimizers[0].state_dict()),
                    'scheduler':digest(tr.lr_scheduler_configs[0].scheduler.state_dict()),'global_step':tr.global_step}
            for key in actual:
                if actual[key]!=expected[key]: raise RuntimeError(f'Resume mismatch {key}: {actual[key]} != {expected[key]}')
            actual.update(state='PASS_FULL_MODEL_OPTIMIZER_SCHEDULER_RESTORE',epoch=tr.current_epoch,
                          world_size=4,global_batch=16,steps_per_epoch=256,reduction_witness=witnesses)
            if tr.is_global_zero:
                (arm/'RESTORE_PASS.json').write_text(json.dumps(actual,indent=2))
                print('RESTORE_PASS',json.dumps(actual),flush=True)
            samplers['train'].set_epoch(tr.current_epoch)
        def on_train_batch_start(self,tr,mod,batch,batch_idx):
            torch.cuda.synchronize(); self.start=time.perf_counter()
            if batch_idx==0:
                ids=[None]*4; dist.all_gather_object(ids,batch[3].cpu().tolist())
                flat=[v for row in ids for v in row]
                assert len(flat)==16 and len(set(flat))==16
        def on_train_batch_end(self,tr,mod,outputs,batch,batch_idx):
            torch.cuda.synchronize(); now=time.perf_counter();self.times.append(now-self.start)
            if self.last_end is not None: self.intervals.append(now-self.last_end)
            self.last_end=now
            if tr.is_global_zero and (batch_idx==0 or (batch_idx+1)%32==0):
                (arm/'TRAIN_PROGRESS.json').write_text(json.dumps(dict(epoch=tr.current_epoch,
                    global_step=tr.global_step,local_batch_index=batch_idx,world_size=4,global_batch=16,
                    recent_step_seconds=float(np.median(self.intervals[-32:])) if self.intervals else None,
                    recorded_unix=time.time()),indent=2))
            if a.mode=='probe' and len(self.times)==8: tr.should_stop=True
        def on_train_epoch_start(self,tr,mod):
            samplers['train'].set_epoch(tr.current_epoch)
        def on_before_optimizer_step(self,tr,mod,optimizer,optimizer_idx):
            if tr.global_step==expected['global_step']:
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in mod.parameters())
                assert all(p.grad is None for p in mod.__dict__['_attempt13_cot_state']['r_model'].parameters())
    cb=BestCheckpointCallback(monitor='val_cot_rmse',filename='best-{epoch:02d}-{val_loss:.3f}',
                             mode='min',save_last=True,dirpath=str(arm/'checkpoints'),every_n_epochs=1,verbose=True)
    class FitEpochEnd(EpochEndCallback):
        def on_validation_epoch_end(self,tr,mod):
            if tr.training: super().on_validation_epoch_end(tr,mod)
    audit=Audit()
    callbacks=[cb,FitEpochEnd(),EarlyStopping(monitor='val_cot_rmse',patience=100,mode='min'),
               LearningRateMonitor(logging_interval=None),audit]
    kwargs=dict(accelerator='gpu',devices=4,strategy=DDPStrategy(find_unused_parameters=False),
                sync_batchnorm=True,replace_sampler_ddp=False,max_epochs=50,num_sanity_val_steps=0,
                callbacks=callbacks,logger=CSVLogger(str(arm/'cwd')),log_every_n_steps=50,
                max_steps=-1)
    trainer=Trainer(**kwargs)
    start=time.perf_counter()
    if a.mode=='probe':
        score=trainer.validate(method,datamodule=Data(),ckpt_path=str(arm/'resume.ckpt'),verbose=False)[0]['val_cot_rmse']
        assert abs(score-expected['serial_val_cot_rmse']) <= max(.002,abs(score)*.0002), (score,expected['serial_val_cot_rmse'])
        if rank==0:
            (arm/'VALIDATION_REDUCTION_PASS.json').write_text(json.dumps(dict(
                serial=expected['serial_val_cot_rmse'],ddp=score,absolute_difference=abs(score-expected['serial_val_cot_rmse']))))
    trainer.fit(method,datamodule=Data(),ckpt_path=str(arm/'resume.ckpt'))
    elapsed=time.perf_counter()-start
    if rank==0:
        if a.mode=='probe':
            assert trainer.global_step==expected['global_step']+8
            result=dict(state='PASS_DDP4_RESUME_PROBE',updates=8,global_step=trainer.global_step,
                        source_global_step=expected['global_step'],global_batch=16,
                        wall_seconds=elapsed,train_batch_seconds=audit.times,
                        warm_batch_median_seconds=float(np.median(audit.times[3:])),
                        warm_end_to_end_step_median_seconds=float(np.median(audit.intervals[2:])),
                        reduction_witness=witnesses,test_payloads_read=0,buffer_payloads_read=0)
            (a.root/'PROBE_PASS.json').write_text(json.dumps(result,indent=2))
        else:
            assert trainer.global_step==12800 and trainer.current_epoch==50
            best=Path(cb.best_model_path); assert best.is_file()
            if best!=arm/'checkpoints/best.ckpt': shutil.copy2(best,arm/'checkpoints/best.ckpt')
            initial=json.loads((arm/'attempt13_init_receipt.json').read_text())
            meta=dict(initial)
            meta.update(state='ATTEMPT13_SIMVP_COT_AUX_ARM_COMPLETE',arm='cot_aux',completed_epochs=50,
                      configured_max_epochs=50,global_updates=12800,world_size=4,global_batch=16,micro_batch=4,
                      best_checkpoint=str(arm/'checkpoints/best.ckpt'),callback_best_source=str(best),
                      callback_best_score=float(cb.best_model_score),migration=expected,
                      selection_metric='equal-lead global validation physical COT RMSE',
                      leadwise_val_cot_rmse_physical=method.__dict__['_attempt13_last_val_lead_rmse'],
                      role='full',solar_geometry_corrected=True,continued_training=True,
                      frozen_r_checkpoint=prep['r_checkpoint'],cot_target_sidecar=prep['sidecar'],
                      source_config=str(arm/'source_config.py'),
                      test_payloads_read=0,buffer_payloads_read=0)
            # Tensor hashes above attest restoration; file hashes bind inference assets.
            import hashlib
            meta['best_checkpoint_sha256']=hashlib.sha256((arm/'checkpoints/best.ckpt').read_bytes()).hexdigest()
            (arm/'run_metadata.json').write_text(json.dumps(meta,indent=2))
            pair=json.loads((src/'PAIR_STARTED_attempt13.json').read_text())
            control=json.loads((src/'runs/mse_control/run_metadata.json').read_text())
            assert control['global_updates']==meta['global_updates']==12800
            assert control['initial_simvp_state_sha256']==meta['initial_simvp_state_sha256']
            pair.update(state='ATTEMPT13_PAIR_COMPLETE',completed_arms={'mse_control':{'run_metadata':control},'cot_aux':{'run_metadata':meta}},
                        same_initial_state=True,same_optimizer_updates=True,
                        execution_change='COT arm resumed on 4 A800; global batch 16; SyncBatchNorm and global mask reductions',
                        test_payloads_read=0,buffer_payloads_read=0)
            (a.root/'PAIR_COMPLETE_attempt13.json').write_text(json.dumps(pair,indent=2))
            print('DDP4_PAIR_COMPLETE',flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__=='__main__': main()
