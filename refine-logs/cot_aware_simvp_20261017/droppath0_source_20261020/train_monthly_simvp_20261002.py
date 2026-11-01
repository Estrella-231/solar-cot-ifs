"""Isolated monthly-block SimVP: fresh normalization, random initialization."""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


class RawCache:
    def __init__(self,source,root,cache):
        sys.path.insert(0,str(source))
        from hunan_data import load_frame
        self.load_frame=load_frame;self.root=root;self.cache=Path(cache)

    def frame(self,rel):
        import fcntl
        dst=self.cache/(rel+'.npz')
        if dst.exists():
            with np.load(dst,allow_pickle=False) as z:return z['a'],z['v'],z['g']
        dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.with_suffix('.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if dst.exists():
                with np.load(dst,allow_pickle=False) as z:return z['a'],z['v'],z['g']
            if shutil.disk_usage(self.cache).free < 25*2**30:
                raise RuntimeError('node-local disk reserve below 25 GiB; refusing cache growth')
            a,v,g=self.load_frame(self.root,rel)
            tmp=dst.with_name(dst.name+f'.{os.getpid()}.tmp')
            with tmp.open('wb') as f:np.savez(f,a=a,v=v,g=g)
            tmp.replace(dst)
            return a,v,g


def frame_moments(task):
    source,root,cache,rel=task
    a,v,_=RawCache(source,root,cache).frame(rel)
    x=np.where(v,a,0).astype(np.float64)
    return v.sum((1,2)),x.sum((1,2)),(x*x).sum((1,2))


def normalize(a):
    manifest=Path(a.manifests)/'hunan_8to16.csv'
    audit=json.loads((Path(a.manifests)/'manifest_audit.json').read_text())
    assert audit['manifest_sha256']==sha(manifest) and audit['frame_disjoint_splits']
    train_list=Path(a.manifests)/'train_frames.txt'
    assert audit['frame_list_sha256']['train']==sha(train_list)
    path=Path(a.manifests)/'hunan_train_norm.json'
    if path.exists():
        norm=json.loads(path.read_text());assert norm['manifest_sha256']==sha(manifest)
        assert norm['train_frames_sha256']==sha(train_list)
        return norm
    rels=(Path(a.manifests)/'train_frames.txt').read_text().splitlines()
    Path(a.cache).mkdir(parents=True,exist_ok=True)
    required=len(rels)*6*2**20+40*2**30
    assert shutil.disk_usage(a.cache).free > required, 'insufficient space for training raw-frame cache'
    n=np.zeros(13,np.int64);s=np.zeros(13,np.float64);q=s.copy()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        tasks=((a.source,a.root,a.cache,r) for r in rels)
        for i,(ni,si,qi) in enumerate(pool.map(frame_moments,tasks,chunksize=16),1):
            n+=ni;s+=si;q+=qi
            if i%500==0:print('NORMALIZATION',i,len(rels),flush=True)
    assert (n>0).all()
    mean=s/n;std=np.sqrt(np.maximum(q/n-mean**2,1e-8))
    assert np.isfinite(mean).all() and np.isfinite(std).all()
    norm={'manifest_sha256':sha(manifest),'source_indices':[0,1,2,3,4,5,8,9,10,11,12,13,14],
          'train_frames_sha256':sha(train_list),
          'mean':mean.tolist(),'std':std.tolist(),'count':n.tolist(),
          'train_unique_frames':len(rels),'validation_test_used':False,
          'validity':'channel QA bits AND finite; valid nighttime zero retained',
          'geometry':'raw cosSOZ,cosRAA,day_mask; no fitted normalization'}
    write_json(path,norm);print('NORMALIZATION_DONE',flush=True);return norm


class MonthlyDataset(Dataset):
    def __init__(self,a,norm,split):
        assert split in ('train','val'), 'trainer/profile must never read test payloads'
        with (Path(a.manifests)/'hunan_8to16.csv').open(newline='') as f:
            self.rows=[r for r in csv.DictReader(f) if r['split']==split]
        self.raw=RawCache(a.source,a.root,a.cache)
        self.mean=np.array(norm['mean'],np.float32)[:,None,None]
        self.std=np.array(norm['std'],np.float32)[:,None,None]
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        paths=self.rows[i]['data_relpaths'].split('|')
        assert len(paths)==24
        images=[];masks=[];geometry=[]
        for p in paths:
            x,v,g=self.raw.frame(p)
            images.append(np.where(v,(x-self.mean)/self.std,0).astype(np.float32))
            masks.append(v);geometry.append(g)
        t=datetime.strptime(self.rows[i]['BJT_start'],'%Y-%m-%d %H:%M:%S')+timedelta(minutes=105)
        return np.stack(images[:8]),np.stack(geometry),np.stack(images[8:]),np.stack(masks[8:]),t.month


def seed(value=42):
    random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)


def loss_fn(pred,truth,valid):
    assert pred.shape==truth.shape==valid.shape
    return ((pred.float()-truth.float()).square()*valid).sum()/valid.sum().clamp_min(1)


def model(a):
    sys.path.insert(0,str(a.source))
    from triad_models import SimVPBackbone
    return SimVPBackbone(a.source).cuda()


def profile(a,norm):
    path=Path(a.output)/'profile.json'
    assert not path.exists()
    data=MonthlyDataset(a,norm,'train'); trials=[]
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    seed();w=torch.randn(64,64,device='cuda');z=w@w
    assert torch.isfinite(z).all()
    print('CUDA_WITNESS',torch.cuda.get_device_name(0),list(z.shape),flush=True)
    for b in (4,8,12,16,20,24,32):
        net=opt=batch=pred=loss=None
        seed();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        try:
            # Real, distinct training rows; outputs use all 16 target leads.
            samples=[data[(i*137)%len(data)] for i in range(b)]
            batch=[torch.from_numpy(np.stack([s[j] for s in samples])).cuda() for j in range(4)]
            net=model(a).train();opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=1e-4)
            durations=[]
            for k in range(5):
                torch.cuda.synchronize();start=time.monotonic();opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=net(batch[0],batch[1]);loss=loss_fn(pred,batch[2],batch[3])
                assert torch.isfinite(loss);loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
                torch.cuda.synchronize();durations.append(time.monotonic()-start)
            peak=torch.cuda.max_memory_allocated();total=torch.cuda.get_device_properties(0).total_memory
            rec={'batch':b,'peak_gib':peak/2**30,'seconds_per_step':sum(durations[2:])/3,
                 'samples_per_second':b/(sum(durations[2:])/3),'within_85pct_memory':peak<=.85*total,
                 'loss':float(loss)}
            trials.append(rec);print('PROFILE_BATCH',json.dumps(rec),flush=True)
            if peak>.85*total:break
        except torch.cuda.OutOfMemoryError:
            trials.append({'batch':b,'status':'OOM'});break
        finally:
            del net,opt,batch,pred,loss;torch.cuda.empty_cache()
    good=[r for r in trials if r.get('within_85pct_memory')]
    assert good,'no stable profiled batch'
    fastest=max(r['samples_per_second'] for r in good)
    chosen=max((r for r in good if r['samples_per_second']>=.95*fastest),key=lambda r:r['batch'])
    # Exact checkpoint reload witness under the selected architecture/runtime.
    seed();net=model(a).eval();sample=data[0]
    x,g=[torch.from_numpy(sample[j]).unsqueeze(0).cuda() for j in (0,1)]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):before=net(x,g).float()
    witness=Path(a.output)/'profile_reload.pt';torch.save(net.state_dict(),witness)
    clone=model(a).eval();clone.load_state_dict(torch.load(witness,map_location='cpu',weights_only=False),strict=True)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):after=clone(x,g).float()
    delta=float((after-before).abs().max());assert delta==0
    result={'status':'PASS','selected_batch':chosen['batch'],'trials':trials,
            'selection':'largest batch within 5% of peak compute throughput, <=85% GPU memory',
            'reload_output_max_delta':delta,'test_payloads_read':0,
            'device':torch.cuda.get_device_name(0),'manifest_sha256':sha(Path(a.manifests)/'hunan_8to16.csv')}
    write_json(path,result);print('PROFILE_PASS',json.dumps(result),flush=True)


def train(a,norm):
    output=Path(a.output)/'training';output.mkdir(exist_ok=False)
    batch=json.loads((Path(a.output)/'profile.json').read_text())['selected_batch']
    seed();torch.backends.cudnn.benchmark=True
    tr=MonthlyDataset(a,norm,'train');va=MonthlyDataset(a,norm,'val');net=model(a)
    opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=1e-4)
    meta={'region':'Hunan','split_id':'hunan_monthly_block_2d_buffer1d_seed42_v1',
          'initialization':'random','seed':42,'epochs_fixed':12,'batch':batch,
          'train_samples':len(tr),'val_samples':len(va),'test_payloads_read':0,
          'geometry_contract':'history_8_then_target_16_v2','input_agri_channels':13,
          'calendar_features_added':False,'history_frames':8,'target_frames':16,
          'training_loss':'pooled-valid-pixel normalized AGRI MSE',
          'selection_metric':'equal-four-season mean validation pixel RMSE; DJF/MAM/JJA/SON',
          'manifest_sha256':sha(Path(a.manifests)/'hunan_8to16.csv'),
          'normalization_sha256':sha(Path(a.manifests)/'hunan_train_norm.json'),
          'model_code_sha256':sha(Path(a.source)/'triad_models.py'),
          'data_code_sha256':sha(Path(a.source)/'hunan_data.py'),
          'simvp_code_sha256':sha(Path(a.source)/'simvp.py'),
          'training_code_sha256':sha(__file__),'pbs':os.environ.get('PBS_JOBID'),
          'node_cache':a.cache,'old_checkpoint_loaded':False}
    write_json(output/'run_metadata.json',meta)
    best=float('inf');step=0
    season_of={12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}
    for epoch in range(12):
        started=time.monotonic();net.train();losses=[]
        loader=DataLoader(tr,batch_size=batch,shuffle=True,generator=torch.Generator().manual_seed(42+epoch),
                          num_workers=a.workers,pin_memory=True,persistent_workers=True,prefetch_factor=1)
        for k,(x,g,y,v,_) in enumerate(loader):
            x,g,y,v=[z.cuda(non_blocking=True) for z in (x,g,y,v)]
            progress=(epoch+k/len(loader))/12
            opt.param_groups[0]['lr']=1e-6+.5*(1e-4-1e-6)*(1+math.cos(math.pi*progress))
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=loss_fn(net(x,g),y,v)
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            opt.step();losses.append(float(loss));step+=1
            if step%100==0:print('TRAIN_STEP',json.dumps({'epoch':epoch+1,'step':step,'loss':losses[-1]}),flush=True)
        net.eval();sse=np.zeros(12,np.float64);count=sse.copy()
        with torch.no_grad():
            for x,g,y,v,months in DataLoader(va,batch_size=batch,num_workers=a.workers,pin_memory=True,
                                            persistent_workers=True,prefetch_factor=1):
                x,g,y,v=[z.cuda(non_blocking=True) for z in (x,g,y,v)]
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=net(x,g)
                sums=(((pred.float()-y.float()).square()*v).sum((1,2,3,4)).double()).cpu().numpy()
                ns=v.sum((1,2,3,4)).cpu().numpy()
                for m,e,n in zip(months.tolist(),sums,ns):sse[m-1]+=e;count[m-1]+=n
        season_sse=np.zeros(4);season_count=np.zeros(4)
        for m in range(1,13):season_sse[season_of[m]]+=sse[m-1];season_count[season_of[m]]+=count[m-1]
        assert (season_count>0).all()
        season_rmse=np.sqrt(season_sse/season_count);score=float(season_rmse.mean())
        rec={'epoch':epoch+1,'step':step,'train_loss_mean':float(np.mean(losses)),
             'val_pixel_rmse':float(np.sqrt(sse.sum()/count.sum())),
             'val_equal_season_rmse':score,'season_rmse':dict(zip(('DJF','MAM','JJA','SON'),season_rmse.tolist())),
             'monthly_rmse':{str(m+1):float(np.sqrt(sse[m]/count[m])) for m in range(12) if count[m]>0},
             'elapsed_s':time.monotonic()-started}
        write_json(output/f'epoch_{epoch+1:03d}.json',rec)
        payload={'model':net.state_dict(),'optimizer':opt.state_dict(),'metadata':meta,'epoch':epoch+1,'step':step,'best':min(best,score)}
        torch.save(payload,output/'last.tmp');(output/'last.tmp').replace(output/'last.pt')
        if score<best:
            best=score;torch.save(payload,output/'best.tmp');(output/'best.tmp').replace(output/'best.pt')
        print('EPOCH_END',json.dumps(rec),flush=True)
    write_json(output/'training_complete.json',{'epochs':12,'steps':step,'best_equal_season_val_rmse':best,
                                               'test_payloads_read':0,'best_checkpoint_sha256':sha(output/'best.pt')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('normalize','profile','train'),required=True)
    for name in ('root','source','cache','manifests','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args();Path(a.output).mkdir(parents=True,exist_ok=True)
    if a.mode=='normalize':normalize(a)
    else:
        norm=json.loads((Path(a.manifests)/'hunan_train_norm.json').read_text())
        assert norm['manifest_sha256']==sha(Path(a.manifests)/'hunan_8to16.csv')
        assert norm['train_frames_sha256']==sha(Path(a.manifests)/'train_frames.txt')
        if a.mode=='profile':profile(a,norm)
        else:train(a,norm)
