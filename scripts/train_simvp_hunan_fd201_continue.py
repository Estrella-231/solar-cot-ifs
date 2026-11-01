"""FD-201 Hunan gSTA SimVP continuation from the audited academy checkpoint."""
import argparse,hashlib,json,math,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from hunan_data import HunanDataset
from simvp import SimVP


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=SimVP(input_channels=16,output_channels=13,input_frames=8,
            hid_S=32,hid_T=256,N_S=4,N_T=8,mlp_ratio=8.0,drop=0.0,
            drop_path_prob=0.1,spatial_kernel=3,gsta_kernel=21)
    def forward(self,x,g):
        first=self.net(torch.cat((x,g[:,:8]),2))
        second=self.net(torch.cat((first,g[:,8:16]),2))
        return torch.cat((first,second),1)


def seed():
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)


def loss_fn(pred,y,v):
    count=v.sum((0,1,3,4))
    if not (count>0).any():raise ValueError('empty valid batch')
    channel=((pred.float()-y.float()).square()*v).sum((0,1,3,4))/count.clamp_min(1)
    return channel[count>0].mean()


def save(path,payload):
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--manifests',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--init-checkpoint',type=Path,required=True);p.add_argument('--init-sha256',required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    normpath=a.manifests/'hunan_train_norm.json';manifest=a.manifests/'hunan_8to16.csv'
    norm=json.loads(normpath.read_text())
    manifest_sha=hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert norm['manifest_sha256']==manifest_sha
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    seed();torch.backends.cudnn.benchmark=True
    train=HunanDataset(a.root,manifest,norm,'train');val=HunanDataset(a.root,manifest,norm,'val')
    profilepath=a.output/'profile.json'
    if not profilepath.exists():
        sample=train[0];profile=[];chosen=0
        free_before,total_memory=torch.cuda.mem_get_info()
        safe_limit=0.80*free_before
        for b in (1,2,4,8,16,32):
            model=None;opt=None;batch=None;pred=None;loss=None
            try:
                seed();model=Model().cuda();opt=torch.optim.Adam(model.parameters(),lr=1e-3)
                batch=[torch.from_numpy(x).unsqueeze(0).repeat(b,1,1,1,1).cuda() for x in sample]
                torch.cuda.reset_peak_memory_stats();start=time.monotonic()
                for _ in range(3):
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        pred=model(batch[0],batch[1]);loss=loss_fn(pred,batch[2],batch[3])
                    if not torch.isfinite(loss):raise ValueError('nonfinite GPU witness')
                    loss.backward();opt.step()
                torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()
                entry={'batch':b,'peak_gib':peak/2**30,'seconds_per_step':(time.monotonic()-start)/3,
                       'free_before_gib':free_before/2**30,'safe_limit_gib':safe_limit/2**30,
                       'loss':float(loss.detach()),'device':torch.cuda.get_device_name()}
                profile.append(entry);print('WITNESS',json.dumps(entry),flush=True)
                if peak<safe_limit:chosen=b
                else:break
            except torch.cuda.OutOfMemoryError:
                profile.append({'batch':b,'status':'OOM'});break
            finally:
                del model,opt,batch,pred,loss;torch.cuda.empty_cache()
        if chosen==0:raise RuntimeError('no safe training batch')
        profilepath.write_text(json.dumps({'selected_batch':chosen,'trials':profile},indent=2))
    b=json.loads(profilepath.read_text())['selected_batch']
    seed();model=Model().cuda()
    init_sha=hashlib.sha256(a.init_checkpoint.read_bytes()).hexdigest()
    if init_sha != a.init_sha256: raise RuntimeError('initial checkpoint hash drift')
    initial=torch.load(a.init_checkpoint,map_location='cpu')
    initial_state=initial.get('model',initial.get('state_dict')) if isinstance(initial,dict) else None
    if initial_state is None: raise RuntimeError('checkpoint lacks model state')
    model.load_state_dict(initial_state,strict=True);opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    metadata={'region':'Hunan','model':'gSTA_SimVP16_recurrent_8to8to16','initialization':'academy_checkpoint_continuation',
        'pretrained_checkpoint':str(a.init_checkpoint),'pretrained_checkpoint_sha256':init_sha,'train_samples':len(train),'val_samples':len(val),'batch':b,
        'max_epochs':a.epochs,'manifest_sha256':manifest_sha,
        'normalization_sha256':hashlib.sha256(normpath.read_bytes()).hexdigest(),
        'pbs_job_id':os.environ.get('PBS_JOBID'),'parameters':sum(x.numel() for x in model.parameters()),
        'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
        'code_sha256':{n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                       for n in (Path(__file__).name,'hunan_data.py','simvp.py')}}
    ck=a.output/'last.pt';epoch0=0;batch0=0;step=0;best=float('inf');wait=0
    if ck.exists():
        c=torch.load(ck,map_location='cpu')
        for k in ('manifest_sha256','normalization_sha256','code_sha256','batch','max_epochs'):
            assert c['metadata'][k]==metadata[k], 'resume contract changed: '+k
        model.load_state_dict(c['model']);opt.load_state_dict(c['optimizer'])
        epoch0=c['epoch'];batch0=c['next_batch'];step=c['step'];best=c['best'];wait=c['wait']
        torch.set_rng_state(c['rng']);torch.cuda.set_rng_state_all(c['cuda_rng'])
    (a.output/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
    print('FULL_TRAIN_START',json.dumps(metadata),flush=True)
    def checkpoint(epoch,next_batch):
        return {'model':model.state_dict(),'optimizer':opt.state_dict(),'metadata':metadata,
                'epoch':epoch,'next_batch':next_batch,'step':step,'best':best,'wait':wait,
                'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
    val_loader=DataLoader(val,batch_size=b,num_workers=a.workers,pin_memory=True,
                          persistent_workers=True,prefetch_factor=1)
    for epoch in range(epoch0,a.epochs):
        gen=torch.Generator().manual_seed(42+epoch)
        loader=DataLoader(train,batch_size=b,shuffle=True,generator=gen,num_workers=a.workers,
                          pin_memory=True,persistent_workers=True,prefetch_factor=1)
        model.train();started=time.monotonic()
        for j,batch in enumerate(loader):
            if epoch==epoch0 and j<batch0:continue
            x,g,y,v=[z.cuda(non_blocking=True) for z in batch]
            progress=(epoch+j/len(loader))/a.epochs
            lr=1e-6+0.5*(1e-3-1e-6)*(1+math.cos(math.pi*progress))
            if epoch<5:lr*=min(1,(epoch+(j+1)/len(loader))/5)
            for group in opt.param_groups:group['lr']=lr
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=loss_fn(model(x,g),y,v)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            if not torch.isfinite(grad):raise RuntimeError('nonfinite gradient')
            opt.step();step+=1
            if step%10==0 or step==1:print(json.dumps({'event':'train_step','epoch':epoch,'step':step,
                'batch':j,'batches_per_epoch':len(loader),'loss':float(loss.detach()),'lr':lr}),flush=True)
            if step%300==0:save(ck,checkpoint(epoch,j+1))
        model.eval();s=torch.zeros(13,device='cuda',dtype=torch.float64);n=s.clone()
        with torch.no_grad():
            for batch in val_loader:
                x,g,y,v=[z.cuda(non_blocking=True) for z in batch]
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(x,g)
                s+=((pred.float()-y.float()).square()*v).sum((0,1,3,4)).double()
                n+=v.sum((0,1,3,4)).double()
        score=float(torch.sqrt(s/n.clamp_min(1)).mean())
        if not math.isfinite(score):raise RuntimeError('nonfinite validation')
        improved=score<best;best=min(best,score);wait=0 if improved else wait+1
        record={'epoch':epoch,'step':step,'val_rmse':score,'best':best,'elapsed':time.monotonic()-started}
        print('EPOCH_END',json.dumps(record),flush=True)
        (a.output/f'epoch_{epoch:03d}.json').write_text(json.dumps(record,indent=2))
        save(ck,checkpoint(epoch+1,0))
        if improved:save(a.output/'best.pt',checkpoint(epoch+1,0))
        if wait>=20:break
    (a.output/'training_complete.json').write_text(json.dumps({'best_val_rmse':best,'steps':step}))


if __name__=='__main__':main()

