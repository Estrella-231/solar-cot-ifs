"""Pilot train a causal-input transport plus COT-thickness correction model.
Targets are CPP retrieval labels during training/validation; inputs use only historical
R(COT), history-derived local motion, and frozen SimVP->R forecast COT.
"""
import argparse,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self,c):
        super().__init__();self.c1=nn.Conv3d(c,c,3,padding=1);self.c2=nn.Conv3d(c,c,3,padding=1)
    def forward(self,x):return F.silu(x+self.c2(F.silu(self.c1(x))))

class TransportThicknessNet(nn.Module):
    def __init__(self,width=24):
        super().__init__();self.inp=nn.Conv3d(8,width,3,padding=1);self.b1=ResidualBlock(width);self.b2=ResidualBlock(width);self.out=nn.Conv3d(width,1,3,padding=1)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
    def forward(self,x,adv):
        delta=self.out(self.b2(self.b1(F.silu(self.inp(x)))))
        return torch.clamp(adv+delta,min=0,max=float(np.log1p(100)))

def warp_batch(base, motion):
    # base [B,1,H,W] in log1p COT; velocity [B,2] pixels / 15 min.
    b,_,h,w=base.shape;device=base.device
    yy,xx=torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device),indexing='ij')
    fields=[];supports=[]
    for lead in range(1,17):
        sy=yy[None]-lead*motion[:,0,None,None];sx=xx[None]-lead*motion[:,1,None,None]
        grid=torch.stack((2*sx/(w-1)-1,2*sy/(h-1)-1),-1)
        fields.append(F.grid_sample(torch.expm1(base),grid,mode='bilinear',padding_mode='zeros',align_corners=True))
        supports.append((sy>=0)&(sy<=h-1)&(sx>=0)&(sx<=w-1))
    adv=torch.cat(fields,dim=1) # [B,T,H,W] COT
    adv=torch.log1p(torch.clamp(adv,min=0))[:,None] # [B,1,T,H,W]
    support=torch.stack(supports,dim=1)[:,None]
    return adv,support

def make_input(history,forecast,adv,motion):
    # Shapes: history [B,8,1,H,W], forecast [B,16,1,H,W], adv [B,1,16,H,W]
    hlast=history[:,-1].unsqueeze(2)
    hprev=history[:,-2].unsqueeze(2)
    hlast=hlast.expand(-1,-1,16,-1,-1);trend=(hlast-hprev.expand_as(hlast))
    fc=forecast.permute(0,2,1,3,4)
    lead=torch.linspace(1/16,1,16,device=forecast.device)[None,None,:,None,None].expand(forecast.shape[0],1,-1,forecast.shape[-2],forecast.shape[-1])
    mot=(motion/4)[:,:,None,None,None].expand(-1,-1,16,forecast.shape[-2],forecast.shape[-1])
    return torch.cat((adv,fc,hlast,trend,fc-adv,lead,mot),dim=1)

def load_split(root,split):
    d=root/split
    return {k:np.load(d/(k+'.npy'),mmap_mode='r') for k in ['cot_local_history','cot_local_forecast','cot_local_cpp','cot_local_cpp_mask','motion_local32']}

def prepare(root,split,ids,device):
    arr=load_split(root,split);n=len(ids)
    histories=[];forecasts=[];targets=[];masks=[];motions=[];advs=[];supports=[]
    for s in range(0,n,8):
        ii=ids[s:s+8]
        h=torch.tensor(np.asarray(arr['cot_local_history'][ii,0]),device=device)
        f=torch.tensor(np.asarray(arr['cot_local_forecast'][ii,0]),device=device)
        y=torch.tensor(np.asarray(arr['cot_local_cpp'][ii,0,:,0]),device=device)
        m=torch.tensor(np.asarray(arr['cot_local_cpp_mask'][ii,0,:,0])>0.5,device=device)
        v=torch.tensor(np.asarray(arr['motion_local32'][ii]),device=device)
        adv,sup=warp_batch(h[:,-1],v)
        histories.extend(torch.split(h,1));forecasts.extend(torch.split(f,1));targets.extend(torch.split(y,1));masks.extend(torch.split(m,1));motions.extend(torch.split(v,1));advs.extend(torch.split(adv,1));supports.extend(torch.split(sup,1))
    return {'h':torch.cat(histories),'f':torch.cat(forecasts),'y':torch.cat(targets),'mask':torch.cat(masks),'motion':torch.cat(motions),'adv':torch.cat(advs),'support':torch.cat(supports)}

def evaluate(model,d,idx):
    out={k:{'n':0,'abs':0.,'sq':0.,'thick_n':0,'thick_miss':0,'thick_sq':0.,'clear_n':0,'false10':0,'tp':0,'fp':0,'fn':0} for k in ['simvp','persistence','transport','learned']}
    model.eval()
    with torch.no_grad():
        for s in range(0,len(idx),8):
            ids=idx[s:s+8];h=d['h'][ids].to('cuda');f=d['f'][ids].to('cuda');y=d['y'][ids].to('cuda');mask=d['mask'][ids].to('cuda');motion=d['motion'][ids].to('cuda');adv=d['adv'][ids].to('cuda');sup=d['support'][ids].to('cuda')
            x=make_input(h,f,adv,motion);pred=model(x,adv)
            for name,z in [('simvp',f.permute(0,2,1,3,4)),('persistence',h[:,-1].unsqueeze(2).expand(-1,-1,16,-1,-1)),('transport',adv),('learned',pred)]:
                zc=torch.expm1(z[:,0]);yc=torch.expm1(y);valid=mask&sup[:,0]
                for b in range(len(ids)):
                    q=valid[b];t=yc[b];p=zc[b];a=out[name]
                    if not q.any():continue
                    e=p[q]-t[q];a['n']+=int(q.sum());a['abs']+=float(e.abs().sum());a['sq']+=float((e*e).sum())
                    thick=q&(t>=30);a['thick_n']+=int(thick.sum());a['thick_miss']+=int((thick&(p<10)).sum());a['thick_sq']+=float(((p-t)[thick]**2).sum())
                    clear=q&(t<5);a['clear_n']+=int(clear.sum());a['false10']+=int((clear&(p>=10)).sum())
                    pm=q&(p>=30);tm=q&(t>=30);a['tp']+=int((pm&tm).sum());a['fp']+=int((pm&~tm).sum());a['fn']+=int((~pm&tm).sum())
    result={}
    for n,a in out.items():
        result[n]={'mae':a['abs']/a['n'] if a['n'] else None,'rmse':(a['sq']/a['n'])**.5 if a['n'] else None,'thick_lead_records':a['thick_n'],'thick_misses_pred_lt10':a['thick_miss'],'thick_rmse':(a['thick_sq']/a['thick_n'])**.5 if a['thick_n'] else None,'clear_false10_rate':a['false10']/a['clear_n'] if a['clear_n'] else None,'thick_iou_cot30':a['tp']/(a['tp']+a['fp']+a['fn']) if a['tp']+a['fp']+a['fn'] else None}
    return result

def training_loss(prediction, target, mask):
    # prepare() already removes the singleton COT channel from target/mask.
    # Preserve all 16 target times; only prediction still has a channel axis.
    pred = prediction[:, 0]
    if pred.shape != target.shape or mask.shape != target.shape:
        raise ValueError(f'Loss shape mismatch: pred={pred.shape}, target={target.shape}, mask={mask.shape}')
    truth_c = torch.expm1(target)
    pred_c = torch.expm1(pred)
    weight = 1 + 2 * (truth_c >= 30).float() + (truth_c < 5).float()
    hub = F.smooth_l1_loss(pred, target, reduction='none')
    loss = (hub * weight * mask).sum() / torch.clamp((weight * mask).sum(), min=1)
    clear = mask & (truth_c < 5)
    false = F.relu(pred_c - 10)
    return loss + 0.015 * (false.square() * clear).sum() / torch.clamp(clear.sum(), min=1)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bank',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--epochs',type=int,default=40);ap.add_argument('--batch-size',type=int,default=8);a=ap.parse_args()
    torch.manual_seed(42);torch.cuda.manual_seed_all(42);np.random.seed(42);random.seed(42);torch.set_num_threads(2)
    a.output.mkdir(parents=True,exist_ok=False)
    print('PREPARE_TRAIN',flush=True);tr=prepare(a.bank,'train',np.arange(512),'cuda')
    print('PREPARE_VAL',flush=True);va=prepare(a.bank,'val',np.arange(512),'cuda')
    model=TransportThicknessNet().cuda();opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    best=float('inf');stale=0;history=[];started=time.time()
    for epoch in range(1,a.epochs+1):
        model.train();order=np.random.permutation(512);loss_sum=0.;steps=0
        for s in range(0,512,a.batch_size):
            ii=order[s:s+a.batch_size];h=tr['h'][ii].cuda();f=tr['f'][ii].cuda();y=tr['y'][ii].cuda();m=tr['mask'][ii].cuda();v=tr['motion'][ii].cuda();adv=tr['adv'][ii].cuda()
            x=make_input(h,f,adv,v);p=model(x,adv)
            loss=training_loss(p,y,m)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();loss_sum+=float(loss.detach());steps+=1
        val=evaluate(model,va,np.arange(512));val_score=val['learned']['thick_rmse'] if val['learned']['thick_rmse'] is not None else val['learned']['rmse']
        rec={'epoch':epoch,'train_loss':loss_sum/steps,'val':val,'elapsed_seconds':time.time()-started};history.append(rec)
        print(json.dumps(rec),flush=True)
        (a.output/'epochs.jsonl').open('a').write(json.dumps(rec)+'\n')
        if val_score<best:
            best=val_score;stale=0;torch.save({'model':model.state_dict(),'epoch':epoch,'seed':42,'val':val},a.output/'best.pt')
            (a.output/'best.json').write_text(json.dumps({'epoch':epoch,'score':best,'val':val},indent=2))
        else:stale+=1
        if stale>=8:break
    report={'state':'COMPLETE_EXPLORATORY_SILI_TRANSPORT_THICKNESS_PILOT','test_used':False,'bank':str(a.bank),'epochs':len(history),'best_epoch':json.loads((a.output/'best.json').read_text())['epoch'],'best_val':json.loads((a.output/'best.json').read_text())['val'],'selection':'fixed monthly stratification; train and validation are separate time splits','inputs':'historical R COT sequence, local32 seven-pair C13 phase motion, frozen SimVP->R COT forecast, lead','target':'CPP COT training/validation labels; CPP is retrieval reference, not independent ground truth','loss':'masked log1p-COT SmoothL1 weighted 3x for COT>=30, plus small clear false-COT penalty','notes':'validation metrics are exploratory; windows overlap; not paper/test evidence'}
    (a.output/'training_complete.json').write_text(json.dumps(report,indent=2));print('COMPLETE',json.dumps(report),flush=True)
if __name__=='__main__':main()
