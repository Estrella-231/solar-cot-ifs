"""Old-split Sili forecast-GHI diagnostic, reusing frozen historical-COT assets."""
import argparse, csv, hashlib, json, math, random, time
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import torch
from torch import nn
from train_sili_transport_thickness_pilot_20260927 import warp_batch
from train_sili_history_repeat_control_20260928 import HistoryControlNet, history_input


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()


def write(p,d):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');q.replace(p)


def seed(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(64,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,32,3,padding=1),nn.SiLU(),nn.Flatten(),nn.Linear(32*3*3,64),nn.SiLU())
        self.lead=nn.Embedding(16,8)
        self.read=nn.Sequential(nn.Linear(72,64),nn.SiLU(),nn.Linear(64,1),nn.Softplus())

    def forward(self,image,cot,support,lead,use_cot):
        prefix=(torch.arange(24,device=lead.device)[None]<=8+lead[:,None])[:,:,None,None]
        z=torch.where(prefix & support, cot, torch.zeros_like(cot))
        if not use_cot:z=torch.zeros_like(z)
        # Same support channels in A and AC; only COT values differ.
        x=torch.cat((image,z,(prefix & support).float()),1)
        return self.read(torch.cat((self.encoder(x),self.lead(lead)),1)).squeeze(1)


@torch.no_grad()
def prepare(a,split,model):
    source=a.cot_bank/split
    original_ids=np.load(source/'sequence_indices.npy')
    ids=original_ids[np.isin(original_ids,list(a.allowed_sequences[split]))]
    source_lookup={int(k):i for i,k in enumerate(original_ids)}
    image_dir=(a.train_images if split=='train' else a.val_images)/split
    image_ids=np.load(image_dir/'sequence_indices.npy')
    lookup={int(k):i for i,k in enumerate(image_ids)}
    assert set(map(int,ids))<=set(lookup),'Image bank lacks selected sequence keys'
    image=np.load(image_dir/'image_local.npy',mmap_mode='r')
    assert image.shape[1:]==(2,16,16,64,64)
    hh=np.load(source/'cot_local_history.npy',mmap_mode='r')
    ff=np.load(source/'cot_local_forecast.npy',mmap_mode='r')
    motion=np.load(source/'motion_local32.npy')
    history=[]; advs=[]; hybrids=[]; supports=[]; images=[]
    for lo in range(0,len(ids),64):
        ix=np.array([source_lookup[int(k)] for k in ids[lo:lo+64]])
        h=torch.tensor(np.asarray(hh[ix,0]),device='cuda')
        f=torch.tensor(np.asarray(ff[ix,0]),device='cuda')
        v=torch.tensor(motion[ix],device='cuda')
        adv,sup=warp_batch(h[:,-1],v)
        pred=model(history_input(h,f,adv,v,'full_history'),adv)
        history.append(h[:,:,0,31:34,31:34].cpu().numpy())
        advs.append(adv[:,0,:,31:34,31:34].cpu().numpy())
        hybrids.append(pred[:,0,:,31:34,31:34].cpu().numpy())
        supports.append(sup[:,0,:,31:34,31:34].cpu().numpy())
        images.append(np.stack([np.asarray(image[lookup[int(k)],0,:,:,31:34,31:34]) for k in ids[lo:lo+64]]))
        print('PREPARE',split,lo+len(ix),len(ids),flush=True)
    h=np.concatenate(history);ad=np.concatenate(advs);hy=np.concatenate(hybrids)
    support=np.concatenate((np.ones_like(h,dtype=bool),np.concatenate(supports)),1)
    rows=np.load(source/'head_row_ids.npy')
    seq_meta=np.load(a.base/'sequence.npy',mmap_mode='r')[rows]
    station_meta=np.load(a.base/'station.npy',mmap_mode='r')[rows]
    rows=rows[np.isin(seq_meta,ids)&(station_meta==0)]
    labels={k:np.asarray(np.load(a.base/(k+'.npy'),mmap_mode='r')[rows]) for k in ('sequence','station','lead','ghi','clear','split','weight')}
    assert np.all(labels['split']==(0 if split=='train' else 1))
    assert np.isfinite(labels['ghi']).all() and np.isfinite(labels['clear']).all() and (labels['clear']>0).all()
    assert (labels['weight']>0).all()
    index={int(k):i for i,k in enumerate(ids)}
    seq=np.array([index[int(k)] for k in labels['sequence']])
    lead=labels['lead'].astype(np.int64)
    assert ((lead>=0)&(lead<16)).all()
    x=np.concatenate(images)[seq,lead]
    cot={'transport':np.concatenate((h,ad),1)[seq], 'hybrid':np.concatenate((h,hy),1)[seq]}
    mask=support[seq]
    assert np.isfinite(x).all() and all(np.isfinite(v).all() for v in cot.values())
    return dict(rows=rows,ids=ids,seq=seq,lead=lead,x=x,cot=cot,support=mask,ghi=labels['ghi'].astype(np.float64),clear=labels['clear'].astype(np.float64))


def metrics(pred,true):
    e=np.asarray(pred,dtype=np.float64)-np.asarray(true,dtype=np.float64)
    return dict(n=len(e),rmse=float(np.sqrt(np.mean(e*e))),mae=float(np.mean(abs(e))),bias=float(e.mean()))


def main():
    p=argparse.ArgumentParser()
    for n in ('cot-bank','train-images','val-images','base','cot-run','day-assignment','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--updates',type=int,default=1200)
    a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True);torch.set_num_threads(2)
    # Preserve old train/val roles, while excluding every window touching new test/buffer dates.
    with a.day_assignment.open(newline='',encoding='utf-8-sig') as f:
        assignments=list(csv.DictReader(f))
    date_key=next(k for k in assignments[0] if 'date' in k.lower() or 'day' in k.lower())
    split_key=next(k for k in assignments[0] if 'split' in k.lower())
    allowed_dates={r[date_key] for r in assignments if r[split_key] in ('train','val')}
    a.allowed_sequences={s:set() for s in ('train','val')}
    with (a.base/'rows.csv').open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['station']!='sili':continue
            it=datetime.fromisoformat(r['init_time_bjt'])
            dates={str((it+timedelta(minutes=k*15)).date()) for k in range(-7,17)}
            if dates<=allowed_dates:
                a.allowed_sequences[{'validation':'val'}.get(r['split'],r['split'])].add(int(r['cache_index']))
    meta=json.loads((a.cot_bank/'complete.json').read_text());assert meta['test_used'] is False
    assert meta['state']=='COMPLETE_SILI_COT_DYNAMICS_BANK'
    for im in (a.train_images,a.val_images):
        m=json.loads((im/'complete.json').read_text());assert m['test_used'] is False
        for key in ('simvp_checkpoint_sha256','r_checkpoint_sha256'):
            if key in meta and key in m: assert meta[key]==m[key],key
    c=json.loads((a.cot_run/'training_complete.json').read_text())
    assert c['best_learned_update_by_patch_thick_rmse']==96 and c['test_used'] is False
    state=torch.load(a.cot_run/'best_learned.pt',map_location='cpu',weights_only=False)
    assert state['update']==96
    model=HistoryControlNet().cuda().eval().requires_grad_(False);model.load_state_dict(state['model'])
    data={s:prepare(a,s,model) for s in ('train','val')};del model;torch.cuda.empty_cache()
    # Check identities/times against original row table; no future observations are loaded.
    wanted={int(r) for d in data.values() for r in d['rows']}; rowmap={}
    with (a.base/'rows.csv').open(newline='',encoding='utf-8') as f:
        for i,r in enumerate(csv.DictReader(f)):
            if i in wanted:rowmap[i]=r
    assert len(rowmap)==len(wanted)
    for split,d in data.items():
        dates=[];keys=[];init=[]
        for j,row in enumerate(d['rows']):
            r=rowmap[int(row)];assert {'validation':'val'}.get(r['split'],r['split'])==split and r['station']=='sili'
            it=datetime.fromisoformat(r['init_time_bjt']);tt=datetime.fromisoformat(r['target_time_bjt'])
            assert tt-it==timedelta(minutes=(int(d['lead'][j])+1)*15)
            assert abs(float(r['observed_ghi_15min_wm2'])-d['ghi'][j])<1e-3
            stamps=r['label_record_times_bjt'].split('|')
            assert [datetime.fromisoformat(t) for t in stamps]==[tt-timedelta(minutes=t) for t in (10,5,0)]
            dates.append(str(tt.date()));init.append(it.isoformat());keys.append(r['target_time_utc'])
        d['dates']=np.array(dates);d['init']=np.array(init);d['target']=np.array(keys)
    # Frozen COT normalization is fitted separately to each arm's identical train support.
    norms={}
    for arm in ('transport','hybrid'):
        v=data['train']['cot'][arm][data['train']['support']]
        norms[arm]=(float(v.mean(dtype=np.float64)),max(float(v.std(dtype=np.float64)),1e-6))
    np.savez_compressed(a.output/'paired_pack.npz',**{f'{s}_{k}':v for s,d in data.items() for k,v in d.items() if isinstance(v,np.ndarray)},**{f'{s}_cot_{arm}':d['cot'][arm] for s,d in data.items() for arm in ('transport','hybrid')})
    write(a.output/'source_audit.json',dict(state='PASS_OLD_TRAIN_VAL_PAIRED_KEYS',train_rows=len(data['train']['rows']),val_rows=len(data['val']['rows']),sequences={s:len(d['ids']) for s,d in data.items()},cot_checkpoint=sha(a.cot_run/'best_learned.pt'),cot_bank_manifest=meta,cot_norm_train_only=norms,input_mode='forecast AGRI+geometry; historical R-COT and history transport / frozen hybrid future COT; 3x3',test_payload_read=False,cpp_payload_read=False,oracle_read=False,scope='old Hunan split, Sili only, validation pilot'))
    # Train arrays fit on device; only local 3x3 arrays are retained.
    for d in data.values():
        d['tx']=torch.tensor(d['x'],device='cuda');d['tl']=torch.tensor(d['lead'],device='cuda');d['tm']=torch.tensor(d['support'],device='cuda')
        d['tc']={arm:torch.tensor((v-norms[arm][0])/norms[arm][1],device='cuda') for arm,v in d['cot'].items()}
        d['ty']=torch.tensor(d['ghi'],device='cuda',dtype=torch.float32);d['ts']=torch.tensor(d['clear'],device='cuda',dtype=torch.float32)
    tr,va=data['train'],data['val']
    def forward(m,d,ix,arm):return m(d['tx'][ix],d['tc']['transport' if arm=='A' else arm][ix],d['tm'][ix],d['tl'][ix],arm!='A')
    def update(m,opt,ix,arm):
        opt.zero_grad(set_to_none=True);pr=forward(m,tr,ix,arm)*tr['ts'][ix]
        loss=((pr-tr['ty'][ix])/1000).square().mean();assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);opt.step();return float(loss.detach())
    @torch.no_grad()
    def evaluate(m):
        out=[]
        for lo in range(0,len(va['rows']),2048):
            ix=torch.arange(lo,min(lo+2048,len(va['rows'])),device='cuda')
            out.append((forward(m,va,ix,arm).double()*va['ts'][ix].double()).cpu().numpy())
        return np.concatenate(out)
    # Actual gradient witness: A independent of COT; AC receives gradients; post-target channels unused.
    seed(42);w=Head().cuda();x=tr['tx'][:2];z=tr['tc']['transport'][:2].clone().requires_grad_(True);mask=tr['tm'][:2];ld=tr['tl'][:2]
    pred=w(x,z,mask,ld,True);gz=torch.autograd.grad(pred.sum(),z)[0]
    invalid=torch.arange(24,device='cuda')[None]>8+ld[:,None];assert (gz[invalid]==0).all() and (gz.abs().sum()>0)
    assert torch.equal(w(x,z,mask,ld,False),w(x,z+123,mask,ld,False))
    write(a.output/'witness.json',dict(state='PASS_COT_SWITCH_PREFIX_GRADIENT',parameters=sum(v.numel() for v in w.parameters())))
    del w;trials=[]
    for batch in (256,512,1024,2048,4096,8192):
        seed(42);m=Head().cuda();opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
        ix=torch.arange(batch,device='cuda')%len(tr['rows']);torch.cuda.reset_peak_memory_stats()
        for _ in range(2):update(m,opt,ix,'hybrid')
        torch.cuda.synchronize();t=time.monotonic()
        for _ in range(5):update(m,opt,ix,'hybrid')
        torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated();rate=5*batch/(time.monotonic()-t)
        if peak<.85*torch.cuda.get_device_properties(0).total_memory:trials.append(dict(batch=batch,samples_per_second=rate,peak_gib=peak/2**30))
        del m,opt;torch.cuda.empty_cache()
    speed=max(x['samples_per_second'] for x in trials);batch=min(x['batch'] for x in trials if x['samples_per_second']>=.95*speed)
    write(a.output/'profile.json',dict(candidates=trials,selected_batch=batch));print('PROFILE',batch,flush=True)
    outputs={};records=[];started=time.monotonic()
    for sv in (42,43,44):
        rng=np.random.default_rng(sv);stream=np.concatenate([rng.permutation(len(tr['rows'])) for _ in range(math.ceil(a.updates*batch/len(tr['rows'])))])[:a.updates*batch].reshape(a.updates,batch)
        for arm in ('A','transport','hybrid'):
            seed(sv);m=Head().cuda();initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in m.state_dict().values())).hexdigest()
            opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4);best=float('inf');run=a.output/f'{arm}_seed{sv}';run.mkdir();curve=[]
            for step in range(1,a.updates+1):
                lr=1e-5+.5*(1e-3-1e-5)*(1+math.cos(math.pi*(step-1)/a.updates))
                for group in opt.param_groups:group['lr']=lr
                loss=update(m,opt,torch.tensor(stream[step-1],device='cuda'),arm)
                if step%100==0 or step==a.updates:
                    pr=evaluate(m);me=metrics(pr,va['ghi']);curve.append(dict(step=step,loss=loss,**me))
                    if me['rmse']<best:
                        best=me['rmse'];torch.save(dict(model=m.state_dict(),step=step,seed=sv,arm=arm,initial_sha256=initial),run/'best.pt')
                        np.savez_compressed(run/'predictions.npz',pred_ghi=pr,true_ghi=va['ghi'],row_ids=va['rows'],lead_minutes=(va['lead']+1)*15,target_day=va['dates'],init_time=va['init'],target_time=va['target'])
                    write(a.output/'status.json',dict(state='TRAINING',arm=arm,seed=sv,step=step,completed_runs=len(records),best_rmse=best));print('TRAIN',sv,arm,step,me['rmse'],flush=True)
            torch.save(dict(model=m.state_dict(),step=a.updates),run/'last.pt');write(run/'curve.json',curve)
            ck=torch.load(run/'best.pt',weights_only=False);m.load_state_dict(ck['model']);pr=evaluate(m)
            with np.load(run/'predictions.npz') as saved:assert np.array_equal(pr,saved['pred_ghi'])
            rec=dict(seed=sv,arm=arm,updates=a.updates,selected_step=ck['step'],initial_sha256=initial,checkpoint_sha256=sha(run/'best.pt'),**metrics(pr,va['ghi']))
            write(run/'complete.json',rec);records.append(rec);outputs[(sv,arm)]=pr;del m,opt;torch.cuda.empty_cache()
    # FP64 metrics and paired target-date bootstrap of mean seed RMSE differences.
    days=np.unique(va['dates']);rng=np.random.default_rng(20261015);draw=rng.integers(0,len(days),(2000,len(days)))
    summary={}
    for arm in ('transport','hybrid'):
        armout={}
        for lead in (0,*range(15,241,15)):
            mask=np.ones(len(va['rows']),bool) if lead==0 else (va['lead']+1)*15==lead
            rm={x:[metrics(outputs[(s,x)][mask],va['ghi'][mask])['rmse'] for s in (42,43,44)] for x in ('A',arm)}
            mass=np.zeros((len(days),3,3),np.float64)
            for di,day in enumerate(days):
                ix=mask&(va['dates']==day)
                for si,sv in enumerate((42,43,44)):
                    mass[di,si]=[ix.sum(),np.square(outputs[(sv,'A')][ix]-va['ghi'][ix]).sum(),np.square(outputs[(sv,arm)][ix]-va['ghi'][ix]).sum()]
            z=mass[draw].sum(1);z=z[(z[:,:,0]>0).all(1)]
            diff=(np.sqrt(z[:,:,2]/z[:,:,0])-np.sqrt(z[:,:,1]/z[:,:,0])).mean(1)
            aa=float(np.mean(rm['A']));cc=float(np.mean(rm[arm]))
            armout[str(lead)]=dict(n=int(mask.sum()),A_rmse=aa,AC_rmse=cc,AC_minus_A=cc-aa,reduction_percent=100*(aa-cc)/aa,delta_ci95=np.quantile(diff,[.025,.975]).tolist(),seed_rmse=rm)
        summary[arm]=armout
    for sv in (42,43,44):assert len({r['initial_sha256'] for r in records if r['seed']==sv})==1
    write(a.output/'summary.json',dict(state='COMPLETE_HISTORY_COT_FORECAST_GHI_PAIRED',scope='Sili old-split validation pilot; validation-selected candidates; not new-monthly/test',test_used=False,runs=records,results=summary,batch=batch,updates=a.updates,elapsed_seconds=time.monotonic()-started))
    write(a.output/'status.json',dict(state='COMPLETE_HISTORY_COT_FORECAST_GHI_PAIRED',completed_runs=9))
    print('COMPLETE_HISTORY_COT_FORECAST_GHI_PAIRED',flush=True)


if __name__=='__main__':main()
