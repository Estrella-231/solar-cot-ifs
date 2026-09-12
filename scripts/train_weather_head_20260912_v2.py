"""Causal forecast-AGRI weather head; cloud-coverage labels, not cloud-phase classes."""
import argparse, csv, gc, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False));t.replace(p)
def coverage_label(counts,n):
    if n<231:return -1  # >=90% valid; missing never interpreted as clear
    cloud=counts[1]+counts[2]
    return 0 if cloud<=.2*n else 2 if cloud>=.8*n else 1

class WeatherHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn=nn.Sequential(nn.Conv2d(13,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,32,3,padding=1),nn.SiLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.lead=nn.Embedding(16,8);self.station=nn.Embedding(2,4)
        self.mlp=nn.Sequential(nn.Linear(47,64),nn.SiLU(),nn.Linear(64,3))
    def forward(self,x,g,station,lead):
        return self.mlp(torch.cat((self.cnn(x),g,self.station(station),self.lead(lead)),dim=1))

def classification(y,p,weights=None):
    if not len(y):return dict(n=0)
    pred=p.argmax(1);w=np.ones(len(y)) if weights is None else weights
    cm=np.bincount(3*y+pred,weights=w,minlength=9).reshape(3,3)
    support=cm.sum(1);recall=np.divide(cm.diagonal(),support,out=np.zeros(3),where=support>0)
    precision=np.divide(cm.diagonal(),cm.sum(0),out=np.zeros(3),where=cm.sum(0)>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(3),where=precision+recall>0)
    return dict(n=len(y),confusion=cm.tolist(),accuracy=float(cm.trace()/cm.sum()),balanced_accuracy=float(recall.mean()),macro_f1=float(f1.mean()),recall=recall.tolist(),precision=precision.tolist(),class_support=support.tolist(),nll=float(np.average(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)),weights=w)))

def cpu_check():
    assert [coverage_label([256,0,0],256),coverage_label([128,64,64],256),coverage_label([0,128,128],256)]==[0,1,2]
    assert coverage_label([0,0,0],0)==-1 and coverage_label([230,0,0],230)==-1
    assert coverage_label([80,20,0],100)==-1
    assert coverage_label([200,50,0],250)==0
    assert coverage_label([199,0,51],250)==1
    assert coverage_label([50,100,100],250)==2
    assert coverage_label([51,100,99],250)==1
    for ncloud in range(257):
        assert coverage_label([256-ncloud,ncloud,0],256)==coverage_label([256-ncloud,0,ncloud],256)
    torch.manual_seed(42);m=WeatherHead();x=torch.randn(6,13,16,16);g=torch.randn(6,3)
    logits=m(x,g,torch.tensor([0,1]*3),torch.arange(6));loss=nn.functional.cross_entropy(logits,torch.tensor([0,1,2]*2));loss.backward()
    assert logits.shape==(6,3) and torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters())
    a=classification(np.array([0,1,2]),np.eye(3));assert a['accuracy']==a['macro_f1']==1
    print('WEATHER_CPU_WITNESS_PASS',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cpu-check',action='store_true');args=ap.parse_args()
    if args.cpu_check:cpu_check();return
    pack=ROOT/'data/head_pack_trainval_20260908_v1';labels=ROOT/'data/weather_clp_trainval_20260912_v1'
    out=ROOT/'runs/weather_coverage_head_seed42_20260912_v2';out.mkdir(exist_ok=False)
    audit=json.loads((pack/'audit.json').read_text());sel=json.loads((labels/'selection.json').read_text());col=json.loads((labels/'collection.json').read_text())
    assert audit['state']=='COMPLETE' and audit['test_used'] is False and audit['S_frozen'] and audit['R_frozen']
    assert sel['pack_audit_sha256']==sha(pack/'audit.json')
    assert sel['targets_sha256']==col['targets_sha256']==sha(labels/'targets.json')
    assert sel['mapping_sha256']==sha(labels/'row_target_id.npy') and col['labels_sha256']==sha(labels/'labels.jsonl')
    targets=json.loads((labels/'targets.json').read_text());records=[json.loads(s) for s in (labels/'labels.jsonl').read_text().splitlines()]
    assert len(records)==len(targets)==col['total']
    target_y=np.full(len(targets),-1,dtype=np.int64)
    for i,(r,t) in enumerate(zip(records,targets)):
        assert all(r[k]==v for k,v in t.items()) and t['target_id']==i
        if r['state']=='READ':target_y[i]=coverage_label(r['class_counts'],r['valid_pixels'])
    mapping=np.load(labels/'row_target_id.npy');y=target_y[mapping]
    arrays={}
    for k in ('x','g','station','lead','split'):
        assert sha(pack/audit['arrays'][k]['name'])==audit['arrays'][k]['sha256'],k
        arrays[k]=np.load(pack/audit['arrays'][k]['name'],allow_pickle=False)
        assert np.isfinite(arrays[k]).all()
    split=arrays['split'];assert np.isin(split,[0,1]).all()
    assert len(y)==637902 and (split==1).sum()==99849
    target_split=np.array([t['split'] for t in targets]);assert np.array_equal(target_split[mapping],split)
    # Original train only: latest 20% of UTC target dates for stopping. One boundary
    # day purged so 0-4h examples do not share an initialization across fit/tune.
    days=np.array([t['target_time_utc'][:10] for t in targets]);train_days=np.unique(days[target_split==0])
    cut=train_days[int(.8*len(train_days))];purge=train_days[int(.8*len(train_days))-1]
    fitmask=(split==0)&(days[mapping]<purge)&(y>=0)
    tunemask=(split==0)&(days[mapping]>=cut)&(y>=0)
    valmask=split==1
    fit=np.flatnonzero(fitmask);tune=np.flatnonzero(tunemask);val=np.flatnonzero(valmask)
    assert len(fit)>1000 and len(tune)>1000 and not np.intersect1d(mapping[fit],mapping[tune]).size
    for ids in (fit,tune):assert np.all(np.bincount(y[ids],minlength=3)>0)
    # Each unique station,target gets total weight one across repeated lead rows.
    repeats=np.bincount(mapping);base=1.0/repeats[mapping]
    tc=np.bincount(target_y[np.unique(mapping[fit])],minlength=3)
    classweight=tc.sum()/(3*tc);weight=base*classweight[np.maximum(y,0)];weight/=weight[fit].mean()
    contract=dict(test_used=False,seed=42,classes=['sunny','partly_cloudy','overcast'],coverage_thresholds=[.2,.8],min_valid_pixels=231,
        inputs=['frozen forecast AGRI 13x16x16','solar geometry 3','station','lead'],excluded_inputs=['observed GHI','observed kt','future true AGRI','future CLP','COT'],
        label='target-instant patch cloud fraction from CLP1+CLP2; regional proxy, not official daily weather',
        fit_rows=len(fit),tune_rows=len(tune),validation_rows=len(val),fit_target_class_counts=tc.tolist(),
        train_purge_date=purge,tune_start_date=cut,stopping='original-train temporal holdout inverse-target-repeat NLL; max30epochs patience6',
        optimizer='AdamW lr0.001 weight_decay0.0001',loss='CE weighted inverse unique-target class frequency and inverse repeated-target rows',
        parameters=sum(p.numel() for p in WeatherHead().parameters()),pack_sha256=sha(pack/'audit.json'),labels_sha256=sha(labels/'labels.jsonl'),code_sha256=sha(__file__),
        label_coverage={str(s):dict(total_targets=int((target_split==s).sum()),labeled_targets=int(((target_split==s)&(target_y>=0)).sum()),classes=np.bincount(target_y[(target_split==s)&(target_y>=0)],minlength=3).tolist()) for s in (0,1)})
    write(out/'contract.json',contract);print('CONTRACT',json.dumps(contract),flush=True)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported();torch.set_num_threads(1);torch.backends.cudnn.benchmark=True
    data={}
    for k in ('x','g','station','lead'):
        print('GPU_LOAD_START',k,flush=True)
        data[k]=torch.from_numpy(arrays[k]).cuda()
        print('GPU_LOAD_DONE',k,flush=True)
    data['y']=torch.from_numpy(y).cuda();data['weight']=torch.from_numpy(weight.astype('float32')).cuda()
    def fresh():
        torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        model=WeatherHead().cuda();return model,torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    def forward(m,ix):
        with torch.autocast('cuda',dtype=torch.bfloat16):return m(*(data[k][ix] for k in ('x','g','station','lead'))).float()
    def step(m,opt,ix):
        m.train();opt.zero_grad(set_to_none=True);logits=forward(m,ix)
        loss=(nn.functional.cross_entropy(logits,data['y'][ix],reduction='none')*data['weight'][ix]).mean()
        assert torch.isfinite(loss);loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        opt.step();return float(loss.detach())
    trials=[];fitdev=torch.from_numpy(fit).cuda()
    for batch in (128,256,512,1024,2048):
        m,opt=fresh()
        try:
            ix=fitdev[:batch];torch.cuda.reset_peak_memory_stats()
            for _ in range(3):step(m,opt,ix)
            torch.cuda.synchronize();start=time.monotonic()
            for _ in range(8):step(m,opt,ix)
            torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()
            if peak>.8*torch.cuda.get_device_properties(0).total_memory:break
            trials.append(dict(batch=batch,samples_per_second=len(ix)*8/(time.monotonic()-start),peak_GiB=peak/2**30))
            print('PROFILE_TRIAL',json.dumps(trials[-1]),flush=True)
        except torch.cuda.OutOfMemoryError:break
        finally:del m,opt;gc.collect();torch.cuda.empty_cache()
    assert trials;top=max(t['samples_per_second'] for t in trials);batch=min(t['batch'] for t in trials if t['samples_per_second']>=.95*top)
    write(out/'profile.json',dict(batch=batch,trials=trials));print('PROFILE',trials,'CHOSEN',batch,flush=True)
    @torch.no_grad()
    def predict(m,ids):
        m.eval();return np.concatenate([forward(m,torch.from_numpy(ix).cuda()).softmax(1).cpu().numpy() for ix in np.array_split(ids,max(1,int(np.ceil(len(ids)/batch))))])
    # Seeded balanced tiny fit is a numeric gate, never used for model selection.
    m,opt=fresh();small=np.concatenate([fit[y[fit]==k][:8] for k in range(3)]);ix=torch.from_numpy(small).cuda()
    initial=classification(y[small],predict(m,small))['nll']
    for _ in range(300):step(m,opt,ix)
    final=classification(y[small],predict(m,small))['nll'];assert final<initial*.5,'tiny overfit gate failed'
    write(out/'overfit.json',dict(initial_nll=initial,final_nll=final,passed=True,pack_rows=small.tolist()))
    del m,opt;m,opt=fresh();best=float('inf');wait=0
    for epoch in range(30):
        start=time.monotonic();gen=torch.Generator(device='cuda').manual_seed(42+epoch);ids=fitdev[torch.randperm(len(fitdev),device='cuda',generator=gen)]
        loss=sum(step(m,opt,ix)*len(ix) for ix in ids.split(batch))/len(fit)
        prob=predict(m,tune);metric=classification(y[tune],prob,base[tune]);score=metric['nll']
        if score<best:
            best=score;wait=0;torch.save(dict(model=m.state_dict(),contract=contract,epoch=epoch,batch=batch),out/'best.pt')
        else:wait+=1
        report=dict(epoch=epoch,train_loss=loss,tune=metric,seconds=time.monotonic()-start,wait=wait)
        write(out/f'epoch_{epoch:03d}.json',report);print('EPOCH',json.dumps(report),flush=True)
        if wait>=6:break
    ck=torch.load(out/'best.pt',map_location='cpu');m.load_state_dict(ck['model']);prob=predict(m,val)
    np.savez_compressed(out/'validation_predictions.npz',pack_row=val,probabilities=prob,predicted_class=prob.argmax(1),reference_class=y[val],target_id=mapping[val],station=arrays['station'][val],lead=arrays['lead'][val])
    valid=y[val]>=0
    metrics=dict(forecast_rows=classification(y[val][valid],prob[valid]),target_equal=classification(y[val][valid],prob[valid],base[val][valid]))
    metrics['station']={str(s):classification(y[val][valid&(arrays['station'][val]==s)],prob[valid&(arrays['station'][val]==s)]) for s in (0,1)}
    metrics['lead']={str((l+1)*15):classification(y[val][valid&(arrays['lead'][val]==l)],prob[valid&(arrays['lead'][val]==l)]) for l in range(16)}
    majority=np.argmax(tc);dummy=np.eye(3)[np.full(valid.sum(),majority)]
    metrics['train_majority_baseline']=classification(y[val][valid],dummy)
    cpu=WeatherHead().eval();cpu.load_state_dict(ck['model']);sample=val[np.linspace(0,len(val)-1,32,dtype=int)]
    with torch.no_grad():p_cpu=cpu(*(torch.from_numpy(np.array(arrays[k][sample])) for k in ('x','g','station','lead'))).softmax(1).numpy()
    p_gpu=predict(m,sample);error=float(abs(p_cpu-p_gpu).max());assert error<.03,'CPU reload parity'
    write(out/'complete.json',dict(state='COMPLETE_WEATHER_COVERAGE_HEAD',test_used=False,metrics=metrics,checkpoint_sha256=sha(out/'best.pt'),predictions_sha256=sha(out/'validation_predictions.npz'),best_epoch=ck['epoch'],cpu_reload_max_probability_error=error))
    print('WEATHER_TRAINING_COMPLETE',json.dumps(metrics['forecast_rows']),flush=True)
if __name__=='__main__':main()
