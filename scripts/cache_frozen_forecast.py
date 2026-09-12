"""Frozen S validation reload + train/val forecast patches. No test payloads.

Future AGRI is read only by the separately named validation metric pass.
Production feature pass reads historical AGRI and deterministic geometry only.
"""
import argparse,csv,gc,hashlib,json,os,time
from datetime import datetime,timedelta,timezone
from pathlib import Path
import numpy as np
import torch
import inspect
from torch.utils.data import Dataset,DataLoader
from train_full import Model
from train_ddp import CachedData
from hunan_data import resolve_aux
from train_cot_repaired import COTUNet

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def atomic(p,obj):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)

def npz(p,**arrays):
    t=p.with_suffix('.tmp')
    with t.open('wb') as f:np.savez(f,**arrays)
    t.replace(p)

def identity(row):
    rels=row['data_relpaths'].split('|')
    times=[datetime.strptime(Path(p).stem,'%Y%m%d%H%M').replace(tzinfo=timezone.utc) for p in rels]
    assert len(times)==24 and all(t==times[0]+timedelta(minutes=15*i) for i,t in enumerate(times))
    assert times[0]+timedelta(hours=8)==datetime.fromisoformat(row['BJT_start']).replace(tzinfo=timezone.utc)
    return times[7]

class ForecastInputs(CachedData):
    def __getitem__(self,i):
        row=self.rows[i];rels=row['data_relpaths'].split('|');images=[];geos=[]
        for rel in rels[:8]:
            a,v,g=self.frame(rel)
            images.append(np.where(v,(a-self.mean)/self.std,0).astype('float32'));geos.append(g)
        for rel in rels[8:]:
            # Deliberately never read future AGRI or validity masks here.
            cached=self.cache/(rel+'.npz')
            if cached.exists():
                with np.load(cached,allow_pickle=False) as z:g=z['g']
            else:
                t=datetime.strptime(Path(rel).stem,'%Y%m%d%H%M')+timedelta(hours=8)
                with np.load(resolve_aux(self.root,t),allow_pickle=False) as z:
                    g=np.stack([z[k] for k in ('cosSOZ','cosRAA','day_mask')])
            assert np.isfinite(g).all();geos.append(g)
        return i,np.stack(images),np.stack(geos)

def loader(data,batch,workers):
    return DataLoader(data,batch_size=batch,num_workers=workers,pin_memory=True,
                      **({'prefetch_factor':1,'persistent_workers':True} if workers else {}))

@torch.no_grad()
def validation(model,data,out,expected):
    path=out/'validation_reload.json'
    if path.exists():
        report=json.loads(path.read_text());assert report['status']=='PASS'
        assert sha(out/'validation_sufficient_statistics.npz')==report['statistic_sha256']
        with np.load(out/'validation_sufficient_statistics.npz') as z:
            score=float(np.sqrt(z['squared_error'].sum((0,1))/z['valid_count'].sum((0,1))).mean())
        assert abs(score-report['rmse'])<1e-12 and abs(score-expected)<0.001
        return
    sums=[];counts=[];persistence=[];start=time.monotonic()
    for j,(x,g,y,v) in enumerate(loader(data,8,4)):
        x,g,y,v=[t.cuda(non_blocking=True) for t in (x,g,y,v)]
        with torch.autocast('cuda',dtype=torch.bfloat16):p=model(x,g)
        assert torch.isfinite(p).all()
        sums.append(((p.float()-y).square()*v).sum((-1,-2)).double().cpu().numpy())
        counts.append(v.sum((-1,-2)).cpu().numpy())
        persistence.append(((x[:,-1:,]+torch.zeros_like(y)-y).square()*v).sum((-1,-2)).double().cpu().numpy())
        if j%50==0:print('VALIDATION',j,flush=True)
    s=np.concatenate(sums);n=np.concatenate(counts);base=np.concatenate(persistence)
    npz(out/'validation_sufficient_statistics.npz',squared_error=s,valid_count=n,persistence_squared_error=base)
    with np.load(out/'validation_sufficient_statistics.npz') as z:
        score=float(np.sqrt(z['squared_error'].sum((0,1))/z['valid_count'].sum((0,1))).mean())
    assert abs(score-expected)<0.001,('reload disagrees',score,expected)
    atomic(path,dict(status='PASS',samples=len(data),rmse=score,training_record_rmse=expected,
        absolute_difference=abs(score-expected),tolerance=0.001,seconds=time.monotonic()-start,
        statistic_sha256=sha(out/'validation_sufficient_statistics.npz'),test_used=False,
        metric='mean of 13 channel normalized full-grid RMSE; QA valid pixels',
        note='saved per-sequence/lead/channel sufficient statistics; not saved full-grid predictions'))

@torch.no_grad()
def profile(model,sample,out):
    path=out/'profile.json'
    if path.exists():return json.loads(path.read_text())['batch']
    trials=[];best=None
    for b in (4,8,16,24,32):
        x=g=p=None
        try:
            x=torch.from_numpy(sample[1]).unsqueeze(0).repeat(b,1,1,1,1).cuda()
            g=torch.from_numpy(sample[2][:16]).unsqueeze(0).repeat(b,1,1,1,1).cuda()
            torch.cuda.reset_peak_memory_stats()
            with torch.autocast('cuda',dtype=torch.bfloat16):p=model(x,g)
            torch.cuda.synchronize();start=time.monotonic()
            for _ in range(3):
                with torch.autocast('cuda',dtype=torch.bfloat16):p=model(x,g)
            torch.cuda.synchronize();secs=(time.monotonic()-start)/3
            assert torch.isfinite(p).all()
            peak=torch.cuda.max_memory_allocated();row=dict(batch=b,seconds=secs,samples_per_second=b/secs,peak_GiB=peak/2**30)
            trials.append(row);print('PROFILE',json.dumps(row),flush=True)
            if peak<0.80*torch.cuda.get_device_properties(0).total_memory:
                if best is None or row['samples_per_second']>best['samples_per_second']:best=row
            else:break
        except torch.cuda.OutOfMemoryError:
            trials.append(dict(batch=b,status='OOM'));break
        finally:
            del x,g,p;gc.collect();torch.cuda.empty_cache()
    assert best is not None
    atomic(path,dict(batch=best['batch'],trials=trials,reserve_fraction=0.20));return best['batch']

@torch.no_grad()
def cache(model,rmodel,data,out,batch,patches,snorm,rnorm,limit=0):
    out.mkdir(parents=True,exist_ok=True)
    if limit:data.rows=data.rows[:limit]
    manifest=out/'index.csv'
    with manifest.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['index','seq_id','split','init_time_utc'])
        w.writerows((i,r['seq_id'],r['split'],identity(r).isoformat()) for i,r in enumerate(data.rows))
    sm=torch.tensor(snorm['mean'],device='cuda')[None,None,None,:,None,None]
    ss=torch.tensor(snorm['std'],device='cuda')[None,None,None,:,None,None]
    rm=torch.tensor(rnorm['mean'],device='cuda')[None,:,None,None]
    rs=torch.tensor(rnorm['std'],device='cuda')[None,:,None,None]
    started=time.monotonic();shards=[]
    for j,(ids,x,g) in enumerate(loader(data,batch,4)):
        path=out/f'shard_{j:05d}.npz'
        receipt=path.with_suffix('.json')
        if path.exists():
            saved=json.loads(receipt.read_text())
            assert sha(path)==saved['sha256'] and saved['samples']==len(ids)
            with np.load(path) as z:
                assert np.array_equal(z['indices'],ids.numpy())
                for key,shape in [('predicted_agri_physical',(len(ids),2,16,13,16,16)),
                        ('geometry',(len(ids),2,16,3,16,16)),('cot_log1p',(len(ids),2,16,16,16)),
                        ('cot_features',(len(ids),2,16,4)),('r_latent_mean',(len(ids),2,16,16))]:
                    assert z[key].shape==shape and np.isfinite(z[key]).all()
            shards.append(saved);continue
        x=x.cuda(non_blocking=True);g=g.cuda(non_blocking=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(x,g[:,:16])
        assert torch.isfinite(pred).all()
        p=torch.stack([pred[:,:,:,a:b,c:d] for a,b,c,d in patches.values()],1).float()*ss+sm
        geom=torch.stack([g[:,8:,:,a:b,c:d] for a,b,c,d in patches.values()],1).float()
        raw=torch.cat((p,geom),3).flatten(0,2)
        inp=(raw-rm)/rs
        assert torch.isfinite(inp).all()
        features=rmodel.forward_features(inp)
        cot=torch.nn.functional.softplus(rmodel.head(features)).reshape(len(ids),2,16,16,16)
        flat=cot.flatten(-2)
        explicit=torch.stack((flat.mean(-1),flat.std(-1,unbiased=False),torch.quantile(flat,0.9,dim=-1),cot[...,8,8]),-1)
        latent=features.mean((-1,-2)).reshape(len(ids),2,16,16)
        assert torch.isfinite(cot).all() and torch.isfinite(latent).all()
        npz(path,indices=ids.numpy(),predicted_agri_physical=p.cpu().numpy(),geometry=geom.cpu().numpy(),
            cot_log1p=cot.cpu().numpy(),cot_features=explicit.cpu().numpy(),r_latent_mean=latent.cpu().numpy())
        # Read-back of serialized values, before completion or downstream consumption.
        with np.load(path) as z:
            assert np.array_equal(z['predicted_agri_physical'],p.cpu().numpy())
            assert np.isfinite(z['cot_features']).all() and np.array_equal(z['indices'],ids.numpy())
        saved=dict(name=path.name,sha256=sha(path),samples=len(ids))
        atomic(receipt,saved);shards.append(saved)
        if j%10==0:
            status=dict(state='CACHING',samples=int(ids[-1])+1,total=len(data),seconds=time.monotonic()-started,batch=batch)
            atomic(out/'status.json',status);print(json.dumps(status),flush=True)
    atomic(out/'complete.json',dict(state='COMPLETE',samples=len(data),index_sha256=sha(manifest),shards=shards,
        stations=list(patches),seconds=time.monotonic()-started,test_used=False,center_pixel=[8,8],
        prediction_dtype='float32 physical AGRI; BF16 S forward; FP32 frozen R',future_AGRI_input=False,
        future_CPP_input=False,geometry='nominal-time deterministic solar geometry from aux; no observation-derived fields'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
    sc=json.loads((a.project/'configs/s_frozen_hunan_seed42.json').read_text())
    rc=json.loads((a.project/'configs/r_frozen_repaired_seed42.json').read_text())
    root=Path('/public/home/slfu/ttzhou/Auxiliary_data/HuNan_AGRI_Preprocessed')
    for key in ('checkpoint','manifest','normalization'):assert sha(sc[key])==sc[key+'_sha256'],key
    assert sha(root/'grid_static.npz')==sc['grid_sha256']
    assert sha(rc['checkpoint'])==rc['checkpoint_sha256']
    source_root=Path(inspect.getfile(Model)).parent
    source_hashes={n:sha(source_root/n) for n in ('train_full.py','hunan_data.py','train_ddp.py','simvp.py')}
    metadata=dict(S=sc,R=rc,script_sha256=sha(__file__),imported_source_sha256=source_hashes,
        R_source_sha256=sha(inspect.getfile(COTUNet)),test_used=False)
    if (out/'contract.json').exists():assert json.loads((out/'contract.json').read_text())==metadata
    else:atomic(out/'contract.json',metadata)
    norm=json.loads(Path(sc['normalization']).read_text());checkpoint=torch.load(sc['checkpoint'],map_location='cpu')
    assert checkpoint['metadata']['code_sha256']==source_hashes
    assert checkpoint['metadata']['manifest_sha256']==sc['manifest_sha256']
    assert checkpoint['metadata']['normalization_sha256']==sc['normalization_sha256']
    model=Model();model.load_state_dict(checkpoint['model'],strict=True);model.cuda().eval().requires_grad_(False)
    rck=torch.load(rc['checkpoint'],map_location='cpu');rmodel=COTUNet();rmodel.load_state_dict(rck['model'],strict=True)
    assert rck['metadata']['source_code_sha256']==metadata['R_source_sha256']
    rmodel.cuda().eval().requires_grad_(False);torch.manual_seed(42);torch.backends.cudnn.benchmark=True
    assert all(not p.requires_grad for p in list(model.parameters())+list(rmodel.parameters()))
    args=(str(root),sc['manifest'],norm)
    nodecache='/tmp/slfu_hunan_simvp_frames_a08d895dbe30c7ef'
    val=CachedData(*args,'val',cache=nodecache)
    validation(model,val,out,sc['validation_rmse'])
    inputs=ForecastInputs(*args,'train',cache=nodecache)
    b=profile(model,inputs[0],out)
    cache(model,rmodel,inputs,out/'smoke32',b,sc['station_patches'],norm,rck['metadata']['norm'],32)
    atomic(out/'smoke_audit.json',dict(status='PASS_FEATURE_PIPELINE',sequences=32,S_frozen=True,R_frozen=True,
        test_used=False,head_overfit='PENDING',label_join='PENDING'))
    for split in ('train','val'):
        cache(model,rmodel,ForecastInputs(*args,split,cache=nodecache),out/split,b,sc['station_patches'],norm,rck['metadata']['norm'])
    atomic(out/'complete.json',dict(state='COMPLETE_FEATURE_CACHE',test_used=False,head_training='PENDING_LABEL_JOIN_AND_OVERFIT'))
    print('ALL_COMPLETE',flush=True)

if __name__=='__main__':main()
