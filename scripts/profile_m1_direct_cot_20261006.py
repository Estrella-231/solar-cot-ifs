"""Single-A800 batch profile including fresh AGRI/geometry file reads per batch."""
import argparse, csv, json, sys, time, gc
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from direct_cot_model_20261005 import DirectCOTForecaster, masked_log_cot_mse

class SmokeDataset(Dataset):
    def __init__(self, data_root, code_dir, norm_path, windows_path, labels_path, repeats=40):
        sys.path.insert(0, str(code_dir))
        import hunan_data
        self.hunan_data=hunan_data; self.root=data_root; self.repeats=repeats
        norm=json.loads(norm_path.read_text(encoding='utf-8'))
        self.mean=np.asarray(norm['mean'][:13],np.float32)[:,None,None]
        self.std=np.asarray(norm['std'][:13],np.float32)[:,None,None]
        with windows_path.open(newline='',encoding='utf-8-sig') as f: self.rows=list(csv.DictReader(f))
        with np.load(labels_path,allow_pickle=False) as z:
            self.cot=np.asarray(z['cot'],np.float32); self.mask=np.asarray(z['mask'],bool)
        if len(self.rows)!=8 or self.cot.shape!=(8,16,1,256,256): raise RuntimeError('unexpected smoke support')
    def __len__(self): return len(self.rows)*self.repeats
    def __getitem__(self,index):
        i=index%len(self.rows); row=self.rows[i]; hist=[]; hg=[]; fg=[]
        for stamp in row['history_utc'].split('|'):
            rel=f"{stamp[:4]}/{stamp[:6]}/{stamp[:8]}/{stamp}.npy"
            agri,valid,geom=self.hunan_data.load_frame(self.root,rel)
            hist.append(np.where(valid,(agri-self.mean)/self.std,0.).astype(np.float32)); hg.append(geom.astype(np.float32))
        for stamp in row['future_utc'].split('|'):
            bjt=datetime.strptime(stamp,'%Y%m%d%H%M')+timedelta(hours=8)
            with np.load(self.hunan_data.resolve_aux(self.root,bjt),allow_pickle=False) as z:
                fg.append(np.stack([z[k] for k in ('cosSOZ','cosRAA','day_mask')]).astype(np.float32))
        return np.stack(hist),np.concatenate((np.stack(hg),np.stack(fg))),self.cot[i],self.mask[i]

def main():
    p=argparse.ArgumentParser()
    for name in ['data-root','code-dir','norm','windows','cpp-labels','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--batches',default='2,4,8,12,16,24')
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args(); assert torch.cuda.is_available()
    torch.manual_seed(42); np.random.seed(42)
    ds=SmokeDataset(a.data_root,a.code_dir,a.norm,a.windows,a.cpp_labels)
    result={'state':'RUNNING','gpu':torch.cuda.get_device_name(0),'split_id':'hunan_monthly_block_2d_buffer1d_seed42_v1','support':'8 train windows repeated for systems profiling only; no skill metric','batches':[]}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for bs in map(int,a.batches.split(',')):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model=DirectCOTForecaster(width=96).cuda().train(); opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        loader=DataLoader(ds,batch_size=bs,shuffle=False,num_workers=a.workers,pin_memory=True,prefetch_factor=2,persistent_workers=False,drop_last=True)
        it=iter(loader); times=[]; peak=0; failed=None
        try:
            for step in range(4):
                t=time.perf_counter()
                try: xb,gb,yb,mb=next(it)
                except StopIteration: break
                xb=xb.cuda(non_blocking=True); gb=gb.cuda(non_blocking=True); yb=yb.cuda(non_blocking=True); mb=mb.cuda(non_blocking=True)
                opt.zero_grad(set_to_none=True); pred=model(xb,gb); loss=masked_log_cot_mse(pred,yb,mb); loss.backward(); opt.step(); torch.cuda.synchronize()
                peak=max(peak,torch.cuda.max_memory_allocated()); dt=time.perf_counter()-t
                if step>0: times.append(dt)
            if not times: raise RuntimeError('insufficient profiled batches')
        except torch.cuda.OutOfMemoryError as e:
            failed='CUDA_OOM'; torch.cuda.empty_cache()
        except Exception as e:
            failed=repr(e)
        entry={'batch':bs,'elapsed_seconds_per_measured_batch':times,'median_seconds':float(np.median(times)) if times else None,'samples_per_second':float(bs/np.median(times)) if times else None,'peak_allocated_bytes':int(peak),'failure':failed,'workers':a.workers}
        result['batches'].append(entry); print(json.dumps(entry),flush=True)
        del it,loader,model,opt; gc.collect(); torch.cuda.empty_cache()
        if failed: break
    usable=[r for r in result['batches'] if r['failure'] is None and r['peak_allocated_bytes']<int(torch.cuda.get_device_properties(0).total_memory*.85)]
    best=max([r['samples_per_second'] for r in usable],default=0.)
    result['chosen_batch']=max((r['batch'] for r in usable if r['samples_per_second']>=.95*best),default=None)
    result['state']='COMPLETE_PROFILE' if not any(r['failure'] for r in result['batches']) else 'PROFILE_PARTIAL'
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8'); print(json.dumps(result),flush=True)
if __name__=='__main__': main()
