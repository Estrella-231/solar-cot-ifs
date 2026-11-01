#!/usr/bin/env python3
"""End-to-end data-loader + forward/backward profile on real train sequences."""
from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
REPO_ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO_ROOT))
from scripts.retrieved_cot_forecaster_20261013 import RetrievedCOTForecaster, cot_supervision_masked_mse
from train_retrieved_cot_forecaster_20261017 import SequenceStationDataset

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--branch',type=Path,required=True); p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--pack',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8); p.add_argument('--width',type=int,default=96)
    p.add_argument('--warmup',type=int,default=3); p.add_argument('--measure',type=int,default=8)
    a=p.parse_args(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda': raise RuntimeError('profile requires target CUDA GPU')
    ds=SequenceStationDataset('train',a.branch,a.bank,a.pack)
    future_support=ds.build_future_support()
    eligible_station_samples=len(ds)
    eligible_sequences=int(future_support.any(axis=1).sum())
    results=[]
    for bs in (8,16,32,48,64):
        generator=torch.Generator().manual_seed(20261017)
        loader=DataLoader(ds,batch_size=bs,shuffle=True,generator=generator,
                          num_workers=a.workers,pin_memory=True,
                          persistent_workers=True,prefetch_factor=2)
        model=RetrievedCOTForecaster(width=a.width).to(device).train()
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        torch.cuda.reset_peak_memory_stats(); times=[]; seen=0
        it=iter(loader)
        for step in range(a.warmup+a.measure):
            try: batch=next(it)
            except StopIteration: break
            x,hm,geo,r,_,mask,*_=batch
            start=time.perf_counter()
            x=x.to(device,non_blocking=True); hm=hm.to(device,non_blocking=True)
            geo=geo.to(device,non_blocking=True); r=r.to(device,non_blocking=True); mask=mask.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True); pred=model(x,hm,geo)
            loss=cot_supervision_masked_mse(pred,r,mask); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            torch.cuda.synchronize(); elapsed=time.perf_counter()-start
            if step>=a.warmup: times.append(elapsed); seen += len(x)
        del model,opt,loader,it
        torch.cuda.empty_cache()
        if not times: raise RuntimeError(f'no measured batches for batch={bs}')
        results.append({'batch_size':bs,'measured_batches':len(times),'mean_seconds_per_batch':sum(times)/len(times),
                        'samples_per_second':seen/sum(times),'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
                        'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3})
        print(json.dumps(results[-1]),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    payload={'state':'PASS_REAL_TRAIN_LOADER_GPU_PROFILE','gpu':torch.cuda.get_device_name(0),
             'torch':torch.__version__,'workers':a.workers,'width':a.width,'results':results,
             'eligible_train_sequences':eligible_sequences,
             'eligible_train_sequence_station_samples':eligible_station_samples,
             'excluded_sequence_station_samples_without_any_valid_future_pixel':
                 int(2*len(ds.ids)-eligible_station_samples),
             'test_payload_opened':False}
    a.output.write_text(json.dumps(payload,indent=2)+'\n')
if __name__=='__main__': main()
