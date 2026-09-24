"""Build exploratory R(real future AGRI) COT reference for matched GHI diagnosis.

This is an oracle/supervision bank, never a deployable forecast input.  Train
and validation only; each unique real future frame is read and inverted once.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic(path, payload):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--split', choices=('train','val'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-sequences', type=int)
    p.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    p.add_argument('--frame-batch', type=int, default=64)
    a = p.parse_args()
    if a.output.exists():
        raise RuntimeError('refuse to overwrite real-future oracle bank')
    if a.frame_batch < 1:
        raise ValueError('frame batch must be positive')
    sc = json.loads((a.project/'configs/s_frozen_hunan_seed42.json').read_text())
    rc = json.loads((a.project/'configs/r_frozen_repaired_seed42.json').read_text())
    for path, expected in ((sc['manifest'],sc['manifest_sha256']),
                           (rc['checkpoint'],rc['checkpoint_sha256'])):
        if sha(path) != expected:
            raise RuntimeError('source asset hash mismatch: '+str(path))
    with Path(sc['manifest']).open(newline='') as stream:
        rows = [row for row in csv.DictReader(stream) if row['split'] == a.split]
    if a.max_sequences is not None:
        rows = rows[:a.max_sequences]
    if not rows:
        raise RuntimeError('empty oracle split')
    tasks = defaultdict(list)
    for index, row in enumerate(rows):
        frames = row['data_relpaths'].split('|')
        if len(frames) != 24:
            raise RuntimeError('noncanonical row')
        for lead, rel in enumerate(frames[8:]):
            tasks[rel].append((index, lead))
    import sys
    sys.path.insert(0,str(a.source_root))
    from hunan_data import load_frame
    spec = importlib.util.spec_from_file_location('cot_real_future_r',a.project/'scripts/train_cot_repaired.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = torch.load(rc['checkpoint'],map_location='cpu',weights_only=False)
    if checkpoint['metadata']['source_code_sha256'] != sha(a.project/'scripts/train_cot_repaired.py'):
        raise RuntimeError('frozen R code hash drift')
    device = torch.device(a.device)
    if a.device == 'cpu':
        torch.set_num_threads(1)
    model = module.COTUNet().to(device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model'],strict=True)
    rnorm = checkpoint['metadata']['norm']
    mean = torch.as_tensor(rnorm['mean'],device=device,dtype=torch.float32)[None,:,None,None]
    std = torch.as_tensor(rnorm['std'],device=device,dtype=torch.float32)[None,:,None,None]
    boxes = [tuple(v) for v in sc['station_patches'].values()]
    a.output.mkdir(parents=True)
    output = np.lib.format.open_memmap(a.output/'real_future_cot_log1p.npy',mode='w+',dtype=np.float32,
                                       shape=(len(rows),2,16,1,16,16))
    covered = np.zeros((len(rows),16),bool)
    started = time.monotonic()
    if a.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    task_items = list(tasks.items())
    with torch.no_grad():
        for start in range(0,len(task_items),a.frame_batch):
            chunk = task_items[start:start+a.frame_batch]
            images = []
            for rel,_ in chunk:
                agri, valid, geom = load_frame(a.data_root,rel)
                raw = np.concatenate((np.where(valid,agri,np.nan),geom),axis=0)
                images.extend(raw[:,r0:r1,c0:c1] for r0,r1,c0,c1 in boxes)
            x = torch.from_numpy(np.stack(images).astype(np.float32)).to(device)
            x = (x-mean)/std
            x = torch.where(torch.isfinite(x),x,torch.zeros_like(x))
            cot_batch = model(x.float()).float().cpu().numpy().reshape(len(chunk),2,1,16,16)
            if not np.isfinite(cot_batch).all() or (cot_batch < 0).any():
                raise RuntimeError('invalid R(real future AGRI) output')
            for (_,locations),cot in zip(chunk,cot_batch):
                for index,lead in locations:
                    if covered[index,lead]:
                        raise RuntimeError('duplicate oracle map')
                    output[index,:,lead] = cot
                    covered[index,lead] = True
            if start == 0 or (start//a.frame_batch)%8 == 0:
                output.flush()
                atomic(a.output/'status.json',{'state':'BUILDING_REAL_FUTURE_R_ORACLE',
                       'unique_frames_done':min(start+len(chunk),len(tasks)),'unique_frames_total':len(tasks),
                       'sequences':len(rows),'test_used':False,'inference_use':'forbidden'})
                print('PROGRESS',min(start+len(chunk),len(tasks)),len(tasks),flush=True)
    output.flush()
    if not covered.all():
        raise RuntimeError('incomplete oracle bank')
    del output
    with (a.output/'index.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(('index','seq_id','split','BJT_start'))
        for index,row in enumerate(rows):
            writer.writerow((index,row['seq_id'],row['split'],row['BJT_start']))
    report={'state':'PILOT_REAL_FUTURE_R_ORACLE' if a.max_sequences is not None else 'COMPLETE_REAL_FUTURE_R_ORACLE',
            'test_used':False,'deployable':False,'inference_use':'forbidden',
            'split':a.split,'sequences':len(rows),'unique_real_future_frames':len(tasks),
            'shape':[len(rows),2,16,1,16,16],'layout':'sequence,station,lead,channel,row,col',
            'reference':'frozen R(real future AGRI), not observed independent COT',
            'R_checkpoint_sha256':rc['checkpoint_sha256'],'manifest_sha256':sc['manifest_sha256'],
            'script_sha256':sha(__file__),'elapsed_seconds':time.monotonic()-started,
            'frame_batch':a.frame_batch,
            'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/1024**3 if a.device == 'cuda' else None,
            'device':a.device,
            'cot_sha256':sha(a.output/'real_future_cot_log1p.npy'),'index_sha256':sha(a.output/'index.csv')}
    atomic(a.output/'complete.json',report)
    print(report['state'],report['sequences'],flush=True)


if __name__=='__main__':
    main()
