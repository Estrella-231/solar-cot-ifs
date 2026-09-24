"""CPU witness: does frozen 16x16 R preserve station values on larger crops?

Only fixed observed validation-history frames are read.  Larger-crop outputs
are diagnostics, not authorized future-COT banks or a new R model contract.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=('train','val'), default='val')
    a = p.parse_args()
    if a.output.exists():
        raise RuntimeError('refuse to overwrite halo audit')
    torch.set_num_threads(1)
    sc = json.loads((a.project/'configs/s_frozen_hunan_seed42.json').read_text())
    rc = json.loads((a.project/'configs/r_frozen_repaired_seed42.json').read_text())
    if sha(Path(rc['checkpoint'])) != rc['checkpoint_sha256'] or sha(Path(sc['manifest'])) != sc['manifest_sha256']:
        raise RuntimeError('asset hash drift')
    model_source = a.project/'scripts/train_cot_repaired.py'
    spec = importlib.util.spec_from_file_location('cot_halo_frozen_r', model_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = torch.load(rc['checkpoint'], map_location='cpu', weights_only=False)
    if checkpoint['metadata']['source_code_sha256'] != sha(model_source):
        raise RuntimeError('R implementation changed')
    r = module.COTUNet().eval().requires_grad_(False)
    r.load_state_dict(checkpoint['model'], strict=True)
    mean = np.asarray(checkpoint['metadata']['norm']['mean'], np.float32)[:, None, None]
    std = np.asarray(checkpoint['metadata']['norm']['std'], np.float32)[:, None, None]
    sys.path.insert(0, str(a.source_root))
    from hunan_data import load_frame
    with Path(sc['manifest']).open(newline='') as f:
        rows = [row for row in csv.DictReader(f) if row['split'] == a.split]
    indices = sorted({0, len(rows)//3, 2*len(rows)//3, len(rows)-1})
    records = []
    with torch.no_grad():
        for index in indices:
            row = rows[index]
            # Last observed frame is history index 7, never future index 8+.
            rel = row['data_relpaths'].split('|')[7]
            agri, valid, geom = load_frame(a.data_root, rel)
            physical = np.concatenate((np.where(valid, agri, np.nan), geom), axis=0)
            normalized = (physical - mean) / std
            normalized[~np.isfinite(normalized)] = 0
            for station, box in sc['station_patches'].items():
                r0, r1, c0, c1 = box
                native = r(torch.from_numpy(normalized[None, :, r0:r1, c0:c1])).numpy()[0, 0]
                for size in (32, 64):
                    halo = (size - 16)//2
                    extended = r(torch.from_numpy(normalized[None, :, r0-halo:r1+halo, c0-halo:c1+halo])).numpy()[0, 0]
                    center = extended[halo:halo+16, halo:halo+16]
                    difference = np.abs(center-native)
                    records.append({'sequence_index':index,'seq_id':row['seq_id'],'station':station,
                                    'crop_size':size,'native_center_logcot':float(native[8,8]),
                                    'extended_center_logcot':float(center[8,8]),
                                    'full_16_mae_logcot':float(difference.mean()),
                                    'center_5x5_mae_logcot':float(difference[6:11,6:11].mean()),
                                    'full_16_max_logcot':float(difference.max())})
    report={'state':'COMPLETE_R_HALO_CPU_WITNESS','test_used':False,'split':a.split,
            'selection':'fixed split sequence indices; last observed image only',
            'R_checkpoint_sha256':rc['checkpoint_sha256'],'R_source_sha256':sha(model_source),
            'manifest_sha256':sc['manifest_sha256'],'script_sha256':sha(__file__),
            'records':records,
            'mean_16_mae_by_crop':{str(size):float(np.mean([v['full_16_mae_logcot'] for v in records if v['crop_size']==size]))
                                   for size in (32,64)},
            'mean_center5_mae_by_crop':{str(size):float(np.mean([v['center_5x5_mae_logcot'] for v in records if v['crop_size']==size]))
                                        for size in (32,64)}}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('state','mean_16_mae_by_crop','mean_center5_mae_by_crop')}),flush=True)


if __name__=='__main__':
    main()
