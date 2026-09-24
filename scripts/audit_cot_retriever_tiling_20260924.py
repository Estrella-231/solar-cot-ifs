"""CPU witness of overlap-tiled frozen 16x16 R around each Hunan station.

Only fixed observed validation-history frames are read.  The station-native
16x16 result is kept exactly; this checks disagreement in an outer 8px halo.
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
                canvas = np.zeros((32,32),np.float64)
                squares = np.zeros((32,32),np.float64)
                counts = np.zeros((32,32),np.float64)
                for row_offset in (0,8,16):
                    for col_offset in (0,8,16):
                        rr,cc = r0-8+row_offset,c0-8+col_offset
                        tile = r(torch.from_numpy(normalized[None,:,rr:rr+16,cc:cc+16])).numpy()[0,0]
                        canvas[row_offset:row_offset+16,col_offset:col_offset+16] += tile
                        squares[row_offset:row_offset+16,col_offset:col_offset+16] += tile*tile
                        counts[row_offset:row_offset+16,col_offset:col_offset+16] += 1
                tiled = canvas/counts
                overlap_std = np.sqrt(np.maximum(squares/counts-tiled*tiled,0))
                # Anchored output retains exactly the station R map used by H.
                anchored = tiled.copy()
                anchored[8:24,8:24] = native
                seam = np.concatenate((np.abs(anchored[8,:]-anchored[7,:]),
                                       np.abs(anchored[24,:]-anchored[23,:]),
                                       np.abs(anchored[:,8]-anchored[:,7]),
                                       np.abs(anchored[:,24]-anchored[:,23])))
                native_step = np.concatenate((np.abs(np.diff(native,axis=0)).ravel(),
                                              np.abs(np.diff(native,axis=1)).ravel()))
                records.append({'sequence_index':index,'seq_id':row['seq_id'],'station':station,
                                'unanchored_center16_mae_logcot':float(np.abs(tiled[8:24,8:24]-native).mean()),
                                'unanchored_center5_mae_logcot':float(np.abs(tiled[14:19,14:19]-native[6:11,6:11]).mean()),
                                'overlap_disagreement_mean_std_logcot':float(overlap_std[counts>1].mean()),
                                'overlap_disagreement_p95_std_logcot':float(np.percentile(overlap_std[counts>1],95)),
                                'anchored_seam_mean_abs_step_logcot':float(seam.mean()),
                                'native_interior_mean_abs_step_logcot':float(native_step.mean()),
                                'anchored_center_exact':bool(np.array_equal(anchored[8:24,8:24],native))})
    report={'state':'COMPLETE_R_TILING_CPU_WITNESS','test_used':False,'split':a.split,
            'selection':'fixed split sequence indices; last observed image only',
            'R_checkpoint_sha256':rc['checkpoint_sha256'],'R_source_sha256':sha(model_source),
            'manifest_sha256':sc['manifest_sha256'],'script_sha256':sha(__file__),
            'records':records,
            'mean_unanchored_center16_mae':float(np.mean([v['unanchored_center16_mae_logcot'] for v in records])),
            'mean_overlap_std':float(np.mean([v['overlap_disagreement_mean_std_logcot'] for v in records])),
            'mean_anchored_seam_abs_step':float(np.mean([v['anchored_seam_mean_abs_step_logcot'] for v in records])),
            'mean_native_interior_abs_step':float(np.mean([v['native_interior_mean_abs_step_logcot'] for v in records]))}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('state','mean_unanchored_center16_mae',
          'mean_overlap_std','mean_anchored_seam_abs_step','mean_native_interior_abs_step')}),flush=True)


if __name__=='__main__':
    main()
