"""CPU-only observed-future V patches, solely for OFFLINE frozen-R reference."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

STATIONS = ('sili', 'zhujia')
SMOKE = ('202501040315', '202408171000', '202603020000')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def csvread(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def atomic_json(p, obj):
    q = p.with_suffix('.tmp'); q.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n', encoding='utf-8'); q.replace(p)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--data-root', type=Path, required=True); ap.add_argument('--days', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args(); base = a.base
    gate = json.loads((base/'data/dense_observed_reference_gate_20261021.json').read_text())
    support = base/'data/sequence_support_v1'; src = base/'data/observed_reference_reader_20261021/hunan_data.py'
    for key, p in [('v_support', support/'v_sequences.csv'), ('support_audit', support/'support_audit.json'),
                   ('days', a.days), ('reader', src), ('geometry', base/'solar_geometry_corrected.py'),
                   ('grid', base/'data/grid_static.npz'), ('r_rows', base/'data/r_u_pack_v2/rows_u_supervised.csv')]:
        assert sha(p) == gate['sha256'][key], key
    audit = json.loads((support/'support_audit.json').read_text())
    assert audit['test_or_buffer_numeric_payloads_opened'] == 0 and audit['test_rows_selected'] == 0
    days = {r['date_BJT']: r['split'] for r in csvread(a.days)}
    seqs = csvread(support/'v_sequences.csv'); rels = set()
    assert len(seqs) == 3140
    for r in seqs:
        assert r['split'] == 'val'
        paths = r['data_relpaths'].split('|'); assert len(paths) == 24
        tt = [datetime.strptime(Path(p).stem, '%Y%m%d%H%M') for p in paths]
        assert all(tt[i+1]-tt[i] == timedelta(minutes=15) for i in range(23))
        assert all(days[(t+timedelta(hours=8)).date().isoformat()] == 'val' for t in tt)
        rels.update(paths[8:])
    rels = sorted(rels); assert len(rels) == 4025
    if a.smoke:
        rels = [r for r in rels if Path(r).stem in SMOKE]
        assert len(rels) == 3
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    sys.path.insert(0, str(src.parent)); sys.path.insert(0, str(base))
    from hunan_data import load_frame, resolve_aux
    from predict_direct_cot_hv import station_centers, corrected_geometry
    from solar_geometry_corrected import compute_solar_angles
    centers = station_centers(base/'data/r_u_pack_v2')
    with np.load(base/'data/grid_static.npz', allow_pickle=False) as g:
        lat, lon, saa = (g[k].astype(np.float32) for k in ('lat', 'lon', 'saa'))
    n = len(rels)
    x = np.lib.format.open_memmap(a.output/'x_raw.npy', mode='w+', dtype=np.float32, shape=(n,2,16,16,16))
    validout = np.lib.format.open_memmap(a.output/'mask.npy', mode='w+', dtype=np.bool_, shape=(n,2,1,16,16))
    began = time.time()
    def extract(rel):
        rawpath = a.data_root/'data'/rel
        utc = datetime.strptime(Path(rel).stem, '%Y%m%d%H%M')
        aux = resolve_aux(a.data_root, utc+timedelta(hours=8))
        before = [(p.stat().st_size,p.stat().st_mtime_ns) for p in (rawpath, aux)]
        image, qa, _ = load_frame(a.data_root, rel)
        geom = corrected_geometry(rel, lat, lon, saa, compute_solar_angles)
        xx, mm = [], []
        for st in STATIONS:
            r,c = centers[st]; sl = np.s_[r-8:r+8,c-8:c+8]
            ag = image[:,sl[0],sl[1]].copy(); ok = qa[:,sl[0],sl[1]].all(axis=0)
            ag[:,~ok] = np.nan
            gg = geom[:,sl[0],sl[1]]
            xx.append(np.concatenate([ag,gg],axis=0))
            mm.append((ok & (gg[0] > np.cos(np.deg2rad(80))) & (gg[2] > .5))[None])
        after = [(p.stat().st_size,p.stat().st_mtime_ns) for p in (rawpath, aux)]
        assert before == after, 'source changed during read: '+rel
        xx,mm = np.stack(xx),np.stack(mm)
        return xx,mm,{'data_relpath':rel, 'target_time_utc':utc.isoformat(' '),
            'target_time_bjt':(utc+timedelta(hours=8)).isoformat(' '), 'aux_path':str(aux),
            'raw_size':before[0][0], 'raw_mtime_ns':before[0][1],
            'aux_size':before[1][0], 'aux_mtime_ns':before[1][1],
            'extracted_patch_sha256':hashlib.sha256(xx.tobytes(order='C')).hexdigest()}
    rr = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, (xx, mm, row) in enumerate(pool.map(extract, rels)):
            x[i]=xx; validout[i]=mm; rr.append(dict(row,frame_index=i))
            if (i+1)%100 == 0 or i+1 == n:
                x.flush();validout.flush()
                atomic_json(a.output/'progress.json', {'state':'EXTRACTING_OFFLINE_OBSERVED_V_REFERENCE',
                    'frames':i+1,'total':n,'elapsed_seconds':time.time()-began,'training_started':False})
                print(f'OBS_REFERENCE_FRAME {i+1}/{n} elapsed={time.time()-began:.1f}s',flush=True)
    with (a.output/'rows.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    x.flush(); validout.flush(); del x, validout
    result={'state':'COMPLETE_OBSERVED_V_REFERENCE_PATCHES','smoke':a.smoke,'frames':n,'stations':list(STATIONS),
        'elapsed_seconds':time.time()-began,'future_observed_agri_read_for_reference':True,
        'future_observations_used_by_deployable_inference':False,'ghi_labels_read':False,'cpp_read':False,
        'training_started':False,'test_payload_opened':False,'buffer_payload_opened':False,
        'reader_sha256':sha(src),'script_sha256':sha(__file__),'gate_sha256':sha(base/'data/dense_observed_reference_gate_20261021.json'),
        'source_file_provenance':'size/mtime before=after; extracted patch SHA; raw files are not claimed whole-file SHA-pinned',
        'artifact_sha256':{name:sha(a.output/name) for name in ('x_raw.npy','mask.npy','rows.csv')}}
    atomic_json(a.output/'complete.json', result); print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
