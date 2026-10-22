"""Validate all F-role references against timestamps, solar fields and source manifests."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;rawroot=P/'raw_attempt1';refroot=P/'reference_attempt1'
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def rows(path):return list(csv.DictReader(path.open()))
meta=json.loads((rawroot/'COMPLETE.json').read_text());done=json.loads((refroot/'COMPLETE.json').read_text())
assert done['state']=='MATCHED_FIVE_FOLD_ACTUAL_TRAIN_COT_REFERENCE_COMPLETE'
assert meta['raw_sha256']==sha(rawroot/'raw16.npy')
frames=rows(rawroot/'frames.csv');rf=rows(rawroot/'role_frames.csv');pairs=rows(rawroot/'pairs.csv')
assert len({(r['station'],r['init_time'],r['target_time']) for r in pairs})==len(pairs)
raw=np.load(rawroot/'raw16.npy',mmap_mode='r');fi={r['bjt']:int(r['frame_index']) for r in frames}
byrole={};expected=0
for role in [f'fold_{k}' for k in range(5)]:
    receipt=json.loads((refroot/role/'COMPLETE.json').read_text());fn=refroot/role/'reference.npz'
    assert sha(fn)==receipt['reference_sha256']
    with np.load(fn) as z:
        times=z['bjt'];ids=z['raw_frame_indices'];cot=z['cot'];valid=z['valid']
        assert times.tolist()==sorted(r['bjt'] for r in rf if r['role']==role)
        assert ids.tolist()==[fi[t] for t in times]
        assert cot.shape==(len(times),1,16,16) and valid.shape==cot.shape
        x=raw[ids];mask=(x[:,15:16]>.5)&(x[:,13:14]>np.cos(np.deg2rad(80.)))
        assert np.array_equal(valid,mask)
        assert np.isfinite(cot).all() and (cot>=0).all() and (cot[~valid]==0).all()
        for j,idx in enumerate(ids):assert hashlib.sha256(np.asarray(raw[idx]).tobytes()).hexdigest()==frames[idx]['patch_sha256']
        byrole[role]=dict(frames=len(times),valid_pixels=int(valid.sum()),R_sha256=receipt['R_sha256'],normalization_sha256=receipt['normalization_sha256'],reference_sha256=sha(fn),forecast_replay=receipt['forecast_replay'])
        expected+=len(times)
assert expected==done['frames']==meta['role_frames']
report=dict(state='PASS_ALL_F_REFERENCE_ARRAYS_KEYS_SOLAR_AND_SOURCE_HASHES',frames=len(frames),role_frames=expected,pairs=len(pairs),counts=meta['counts'],roles=byrole,GHI_values_read=0,official_val_new_period_outerbuffer_test_payloads_read=0,new_training_runs=0)
(P/'REFERENCE_AUDIT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
