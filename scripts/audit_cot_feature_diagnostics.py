"""Bounded CPU-only frozen COT feature diagnostic; no training/test/raw AGRI reads."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from train_cot_repaired import COTUNet


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def summary(a):
    a = np.asarray(a, np.float64)
    assert len(a) and np.isfinite(a).all()
    return dict(n=len(a), mean=a.mean(0).tolist(), std=a.std(0).tolist(),
                min=a.min(0).tolist(), max=a.max(0).tolist(),
                quantile_levels=[.01, .1, .5, .9, .99],
                quantiles=np.quantile(a, [.01, .1, .5, .9, .99], axis=0).tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    start = time.monotonic()
    torch.set_num_threads(2)
    root = a.project
    pack = root / 'data/head_pack_trainval_20260908_v1'
    cache = root / 'data/frozen_forecast_trainval_20260907_v1'
    viewdir = root / 'data/head_qc_view_20260908_v1'
    audit = json.loads((pack / 'audit.json').read_text())
    norm = json.loads((pack / 'norm.json').read_text())
    view = json.loads((viewdir / 'view.json').read_text())
    contract = json.loads((cache / 'contract.json').read_text())
    assert audit['state'] == 'COMPLETE' and audit['test_used'] is False
    hashes = {}
    for key in ('c', 'station', 'lead', 'split', 'sequence'):
        hashes[key] = sha(pack / (key + '.npy'))
        assert hashes[key] == audit['arrays'][key]['sha256']
    assert sha(pack / 'norm.json') == audit['norm_sha256']
    assert sha(pack / 'audit.json') == view['pack_audit_sha256']
    assert sha(viewdir / 'keep.npy') == view['keep_sha256']
    assert sha(cache / 'contract.json') == audit['source_cache_contract_sha256']
    for key in ('S', 'R'):
        spec = contract[key]
        assert sha(spec['checkpoint']) == spec['checkpoint_sha256']
    snorm_path = Path(contract['S']['normalization'])
    assert sha(snorm_path) == contract['S']['normalization_sha256']
    snorm = json.loads(snorm_path.read_text())
    rck = torch.load(contract['R']['checkpoint'], map_location='cpu')
    rnorm = rck['metadata']['norm']
    assert sha(root / 'scripts/train_cot_repaired.py') == contract['R_source_sha256']
    assert rck['metadata']['source_code_sha256'] == contract['R_source_sha256']
    expected_channels = ['C%02d' % i for i in (1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15)]
    assert snorm['source_indices'] == [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]
    assert rnorm['feature_names'] == expected_channels + ['cosSOZ', 'cosRAA', 'day_mask']
    assert norm['c_features'] == ['mean_log1p_cot', 'std_log1p_cot', 'p90_log1p_cot', 'center_log1p_cot']
    assert norm['fit_split'] == 'train' and rnorm['test_used'] is False
    data = {k: np.load(pack / (k + '.npy'), allow_pickle=False)
            for k in ('c', 'station', 'lead', 'split', 'sequence')}
    assert set(np.unique(data['split'])) == {0, 1}
    c = data['c'].astype(np.float64)
    keep = np.load(viewdir / 'keep.npy')
    # Invert the float32 constants actually used when packing; roundoff is reported.
    cm = np.asarray(norm['c']['mean'], np.float32).astype(np.float64)
    cs = np.asarray(norm['c']['std'], np.float32).astype(np.float64)
    raw = c * cs + cm
    tr = data['split'] == 0
    qtr = tr & keep
    qspec = view['transforms_in_base_pack_normalized_coordinates']['c']
    qm = np.asarray(qspec['mean'])
    qs = np.asarray(qspec['std'])
    assert np.allclose(c[qtr].mean(0), qm, atol=1e-10, rtol=1e-10)
    assert np.allclose(c[qtr].std(0), qs, atol=1e-10, rtol=1e-10)
    qc = (c - qm) / qs
    # Affine refit in base-normalized coordinates equals a raw-feature train refit.
    direct = (raw - raw[qtr].mean(0)) / raw[qtr].std(0)
    qc_equivalence = float(np.abs(qc - direct).max())
    assert qc_equivalence < 1e-8
    groups = []
    for cohort, mask in [('original', np.ones(len(c), bool)), ('qc1', keep)]:
        for split, sid in [('train', 0), ('validation', 1)]:
            for station, station_name in enumerate(audit['station_order']):
                for lead_name, lo, hi in [('all', 0, 16), ('15-60', 0, 4), ('75-120', 4, 8), ('135-240', 8, 16)]:
                    select = mask & (data['split'] == sid) & (data['station'] == station)
                    select &= (data['lead'] >= lo) & (data['lead'] < hi)
                    v = raw[select]
                    groups.append(dict(cohort=cohort, split=split, station=station_name, lead_minutes=lead_name,
                        raw_log1p_summary=summary(v),
                        near_spatially_constant_fraction=float((v[:, 1] <= 1e-6).mean()),
                        mean_log1p_ge_log101_fraction=float((v[:, 0] >= np.log1p(100)).mean()),
                        center_log1p_ge_log101_fraction=float((v[:, 3] >= np.log1p(100)).mean())))
    model = COTUNet().eval().requires_grad_(False)
    model.load_state_dict(rck['model'], strict=True)
    rm = np.asarray(rnorm['mean'], np.float32)[None, :, None, None]
    rs = np.asarray(rnorm['std'], np.float32)[None, :, None, None]
    samples = []
    for split, sid in [('train', 0), ('val', 1)]:
        complete = json.loads((cache / split / 'complete.json').read_text())
        shards = complete['shards']
        starts = np.cumsum([0] + [s['samples'] for s in shards])
        eligible = np.unique(data['sequence'][data['split'] == sid])
        for anchor in (0, len(shards) // 2, len(shards) - 1):
            nearest = int(eligible[np.abs(eligible - starts[anchor]).argmin()])
            shardpos = int(np.searchsorted(starts, nearest, side='right') - 1)
            shard = shards[shardpos]
            path = cache / split / shard['name']
            assert sha(path) == shard['sha256']
            with np.load(path, allow_pickle=False) as z:
                ids = z['indices']
                candidates = np.where((data['split'] == sid) & np.isin(data['sequence'], ids))[0]
                assert len(candidates), 'sample shard has no labeled rows'
                seq = int(data['sequence'][candidates[0]])
                local = int(np.where(ids == seq)[0][0])
                agri = z['predicted_agri_physical'][local]
                geometry = z['geometry'][local]
                saved = z['cot_log1p'][local]
                feats = z['cot_features'][local]
                inp = (np.concatenate((agri, geometry), axis=2).reshape(32, 16, 16, 16) - rm) / rs
                with torch.no_grad():
                    reloaded = model(torch.from_numpy(inp)).numpy().reshape(2, 16, 16, 16)
                err = float(np.abs(reloaded - saved).max())
                # Descriptive diagnostic: the initial 0.001 threshold failed.
                # Existing verify_cot_run.py:57 uses 0.01 for CPU FP32/GPU FP16;
                # this cache ran R FP32, so that older threshold is not a
                # preregistered pass criterion for the present numerical contract.
                assert np.isfinite(reloaded).all()
                rowids = candidates[data['sequence'][candidates] == seq]
                stations, leads = data['station'][rowids], data['lead'][rowids]
                feature_error = float(np.abs(raw[rowids] - feats[stations, leads]).max())
                assert feature_error < 2e-6
                pix = saved[stations, leads].reshape(-1)
                flat = saved.reshape(2, 16, -1)
                reconstructed = np.stack([flat.mean(-1), flat.std(-1), np.quantile(flat, .9, axis=-1), saved[..., 8, 8]], -1)
                assert np.allclose(reconstructed, feats, atol=2e-6, rtol=2e-5)
                samples.append(dict(split=split, anchor_shard_position=anchor,
                    selected_shard_position=shardpos, shard=shard['name'], shard_sha256=shard['sha256'],
                    sequence=seq, labeled_windows=len(rowids), pixel_count=len(pix),
                    reload_max_absolute_log1p_error=err,
                    reload_rmse_log1p=float(np.sqrt(np.mean((reloaded.astype(np.float64)-saved)**2))),
                    pack_inverse_max_feature_error=feature_error,
                    pixel_log1p_summary=summary(pix), pixel_cot_ge100_fraction=float((pix >= np.log1p(100)).mean()),
                    pixel_nearzero_log1p_fraction=float((pix <= 1e-6).mean()),
                    r_normalized_input_channel_mean=inp.mean((0, 2, 3)).tolist(),
                    r_normalized_input_channel_std=inp.std((0, 2, 3)).tolist()))
    code_hashes = {n: sha(root / 'scripts' / n) for n in
                   ('cache_frozen_forecast.py', 'pack_head_training.py', 'train_head_pilot.py', 'train_head_qc_pilot.py')}
    assert code_hashes['cache_frozen_forecast.py'] == contract['script_sha256']
    report = dict(state='COMPLETE_WITH_RELOAD_TOLERANCE_WARNING', test_used=False,
        source_pack=str(pack), pack_audit_sha256=sha(pack / 'audit.json'), array_sha256=hashes,
        cache_contract_sha256=sha(cache / 'contract.json'), norm_sha256=sha(pack / 'norm.json'),
        qc_view_sha256=sha(viewdir / 'view.json'), script_sha256=sha(__file__), code_sha256=code_hashes,
        S_checkpoint_sha256=contract['S']['checkpoint_sha256'], R_checkpoint_sha256=contract['R']['checkpoint_sha256'],
        channels=expected_channels, r_norm=rnorm, head_norm=norm, c_features=norm['c_features'],
        rows=len(c), nonfinite_c=int((~np.isfinite(c)).sum()), negative_raw_c_below_roundoff=int((raw < -1e-6).sum()),
        original_train_normalized=summary(c[tr]), qc_train_normalized=summary(qc[qtr]),
        qc_transform_direct_refit_max_error=qc_equivalence, groups=groups, cached_shard_samples=samples,
        structural_checks='PASS_CHANNEL_ORDER_NORM_HASH_FEATURE_MAPPING',
        reload_tolerance=dict(status='DESCRIPTIVE_DIFFERENCE_REQUIRES_MATCHED_BACKEND_WITNESS',
            later_comparison_threshold=.01,
            source='scripts/verify_cot_run.py:57 uses max_error<0.01 for CPU FP32 vs saved GPU FP16',
            initial_diagnostic_threshold=.001, initial_observed_error=.003983736038208008,
            contract_difference='current frozen cache R forward is outside autocast and recorded FP32; old GPU FP16 tolerance is not automatically applicable',
            interpretation='initial 0.001 failed; later 0.01 comparison used a different numerical contract and is not a pre-fixed pass gate',
            next='repeat same six fixed sequences with GPU R; record cudnn TF32 on/off and other backend settings; compare CPU and saved cache without training or overwriting'),
        definitions=dict(feature_values='four statistics of log1p(COT); not four physical-COT statistics',
            saturation='No output clipping; log101 exceedance is a diagnostic of reference-range exceedance, not numerical saturation.',
            feature_tail='mean-log1p tail is not pixel-COT tail; center describes one pixel only.',
            sample_scope='nearest shard with labeled rows to first/middle/last train and val shard starts; first eligible sequence in each; 6 deterministic sequences, not representative'),
        limitations=['Train/validation feature differences mix seasons and do not measure real-vs-forecast R domain error.',
            'CPU reload witnesses reuse frozen predicted AGRI and do not independently rerun S.',
            'Only six shards are revisited here; previous full shard acceptance is separately hash-bound.',
            'No new real-AGRI or CPP reference pairing; retrieval degradation or its causal role in the pilot is unestablished.'],
        seconds=time.monotonic()-start)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(dict(state=report['state'], rows=len(c), samples=len(samples),
        qc_refit_error=qc_equivalence, max_reload_error=max(x['reload_max_absolute_log1p_error'] for x in samples),
        seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
