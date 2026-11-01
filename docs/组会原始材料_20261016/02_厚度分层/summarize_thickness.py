"""Saved-validation GHI gain by existing physical COT bins; no model execution."""
from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime, timedelta
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
EXP = ROOT / 'experiments/monthly_R_station16_20261013_v1'
PACK = EXP / 'results/ghi_head_pack_new_r_v1'
SEEDS, STATIONS, ARMS = (42, 43, 44), ('sili', 'zhujia'), ('A', 'AC', 'AP')
BINS = [('nearzero_[0,0.1)', 0, .1), ('thin_[0.1,5)', .1, 5),
        ('moderate_[5,30)', 5, 30), ('thick_[30,100]', 30, 100.0000001),
        ('out_of_reference_range_>100', 100.0000001, float('inf'))]

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def main():
    audit = json.loads((EXP / 'results/ghi_heads_three_seed_validation_audit.json').read_text())
    meta = json.loads((PACK / 'pack_manifest.json').read_text())
    assert sha(PACK / 'pack_manifest.json') == audit['pack_manifest_sha256']
    assert sha(PACK / 'rows.csv') == audit['rows_sha256']
    rows_all = list(csv.DictReader((PACK / 'rows.csv').open(encoding='utf-8', newline='')))
    assert set(r['split'] for r in rows_all) == {'train', 'val'}
    vi = np.array([i for i, r in enumerate(rows_all) if r['split'] == 'val'])
    rows = [rows_all[i] for i in vi]
    keys = [(r['station'], r['target_time_utc']) for r in rows]
    assert len(keys) == len(set(keys)) == 4479
    st = np.array([r['station'] for r in rows])
    y = np.array([float(r['ghi']) for r in rows], dtype=np.float64)
    dates = np.array([(datetime.fromisoformat(r['target_time_utc']) + timedelta(hours=8)).date().isoformat() for r in rows])
    ds = np.unique(dates)
    assert ds.tolist() == audit['bjt_dates']
    pred = np.empty((3, 3, len(rows)), dtype=np.float64)
    for i, seed in enumerate(SEEDS):
        for station in STATIONS:
            use = st == station
            for j, arm in enumerate(ARMS):
                path = EXP / f'results/ghi_heads_new_r_seed{seed}_v1/{station}/{arm}_val_prediction_ghi_wm2.npy'
                assert sha(path) == audit['source_artifacts'][f'{seed}/{station}']['prediction_sha256'][arm]
                a = np.load(path, allow_pickle=False).astype(np.float64)
                assert a.shape == (int(use.sum()),) and np.isfinite(a).all()
                pred[i, j, use] = a
    assert sha(PACK / 'cot_log1p.npy') == meta['array_sha256']['cot_log1p.npy']
    cot = np.load(PACK / 'cot_log1p.npy', mmap_mode='r', allow_pickle=False)
    rc = np.expm1(np.asarray(cot[vi, 0, 1, 1], dtype=np.float64))
    cpp_path = ROOT / 'refine-logs/cot_value_first_20261013/validation_predictions_monthly_R_seed42.npz'
    key_path = ROOT / 'refine-logs/cot_value_first_20261013/clp_reference_monthly_R_seed42.npz'
    cpp = np.load(cpp_path, allow_pickle=False)
    clp = np.load(key_path, allow_pickle=False)
    idx_map = {int(v): i for i, v in enumerate(cpp['sample_indices'])}
    assert len(idx_map) == len(cpp['sample_indices'])
    lut = {}
    for sample_id, station, ts in zip(clp['sample_indices'], clp['station'], clp['target_time_utc']):
        key = (str(station), str(ts))
        assert key not in lut and int(sample_id) in idx_map
        lut[key] = idx_map[int(sample_id)]
    cc = np.full(len(rows), np.nan)
    for i, key in enumerate(keys):
        if key in lut:
            j = lut[key]
            if cpp['mask'][j, 0, 8, 8]: cc[i] = float(cpp['reference_cot'][j, 0, 8, 8])
    assert np.isfinite(rc).all() and (rc >= 0).all()
    rng = np.random.default_rng(20261013)
    # The same 52-date resamples as the frozen primary comparison, shared
    # across stations, seeds, arms, and all bins. No per-group retuning.
    chosen = rng.choice(len(ds), size=(2000, len(ds)), replace=True)
    weights = np.array([np.bincount(v, minlength=len(ds)) for v in chosen], dtype=np.float64)
    err = pred - y[None, None, :]
    out = {'state': 'COMPLETE_SAVED_VALIDATION_THICKNESS_DIAGNOSTIC',
           'split_id': meta['split_id'], 'test_payload_opened': False,
           'training_or_model_inference_run': False, 'validation_rows': len(rows),
           'threshold_source': 'experiments/monthly_R_station16_20261013_v1/scripts/audit_monthly_R_validation.py:39-43',
           'definition': 'physical dimensionless COT at station centre: index [8,8] of 16x16, [1,1] of 3x3; not a spatial mean',
           'bins': {k: [lo, hi if np.isfinite(hi) else None] for k,lo,hi in BINS},
           'bootstrap': {'replicates': 2000, 'rng_seed': 20261013, 'date_blocks': len(ds),
                         'unit': 'shared BJT target dates from the full validation cohort; paired across methods, seeds and stations',
                         'interval': 'pointwise percentile 95%; no multiple-comparison adjustment'},
           'metric': 'RMSE per seed and station, then equal average across 2 stations and 3 seeds; percentage=(A-AC)/A*100',
           'source_sha256': {'cpp': sha(cpp_path), 'key_npz': sha(key_path), 'pack': sha(PACK/'pack_manifest.json'),
                             'primary_audit': sha(EXP/'results/ghi_heads_three_seed_validation_audit.json'),
                             'cot_log1p': sha(PACK/'cot_log1p.npy')}, 'groups': []}
    csvrows=[]
    sources=[('all_validation',None), ('CPP_reference_centre',cc), ('R_retrieved_centre',rc)]
    for source, c in sources:
        masks=[('all', np.ones(len(rows), dtype=bool))] if c is None else [('supported', np.isfinite(c))] + [(name, np.isfinite(c)&(c>=lo)&(c<hi)) for name,lo,hi in BINS]
        for name, mask in masks:
            if not mask.any(): continue
            stats=[]; counts=[]; sums=[]; details={}
            for station in STATIONS:
                use=mask & (st==station)
                if not use.any(): break
                e=err[:,:,use]
                rmse=np.sqrt(np.mean(e**2,axis=-1)); stats.append(rmse)
                details[station]={'n':int(use.sum()),'dates':int(len(np.unique(dates[use]))),
                                  'rmse_by_seed':{str(seed):dict(zip(ARMS,map(float,rmse[i]))) for i,seed in enumerate(SEEDS)},
                                  'mean_mae':dict(zip(ARMS,map(float,np.mean(np.abs(e),axis=(0,2))))),
                                  'mean_bias':dict(zip(ARMS,map(float,np.mean(e,axis=(0,2)))))}
                counts.append(np.array([np.sum(use & (dates==d)) for d in ds],dtype=np.float64))
                sums.append(np.stack([np.sum(err[:,:,use & (dates==d)]**2,axis=-1) for d in ds],axis=-1))
            if len(stats)!=2:
                out['groups'].append({'source':source,'bin':name,'n':int(mask.sum()),'state':'INSUFFICIENT_TWO_STATION_SUPPORT'})
                continue
            means=np.mean(stats,axis=(0,1)); delta=float(means[1]-means[0]); pct=float((means[0]-means[1])/means[0]*100)
            nbs=np.stack([weights @ n for n in counts],axis=1)
            good=(nbs>0).all(axis=1)
            boot=[]
            for j in range(2):
                sse=np.einsum('bd,sad->bsa',weights[good],sums[j])
                boot.append(np.sqrt(sse/nbs[good,j,None,None]))
            bmean=np.mean(np.stack(boot),axis=(0,2))
            ci=np.quantile(bmean[:,1]-bmean[:,0],[.025,.975]).tolist()
            pci=np.quantile((bmean[:,0]-bmean[:,1])/bmean[:,0]*100,[.025,.975]).tolist()
            rec={'source':source,'bin':name,'n':int(mask.sum()),'BJT_dates':int(len(np.unique(dates[mask]))),
                 'by_station':details,'RMSE_Wm2':dict(zip(ARMS,map(float,means))),
                 'AC_minus_A_RMSE_Wm2':delta,'RMSE_reduction_percent':pct,
                 'AC_minus_A_95CI_Wm2':ci,'RMSE_reduction_percent_95CI':pci,
                 'AC_minus_AP_RMSE_Wm2':float(means[1]-means[2]),
                 'bootstrap_valid_replicates':int(good.sum()),
                 'AC_minus_A_by_seed':dict(zip(map(str,SEEDS),map(float,(np.mean(stats,axis=0)[:,1]-np.mean(stats,axis=0)[:,0]))))}
            out['groups'].append(rec)
            csvrows.append({'source':source,'bin':name,'n':rec['n'],'dates':rec['BJT_dates'],
                            'sili_n':details['sili']['n'],'zhujia_n':details['zhujia']['n'],
                            **{f'{a}_RMSE_Wm2':float(means[i]) for i,a in enumerate(ARMS)},
                            'AC_minus_A_Wm2':delta,'reduction_percent':pct,'delta_CI_low':ci[0],'delta_CI_high':ci[1],
                            'percent_CI_low':pci[0],'percent_CI_high':pci[1]})
    overall=out['groups'][0]
    assert max(abs(overall['RMSE_Wm2'][a]-audit['three_seed_arm_rmse_mean_wm2'][a]) for a in ARMS)<1e-10
    expected=audit['bootstrap']['results']['target_bjt_date']['AC_minus_A_wm2']['percentile_95_ci_wm2']
    assert np.max(np.abs(np.array(overall['AC_minus_A_95CI_Wm2'])-expected))<1e-10
    out['independent_primary_reproduction']={'RMSE_and_shared_date_CI_match_max_tolerance':1e-10}
    (OUT/'thickness_metrics.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    with (OUT/'thickness_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0])); w.writeheader();w.writerows(csvrows)
    lines=['# GHI retrieval gain by COT thickness (saved validation predictions)', '',
           'Date: 2026-10-14 BJT. Same-time observed AGRI; no forecast result; no training, model execution, or test payload.', '',
           'Thresholds 0.1, 5 and 30 follow the pre-existing monthly-R reference-bin audit. Classification uses the station-centre pixel, not a full-patch mean or percentile. R bins are based on expm1 of the unstandardized saved log1p COT once; CPP bins use physical reference COT directly.', '',
           'Each value averages station-wise RMSE equally across the two Hunan stations and the three matched seeds. Relative reduction is (RMSE_A - RMSE_AC) / RMSE_A. Shared BJT-date bootstrap (2,000 draws, seed 20261013, 52-date full-cohort universe) preserves pairing; the overall primary point estimate and CI are reproduced to <1e-10. Stratum intervals are descriptive pointwise intervals conditional on validation checkpoint selection.', '']
    for source,_ in sources:
        lines += [f'## {source}', '', '| COT bin | Sili / Zhujia rows | BJT dates | A RMSE | AC RMSE | AP RMSE | AC−A [95% CI] W/m² | RMSE reduction |', '|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in out['groups']:
            if r['source']!=source or 'RMSE_Wm2' not in r: continue
            n=r['by_station']; m=r['RMSE_Wm2']; ci=r['AC_minus_A_95CI_Wm2']
            lines.append(f"| {r['bin']} | {n['sili']['n']} / {n['zhujia']['n']} | {r['BJT_dates']} | {m['A']:.3f} | {m['AC']:.3f} | {m['AP']:.3f} | {r['AC_minus_A_RMSE_Wm2']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | {r['RMSE_reduction_percent']:+.2f}% |")
        lines += ['']
    cpp_thick=next(r for r in out['groups'] if r['source']=='CPP_reference_centre' and r['bin']=='thick_[30,100]')
    lines += ['## Main findings / 主要发现', '',
              '总体同期GHI估计：A 123.980 → AC 120.240 W/m²，降低3.740 W/m²（3.02%）。', '',
              '按CPP参考的站点中心COT分层，厚云层（30≤COT≤100）A 74.929 → AC 58.105 W/m²，降低16.824 W/m²（22.45%）；AC−A的日期块95%区间为[−21.862,−12.359] W/m²。173条样本覆盖25个BJT日期，四里80条/15日、竺家93条/17日。三个种子的两站等权差值均为负。', '',
              '厚云收益有站点差异：四里三种子平均A/AC为69.870/64.661 W/m²；竺家为79.987/51.549 W/m²。厚云子集内的平均GHI高估同时减弱：四里bias由+14.460降到+1.648，竺家由+48.356降到+19.432 W/m²。这支持描述“厚云参考条件下COT增强尤其有帮助”，但不能把误差变化唯一归因为COT物理语义。', '',
              'CPP近零、薄云、中等COT层分别约降低1.48%、1.70%、1.70%，但各自配对区间跨0。各层并非都已确定有收益，也未证明随厚度单调增加。', '',
              '全体4,479行按R检索中心COT分组的补充分解，近零/薄/中等/厚层分别降低0.33%/1.00%/6.09%/24.32%；后两层区间低于0。该分组由模型检索值定义，会受到R误差与样本跨层影响，应和CPP参考层表分开呈现。', '',
              '厚云22.45%是CPP支持子集内的GHI RMSE相对改善，不是总体改善、COT自身反演准确率或未来GHI预报改善。各层RMSE的基准不同，不能把分层百分比直接相加解释总体3.02%。', '',
              '## Interpretation limits', '',
              '- CPP strata describe a sparse keyed subset of the full GHI validation cohort. CPP is a co-produced retrieval reference, not independent truth. Missing reference pixels are excluded only from this diagnostic; the full 4,479-row primary cohort is unchanged.',
              '- R strata classify the model input itself and cover the full cohort; they must be called retrieved-COT groups rather than true-cloud-thickness regimes. Retrieval error can move samples between groups.',
              '- Near-zero COT is not identical to independently verified clear sky. A single centre pixel cannot represent the entire cloud field or sun-path cloud obstruction.',
              '- The intervals are not adjusted for multiple strata. Different gains across bins do not establish a statistically significant interaction or a monotonic physical law.',
              '- All saved predictions and labels retain their original station/time order. CPP sample_indices were keyed to the CLP archive, then joined to GHI by (station,target_time_utc); no row-position join was used across products.',
              '- No hypothesis of forecast gain or extrapolator-only attribution is established by these same-time strata.', '',
              'Outputs: thickness_metrics.json (all seeds, station support and errors, hashes, CIs), thickness_summary.csv, summarize_thickness.py.']
    (OUT/'COT_THICKNESS_GHI_VALIDATION_20261014.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(csvrows,indent=2))

if __name__=='__main__': main()
