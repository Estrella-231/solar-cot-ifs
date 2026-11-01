"""Recompute manuscript table material from archived validation-only evidence.

This script does not train, evaluate checkpoints, change source artifacts, or
open test/buffer payloads. Tables keep the old Hunan geometry contract explicit.
"""
from pathlib import Path
from datetime import datetime, timedelta
import csv
import hashlib
import json
import shutil
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARMS = ['A', 'AC', 'AP', 'AL']
SEEDS = [42, 43, 44]
NAMES = {'A': 'AGRI baseline', 'AC': 'AGRI + COT',
         'AP': 'AGRI + auxiliary features', 'AL': 'AGRI + retrieval features'}
SOURCES = {}


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def source(rel):
    p = ROOT / rel
    SOURCES[rel] = sha(p)
    return p


def write_csv(name, rows):
    with (HERE/name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def jread(rel):
    return json.loads(source(rel).read_text('utf-8'))


def main():
    audit = jread('audits/sili_invalid_day_20261019/RESULTS.json')
    record = audit['experiments']['single_frame_A_AC_AP_AL']
    original = source(record['source'])
    assert sha(original) == record['source_sha256']
    clean = source('audits/sili_invalid_day_20261019/single_frame_A_AC_AP_AL_dayclean.npz')
    with np.load(original, allow_pickle=False) as z:
        old = {k: z[k] for k in z.files}
    with np.load(clean, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    old_dates = np.array([(datetime.fromisoformat(str(t))+timedelta(hours=8)).date().isoformat()
                          for t in old['target_time_utc']])
    keep = ~((old['station'] == 'sili') & (old_dates == '2024-09-05'))
    assert (~keep).sum() == 39 and np.all(old['ghi_truth_wm2'][~keep] == 0)
    for k in old:
        np.testing.assert_array_equal(old[k][keep], data[k])
    assert set(data['split']) == {'val'}
    station, truth, dates = data['station'], data['ghi_truth_wm2'], old_dates[keep]
    assert len(truth) == 4440 and len(np.unique(dates)) == 51
    assert len(set(zip(station, data['target_time_utc']))) == len(truth)
    preds = np.array([[data[f'{a}_seed{s}_ghi_wm2'] for a in ARMS] for s in SEEDS])
    assert np.isfinite(preds).all() and np.isfinite(truth).all()
    days = np.unique(dates)
    weights = np.random.default_rng(20261019).multinomial(len(days), np.full(len(days), 1/len(days)), 2000)
    seed_rows, summary_rows, points, boots = [], [], {}, {}
    for site in ['sili', 'zhujia']:
        mask = station == site
        err = preds[:, :, mask] - truth[mask]
        squared = err**2
        rmse = np.sqrt(squared.mean(axis=-1))
        points[site] = rmse.mean(axis=0)
        for si, seed in enumerate(SEEDS):
            for ai, arm in enumerate(ARMS):
                seed_rows.append(dict(station=site, seed=seed, arm=arm, input=NAMES[arm],
                    n=int(mask.sum()), n_dates=int(len(np.unique(dates[mask]))),
                    rmse_wm2=float(rmse[si, ai]), mae_wm2=float(np.abs(err[si, ai]).mean()),
                    bias_wm2=float(err[si, ai].mean())))
        index = np.searchsorted(days, dates[mask])
        counts = np.bincount(index, minlength=len(days))
        sums = np.stack([squared[:, :, index == d].sum(axis=-1) for d in range(len(days))], axis=-1)
        den = weights @ counts
        assert (den > 0).all()
        boots[site] = np.sqrt(np.einsum('bd,sad->bsa', weights, sums)/den[:, None, None]).mean(axis=1)
    points['Equal-station mean'] = np.stack([points[s] for s in ['sili', 'zhujia']]).mean(axis=0)
    boots['Equal-station mean'] = np.stack([boots[s] for s in ['sili', 'zhujia']]).mean(axis=0)
    for site, values in points.items():
        rows = [r for r in seed_rows if site == 'Equal-station mean' or r['station'] == site]
        for ai, arm in enumerate(ARMS):
            chosen = [r for r in rows if r['arm'] == arm]
            summary_rows.append(dict(station=site, arm=arm, input=NAMES[arm],
                n=len(truth) if site == 'Equal-station mean' else int((station == site).sum()),
                n_dates=len(days) if site == 'Equal-station mean' else len(np.unique(dates[station == site])),
                rmse_wm2=float(values[ai]), mae_wm2=float(np.mean([r['mae_wm2'] for r in chosen])),
                bias_wm2=float(np.mean([r['bias_wm2'] for r in chosen]))))
    contrasts = []
    for site in points:
        for cand, ref in [('AC','A'),('AP','A'),('AL','A'),('AC','AP'),('AL','AC')]:
            a, b = ARMS.index(cand), ARMS.index(ref)
            delta = float(points[site][a] - points[site][b])
            ci = np.quantile(boots[site][:, a]-boots[site][:, b], [.025,.975])
            contrasts.append(dict(station=site, comparison=f'{cand} minus {ref}',
                reference_rmse_wm2=float(points[site][b]), candidate_rmse_wm2=float(points[site][a]),
                delta_rmse_wm2=delta, ci95_low=float(ci[0]), ci95_high=float(ci[1]),
                rmse_reduction_pct=float(-100*delta/points[site][b])))
    expected = record['after']
    for ai, arm in enumerate(ARMS):
        np.testing.assert_allclose(points['Equal-station mean'][ai], expected['station_seed_equal_RMSE'][arm], atol=1e-11, rtol=0)
    for row in contrasts:
        if row['station'] == 'Equal-station mean' and row['comparison'] in expected['contrasts']:
            e = expected['contrasts'][row['comparison']]
            np.testing.assert_allclose([row['delta_rmse_wm2'], row['ci95_low'], row['ci95_high']], [e['delta'], *e['CI95']], atol=1e-11, rtol=0)
    write_csv('hunan_singleframe_station_seed_metrics.csv', seed_rows)
    write_csv('hunan_singleframe_station_arm_metrics.csv', summary_rows)
    write_csv('hunan_singleframe_paired_contrasts.csv', contrasts)
    write_csv('hunan_singleframe_sample_support.csv', [dict(station=s, bjt_target_date=d, n=int(np.sum((station==s)&(dates==d))))
        for s in ['sili','zhujia'] for d in np.unique(dates[station==s])])
    # Source/weight receipts: do not silently relabel old-geometry R as corrected R.
    pack = jread('experiments/monthly_R_station16_20261013_v1/results/ghi_head_pack_new_r_v1/pack_manifest.json')
    assert pack['R_checkpoint_sha256'].startswith('34259609')
    headmeta = jread('experiments/monthly_R_station16_20261013_v1/results/ghi_heads_new_r_seed42_v1/sili/run_metadata.json')
    jread('experiments/monthly_R_station16_20261013_v1/results/ghi_heads_three_seed_cpu_reload_audit.json')
    jread('experiments/monthly_R_station16_20261013/results/r_al_formal_t18_20261014_v1/fp64_audit_v3/fp64_independent_audit.json')
    source('audits/hunan_cpp_time_alignment_20261020/REPORT.md')
    source('experiments/monthly_R_station16_20261013_v1/train_ghi_heads.py')
    source('experiments/monthly_R_station16_20261013_v1/head_candidate_v0.py')
    for rel in [
        'audits/hubei_cpp_time_impact_20261020/REPORT.md',
        'audits/hunan_cpp_time_alignment_20261020/checkpoint_payloads.json',
        'refine-logs/hubei_cloud_probability_20261020/AUDIT_GEOMETRY_CPP_ALIGNMENT_20261020.md',
        'refine-logs/hubei_cloud_probability_20261020/results_fourarm_attempt2/RESULTS.md',
        'refine-logs/hubei_solar_geometry_repair_20261021/EXPERIMENT_PLAN.md',
        'refine-logs/hubei_solar_geometry_repair_20261021/receipts/attempt12/RESULTS.md',
        'refine-logs/hubei_forecast_cot_increment_20261022/RESULT_REPORT.md',
        'refine-logs/hubei_forecast_cot_increment_20261022/evidence_attempt1/PREPARATION.json',
        'refine-logs/hubei_forecast_cot_increment_20261022/evidence_attempt1/PREFLIGHT.json',
        'refine-logs/hubei_forecast_cot_increment_20261022/evidence_attempt1/CHECKPOINT_SHA256.txt',
        'refine-logs/hubei_forecast_cot_increment_20261022/code_attempt1/prepare.py',
        'refine-logs/dydiff_handoff_20261028/REPORT.md',
    ]:
        source(rel)
    forecast_dir = 'paper/figures/redesign_20261030/evidence/'
    fm = jread(forecast_dir+'forecast_ghi_manifest.json')
    for entry in fm['sources']:
        p = source(entry['path'])
        assert sha(p) == entry['sha256'], entry['path']
    for old_name, new_name in [('station_seed_arm_metrics.csv','hubei_forecast_station_seed_metrics.csv'),
            ('paired_comparisons.csv','hubei_forecast_paired_contrasts.csv'),
            ('lead_metrics.csv','hubei_forecast_lead_metrics.csv'),
            ('sample_support.csv','hubei_forecast_sample_support.csv')]:
        shutil.copyfile(source(forecast_dir+old_name), HERE/new_name)
    with (HERE/'hubei_forecast_station_seed_metrics.csv').open(encoding='utf-8-sig') as f:
        forecast_rows = list(csv.DictReader(f))
    fsummary = []
    for site in ['Niushou','Wangzhai','Mizhuang','Equal-station mean']:
        for arm in ['G0','G1','G2','G3']:
            rows=[r for r in forecast_rows if r['arm']==arm and (site=='Equal-station mean' or r['station']==site)]
            fsummary.append(dict(station=site, arm=arm, input=rows[0]['input'],
                n=fm['unique_station_rows'] if site=='Equal-station mean' else int(rows[0]['n']),
                n_dates=len(fm['target_dates']),
                rmse_wm2=float(np.mean([float(r['rmse_wm2']) for r in rows])),
                mae_wm2=float(np.mean([float(r['mae_wm2']) for r in rows])),
                bias_wm2=float(np.mean([float(r['bias_wm2']) for r in rows]))))
    write_csv('hubei_forecast_station_arm_metrics.csv', fsummary)
    lines=['# 可复算表格素材 / Reproducible table material','',
           '所有 Δ=候选−参照；负值表示 RMSE 降低。汇总为逐站、逐 seed RMSE 的等权平均，不是集成预测。', '',
           '## 表 1：湖南同期单帧固定权重质量重评分（旧几何合同，补充诊断）','',
           '|站点|有效行/日期|AGRI|AGRI+COT|辅助特征对照|反演特征对照|COT−AGRI [95% CI]|',
           '|---|---:|---:|---:|---:|---:|---:|']
    for site in points:
        rows=[r for r in summary_rows if r['station']==site]
        c=next(r for r in contrasts if r['station']==site and r['comparison']=='AC minus A')
        vals='|'.join(f'{v:.3f}' for v in points[site])
        lines.append(f"|{site}|{rows[0]['n']}/{rows[0]['n_dates']}|{vals}|{c['delta_rmse_wm2']:+.3f} [{c['ci95_low']:+.3f}, {c['ci95_high']:+.3f}]|")
    lines += ['', '单位 W/m²。三 seed；2,000 次北京时间目标日期配对 bootstrap。保留固定权重；剔除已确认无效的四里 2024-09-05 后重算。模型曾按剔除前验证集选模；本表不是重训或独立测试。R 的 CPP 对时正确，但使用旧太阳几何，不能代表最新修正几何合同。', '',
              '## 表 2：湖北未来 GHI 四臂配对验证（可用的预报扩展证据）','',
              '|站点|有效行/日期|无 COT|历史 COT|历史+对照修正头|历史+预测 COT 修正头|',
              '|---|---:|---:|---:|---:|---:|']
    for site in ['Niushou','Wangzhai','Mizhuang','Equal-station mean']:
        rows=[r for r in fsummary if r['station']==site]
        vals='|'.join(f"{r['rmse_wm2']:.3f}" for r in rows)
        lines.append(f"|{site}|{rows[0]['n']}/{rows[0]['n_dates']}|{vals}|")
    lines += ['', '单位 W/m²。过去 8 帧至未来 16 帧，+15…+240 min；三 seed。15,275 个不同站点预报行、22 个目标日期；不是 45,825 个独立天气样本。后两臂共用已训练历史 COT 基底，修正头参数相同。', '',
              '## 表 3：湖北未来 GHI 的两项可解释配对比较','',
              '|比较|站点|ΔRMSE|日期块 95% CI|','|---|---|---:|---:|']
    with (HERE/'hubei_forecast_paired_contrasts.csv').open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            lines.append(f"|{r['comparison']}|{r['station']}|{float(r['delta_rmse_wm2']):+.3f}|[{float(r['ci95_low']):+.3f}, {float(r['ci95_high']):+.3f}]|")
    lines += ['', '5,000 次目标日期配对 bootstrap。按站区间为描述性点区间，未作多重比较校正。整体历史 COT 收益区间跨零；未来 COT 相对同容量对照的额外收益未获支持。不能据此唯一归因于 SimVP，也不构成同期 GHI 主结果。']
    (HERE/'TABLES.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    manifest = dict(state='RECOMPUTED_HUNAN_AND_REVERIFIED_FORECAST_SOURCES', test_payloads_read=0,
        buffer_payloads_read=0, new_training=False, source_mutations=False,
        hunan=dict(R_checkpoint_sha256=pack['R_checkpoint_sha256'], parameter_counts=headmeta['parameter_counts'],
                   n=4440, target_dates=51, seeds=SEEDS, geometry='legacy; not corrected-geometry result',
                   bootstrap=dict(unit='BJT target date', resamples=2000, seed=20261019)),
        forecast=dict(n=fm['unique_station_rows'], target_dates=fm['target_dates'], seeds=fm['seeds'],
                      bootstrap=fm['bootstrap'], upstream_geometry='corrected 20261021',
                      verified_source_files=len(fm['sources'])), sources=SOURCES)
    manifest['outputs']={p.name:sha(p) for p in HERE.glob('*.csv')}
    manifest['outputs']['TABLES.md']=sha(HERE/'TABLES.md')
    manifest['script_sha256']=sha(Path(__file__))
    (HERE/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(state=manifest['state'],hunan=points['Equal-station mean'].tolist(),
                         hunan_rows=len(truth), forecast_sources=len(fm['sources']),
                         output_files=len(manifest['outputs'])),ensure_ascii=False))


if __name__ == '__main__':
    main()
