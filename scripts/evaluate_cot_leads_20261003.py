"""Recompute historical saved COT validation outputs; no new split payloads."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/regional_station_cot_20260927'
OUT = ROOT / 'results/cot_lead_evaluation_20261003_v2'


def metrics(pred, truth, valid):
    e = (pred - truth)[valid]
    thick = valid & (truth >= 30)
    te = (pred - truth)[thick]
    return dict(n=int(e.size), mae=float(np.abs(e).mean()),
                rmse=float(np.sqrt(np.mean(e * e))), bias=float(e.mean()),
                thick_n=int(te.size), thick_rmse=float(np.sqrt(np.mean(te * te))) if te.size else None,
                thick_miss_rate=float(np.mean(pred[thick] < 10)) if te.size else None)


def main():
    if OUT.exists():
        raise FileExistsError(f'Refuse to overwrite {OUT}')
    tracks = SOURCE / 'sili_station_val_audit_20260928_v2/station_tracks.npz'
    corrected = SOURCE / 'simvp_axis_corrected_val_20260928/corrected_simvp_metrics.json'
    oracle_path = SOURCE / 'sili_trajectories_v2/summary.json'
    with np.load(tracks, allow_pickle=False) as z:
        d = {k: z[k].copy() for k in z.files}
    leads = d['lead_minutes'].astype(int)
    assert np.array_equal(leads, np.arange(15, 241, 15))
    truth = d['cpp'].astype(np.float64)
    valid = d['cpp_valid'].astype(bool)
    arms = {k: d[k].astype(np.float64) for k in ['simvp', 'transport', 'persistence']}
    assert truth.shape == valid.shape == (512, 16)
    assert all(a.shape == truth.shape for a in arms.values())
    assert np.isfinite(truth[valid]).all()
    assert all(np.isfinite(a[valid]).all() for a in arms.values())
    audit = json.loads(corrected.read_text(encoding='utf-8'))
    assert audit['test_used'] is False
    rows = []
    for i, lead in enumerate(leads):
        for arm, a in arms.items():
            row = dict(lead_minutes=int(lead), method=arm, **metrics(a[:, i], truth[:, i], valid[:, i]))
            if arm == 'simvp':
                reference = audit['per_lead']['station'][i]
                assert row['n'] == reference['n']
                assert abs(row['rmse'] - reference['rmse']) < 2e-5
            rows.append(row)
    overall = {k: metrics(a, truth, valid) for k, a in arms.items()}
    days = np.unique(d['init_day_BJT'])
    day_sums = {}
    for arm, a in arms.items():
        day_sums[arm] = np.array([
            [np.square(a[d['init_day_BJT'] == day] - truth[d['init_day_BJT'] == day])[valid[d['init_day_BJT'] == day]].sum(),
             valid[d['init_day_BJT'] == day].sum()] for day in days])
    rng = np.random.default_rng(20261003)
    samples = rng.integers(0, len(days), size=(4000, len(days)))
    boot = {k: np.sqrt(v[samples, 0].sum(axis=1) / v[samples, 1].sum(axis=1)) for k, v in day_sums.items()}
    delta_ci = np.quantile(boot['simvp'] - boot['transport'], [0.025, 0.975]).tolist()
    paired = json.loads(oracle_path.read_text(encoding='utf-8'))
    assert paired['test_used'] is False
    oracle_rows = []
    for scope in ['station_pixel', 'local64']:
        for i, lead in enumerate(leads):
            a = paired[scope]['simvp'][i]
            b = paired[scope]['oracle'][i]
            assert a['n'] == b['n'] and a['lead_minutes'] == b['lead_minutes'] == lead
            oracle_rows.append(dict(scope=scope, lead_minutes=int(lead), n=a['n'],
                                    forecast_rmse=a['rmse'], oracle_rmse=b['rmse'],
                                    forecast_mae=a['mae'], oracle_mae=b['mae']))
    OUT.mkdir(parents=True)
    for name, data in [('station_512_per_lead.csv', rows), ('paired_oracle_128_per_lead.csv', oracle_rows)]:
        with (OUT / name).open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in [tracks, corrected, oracle_path]}
    report = dict(state='COMPLETE_HISTORICAL_COT_LEAD_RECOMPUTATION', training_performed=False,
                  new_test_payload_used=False, evaluated_contract='historical pre-20261002 validation only',
                  station='Sili', cohort_sequences=512, initialization_days=len(days),
                  init_day_range=[str(days[0]), str(days[-1])], lead_minutes=leads.tolist(),
                  cot_units='dimensionless physical COT; stored values already expm1 transformed',
                  support='saved CPP-valid intersect pure-transport in-domain support, common to all arms',
                  reference='CPP retrieval reference, unresolved producer/physical QA',
                  per_lead=rows, overall=overall,
                  simvp_minus_transport_rmse=overall['simvp']['rmse']-overall['transport']['rmse'],
                  date_bootstrap_95ci_simvp_minus_transport=delta_ci, bootstrap_repetitions=4000,
                  bootstrap_seed=20261003, sources_sha256=sources,
                  oracle_128_note='Separate 128-sequence cohort/support; never mix with 512-sequence table',
                  figure_description='Left: 512 historical Sili sequences, station and thick-reference COT error. Right: separate paired 128-sequence forecast versus real-future AGRI retrieval. No smoothing or inferred confidence bands.')
    (OUT / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), layout='constrained')
    frows = [r for r in rows if r['method'] == 'simvp']
    trows = [r for r in rows if r['method'] == 'transport']
    axes[0].plot(leads, [r['rmse'] for r in frows], 'o-', color='#0072B2', label='SimVP -> R256: all COT')
    axes[0].plot(leads, [r['rmse'] for r in trows], 's--', color='#D55E00', label='Pure transport: all COT')
    axes[0].plot(leads, [r['thick_rmse'] for r in frows], '^:', color='#CC79A7', label='SimVP -> R256: CPP COT >= 30')
    axes[0].set_title('Sili station | 512 historical validation sequences')
    a = [r for r in oracle_rows if r['scope'] == 'station_pixel']
    axes[1].plot(leads, [r['forecast_rmse'] for r in a], 'o-', color='#0072B2', label='Forecast AGRI -> R256')
    axes[1].plot(leads, [r['oracle_rmse'] for r in a], 'D--', color='#009E73', label='Real-future AGRI -> R256 (oracle)')
    axes[1].set_title('Separate paired 128-sequence cohort | Sili station')
    for ax in axes:
        ax.set(xlabel='Forecast lead from history end (min)', ylabel='RMSE against CPP (dimensionless COT)',
               xlim=(10, 245), ylim=(0, None), xticks=[15, 60, 120, 180, 240])
        ax.grid(alpha=.22)
        ax.legend(fontsize=8, loc='upper left')
    fig.savefig(OUT / 'cot_lead_rmse.png', dpi=180)
    fig.savefig(OUT / 'cot_lead_rmse.pdf')
    plt.close(fig)
    lines = ['# COT 逐时效精度复算（2026-10-03）', '',
             '本报告重算已保存的旧验证输出，不是2026-10-02新按月合同结果。未训练、未读取新test载荷。', '',
             f'四里512条序列、{len(days)}个初始化日（{days[0]}至{days[-1]}）。参考为CPP检索产品；COT无量纲，指标越小越好。共同掩膜为CPP有效且纯输运在域内；各lead样本不同，不是固定公共目标集。', '',
             '## 四里站点：512序列', '',
             '| 时效/min | N | MAE | RMSE | 纯输运RMSE | 厚云N | 厚云RMSE |',
             '|---:|---:|---:|---:|---:|---:|---:|']
    for a, b in zip(frows, trows):
        lines.append(f"| {a['lead_minutes']} | {a['n']} | {a['mae']:.3f} | {a['rmse']:.3f} | {b['rmse']:.3f} | {a['thick_n']} | {a['thick_rmse']:.3f} |")
    lines += ['', '厚云定义为参考CPP COT≥30。每个lead仅9～16个厚云站点样本，且窗口重叠，逐lead厚云误差只能作描述性诊断。', '',
              '## 128序列共同支持上的预测/反演对照', '',
              '此表与上表来自不同cohort。oracle使用真实未来AGRI，不可部署；其误差是当时图像反演误差，不是未来预报误差。', '',
              '| 时效/min | 站点N | 预测站点RMSE | oracle站点RMSE | 预测64×64 RMSE | oracle64×64 RMSE |',
              '|---:|---:|---:|---:|---:|---:|']
    station = [r for r in oracle_rows if r['scope'] == 'station_pixel']
    patch = [r for r in oracle_rows if r['scope'] == 'local64']
    for a, b in zip(station, patch):
        lines.append(f"| {a['lead_minutes']} | {a['n']} | {a['forecast_rmse']:.3f} | {a['oracle_rmse']:.3f} | {b['forecast_rmse']:.3f} | {b['oracle_rmse']:.3f} |")
    lines += ['', '## 解读', '',
              f"1. 512序列的站点总体SimVP→R256 RMSE={overall['simvp']['rmse']:.3f}，纯输运={overall['transport']['rmse']:.3f}；差值={report['simvp_minus_transport_rmse']:+.3f}，按初始化日配对bootstrap 95%区间[{delta_ci[0]:+.3f},{delta_ci[1]:+.3f}]。区间跨零时不能声称稳定优势。", 
              '2. 预测误差随时效总体增加，真实未来图像反演在128序列配对对照中显著较小（描述性差距，不作独立显著性声明）；上游未来云场质量是重要限制。',
              '3. 全部云况RMSE受到低COT样本占比影响，不能用其掩盖厚云误差。厚云漏报阈值和计数见CSV/JSON。',
              '4. 这些COT误差不能直接换算为GHI提升；新按月合同需新S/R及侧车完成后在新val重建同类评估。', '',
              '下一项可用诊断：以初始化时所有16lead共同有效的序列重算固定支持曲线，检查样本变化对曲线趋势的影响；正式新合同仅用新模型评估。', '',
              '## 产物', '', '- `station_512_per_lead.csv`：三种方法的16lead指标。',
              '- `paired_oracle_128_per_lead.csv`：另一个cohort内的配对指标。',
              '- `summary.json`：结果、来源SHA256、掩膜和bootstrap合同。',
              '- `cot_lead_rmse.png/.pdf`：两cohort分别作图，无平滑或虚构误差带。', '',
              '## 绘图流程来源', '',
              '本报告绘图使用scientific-visualization技能；流程参考Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. arXiv:2609.00065（当前记录v2）。https://doi.org/10.48550/arXiv.2609.00065 。这项流程引用不支持COT性能结论。']
    (OUT / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['state','initialization_days','init_day_range','overall','date_bootstrap_95ci_simvp_minus_transport']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
