"""Plot saved paired forecast-AGRI and real-future-AGRI COT errors."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/regional_station_cot_20260927/sili_trajectories_v2/summary.json'
OUT = ROOT / 'results/paired_forecast_oracle_cot_rmse_20261003_v2'


def main():
    if OUT.exists():
        raise FileExistsError(f'Refuse overwrite: {OUT}')
    d = json.loads(SOURCE.read_text(encoding='utf-8'))
    assert d['test_used'] is False and d['station'] == 'Sili'
    names = {f.name for f in font_manager.fontManager.ttflist}
    font = next(f for f in ['Microsoft YaHei', 'SimHei', 'SimSun'] if f in names)
    plt.rcParams.update({'font.family': font, 'axes.unicode_minus': False, 'font.size': 10})
    rows = []
    for i, lead in enumerate(range(15, 241, 15)):
        row = {'lead_minutes': lead}
        for scope, label in [('station_pixel', 'station'), ('local64', 'patch64')]:
            f = d[scope]['simvp'][i]
            o = d[scope]['oracle'][i]
            assert f['lead_minutes'] == o['lead_minutes'] == lead
            assert f['n'] == o['n'] and f['n'] > 0
            row.update({f'{label}_n': f['n'], f'{label}_forecast_rmse': f['rmse'],
                        f'{label}_oracle_rmse': o['rmse'],
                        f'{label}_oracle_relative_reduction_pct': 100 * (1-o['rmse']/f['rmse'])})
        rows.append(row)
    OUT.mkdir(parents=True)
    with (OUT / 'paired_cot_rmse.csv').open('w', newline='', encoding='utf-8-sig') as file:
        w = csv.DictWriter(file, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lead = np.array([r['lead_minutes'] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), layout='constrained')
    for ax, scope, title in zip(axes, ['station', 'patch64'], ['四里站点像元', '四里站点邻域 64×64（逐像元误差）']):
        ax.plot(lead, [r[f'{scope}_forecast_rmse'] for r in rows], '-o', color='#0072B2',
                markersize=4.2, label='外推未来云图 → R256')
        ax.plot(lead, [r[f'{scope}_oracle_rmse'] for r in rows], '--s', color='#D55E00',
                markersize=4, label='真实未来云图 → R256（oracle）')
        ax.set(title=title, xlabel='距历史末帧的预测时效（分钟）', ylabel='COT RMSE（无量纲）',
               xlim=(10, 245), ylim=(0, 14), xticks=[15, 30, 60, 90, 120, 150, 180, 210, 240])
        ax.grid(alpha=.22)
        ax.legend(loc='upper left', fontsize=9)
    fig.suptitle('同一批旧验证样本：外推云图与真实未来云图的 COT 误差', fontsize=14)
    fig.supxlabel('两条曲线均对照同一 CPP 参考、同一有效掩膜；各时效有效样本数不同。', fontsize=9)
    fig.savefig(OUT / 'paired_cot_rmse.png', dpi=220)
    fig.savefig(OUT / 'paired_cot_rmse.svg')
    plt.close(fig)
    headings = ['时效\n（分钟）', '站点\n外推云图', '站点\n真实未来云图', '64×64\n外推云图', '64×64\n真实未来云图']
    cell = [[str(r['lead_minutes'])]+[f'{r[k]:.2f}' for k in
             ['station_forecast_rmse','station_oracle_rmse','patch64_forecast_rmse','patch64_oracle_rmse']] for r in rows]
    fig, ax = plt.subplots(figsize=(10.4, 8))
    ax.axis('off')
    table = ax.table(cellText=cell, colLabels=headings, cellLoc='center', loc='center',
                     colWidths=[.14,.21,.22,.21,.22], bbox=[.01,.07,.98,.80])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (r,c), item in table.get_celld().items():
        item.set_edgecolor('#D7DFE5')
        item.set_linewidth(.5)
        if r == 0:
            item.set_facecolor('#E5EEF5')
            item.set_text_props(weight='bold')
        elif r % 2 == 0:
            item.set_facecolor('#F5F7F9')
    ax.text(.5,.97,'四里 COT RMSE 逐时效配对表',ha='center',va='top',fontsize=16,transform=ax.transAxes)
    ax.text(.5,.92,'同一 R256；参考为 CPP；数值越小越好',ha='center',va='top',fontsize=10,transform=ax.transAxes)
    ax.text(.02,.025,'旧验证128序列；站点每时效有效N=54～90。真实未来图像仅用于oracle诊断。',fontsize=9,transform=ax.transAxes)
    fig.savefig(OUT / 'paired_cot_rmse_table.png', dpi=220)
    plt.close(fig)
    lines = ['# 外推云图与真实未来云图的COT RMSE配对对照', '',
             '四里；旧验证128条所选序列。按同一目标时刻、同一R256、同一CPP参考与有效掩膜汇总。',
             '本轮从既有summary.json整理指标并绘图，没有重新计算原始128序列误差，没有训练或读取新test载荷。',
             '这不是2026-10-02新按月日期合同的成绩。CPP为检索参考，生产/物理QA未闭合。', '',
             '| 时效/min | 站点N | 站点：外推 | 站点：真实未来 | 64×64：外推 | 64×64：真实未来 |',
             '|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['lead_minutes']} | {r['station_n']} | {r['station_forecast_rmse']:.3f} | {r['station_oracle_rmse']:.3f} | {r['patch64_forecast_rmse']:.3f} | {r['patch64_oracle_rmse']:.3f} |")
    lines += ['', '64×64列是所有有效像元误差的RMSE，先算每个像元的差再汇总；不是先对COT空间平均再求误差。',
              '真实未来AGRI→R256为非部署oracle，仍有反演误差；曲线差不能视为纯外推误差的可加分解，也不代表GHI提升。',
              '每个lead的样本数和像元数不同；缺参考和域外像元不填零。无平滑、插值或未经计算的置信带。', '',
              '## 绘图流程来源', '',
              'Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065 （使用scientific-visualization技能；流程引用不支持性能结论。）']
    (OUT / 'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    sizes={}
    for name in ['paired_cot_rmse.png','paired_cot_rmse_table.png']:
        with Image.open(OUT/name) as im:
            sizes[name]={'pixels':im.size,'dpi':im.info.get('dpi')}
    (OUT / 'provenance.json').write_text(json.dumps(dict(source=str(SOURCE),
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        test_used=False,new_training=False,selected_sequences=128,per_lead=rows,
        font=font,images=sizes),ensure_ascii=False,indent=2),encoding='utf-8')
    print(str(OUT))


if __name__ == '__main__':
    main()
