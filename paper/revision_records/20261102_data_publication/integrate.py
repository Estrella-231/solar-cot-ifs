"""One-time integration of six publication-styled data figures; no result changes."""
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for name in ('main.tex', 'main_zh.md'):
    backup = HERE / (name + '.before')
    if not backup.exists():
        shutil.copy2(ROOT / name, backup)

tex = (ROOT / 'main.tex').read_text(encoding='utf-8')
zh = (ROOT / 'main_zh.md').read_text(encoding='utf-8')
assert tex.count('figures/visual_refresh_20261102/') == 6
assert zh.count('figures/visual_refresh_20261102/') == 6
tex = tex.replace('figures/visual_refresh_20261102/', 'figures/publication_data_20261102/')
zh = zh.replace('figures/visual_refresh_20261102/', 'figures/publication_data_20261102/')

def caption(text, label, replacement):
    pattern = r'\\caption\{[^\n]*\}\\label\{' + re.escape(label) + r'\}'
    new, count = re.subn(pattern, lambda _: '\\caption{' + replacement + '}\\label{' + label + '}', text)
    assert count == 1, (label, count)
    return new

tex = caption(tex, 'fig:regions', r'Station locations and usable dates for observed-image GHI estimation. (a) Hubei and Hunan study regions and station locations; (b) local detail of the three Hubei stations. Provincial boundaries are from Natural Earth. In (c), each tick marks a date with at least one valid station sample after the stated quality exclusions; grey-blue and orange ticks distinguish training and validation. Gaps indicate dates absent from the respective usable sample set, rather than zero irradiance. The display concerns paired AGRI--GHI samples, not continuous CPP availability.')
tex = caption(tex, 'fig:retrieval', r'Hubei corrected-geometry COT retrieval: valid-pixel counts against CPP (left) and mean bias within discrete reference thickness classes (right). Scores use all 67,187 valid pixels across 264 frames and 16 target dates. CPP is the retrieval reference. This evaluates the same frozen R used to generate the GHI COT inputs.')
tex = caption(tex, 'fig:gain', r'Observed three-frame Hubei GHI estimation. Left: station-specific and equally averaged RMSE, first computed per random seed; grey segments connect the two input groups within each row. Right: paired COT-minus-baseline RMSE differences with 95\% target-date-block intervals from 5,000 draws; negative values favour COT. Counts are 663, 607 and 635 validation station--time records at Niushou, Wangzhai and Mizhuang, respectively.')
tex = caption(tex, 'fig:thickness', r'Hubei GHI errors (left) and paired relative reductions (right) conditional on valid station-centre CPP reference COT. Grey segments connect errors with and without COT within each class; positive reductions favour COT. Horizontal intervals use 5,000 paired target-date draws, retaining fixed station--seed groups. Labels give common station--time records and target dates; records without CPP remain in the overall GHI evaluation.')
tex = caption(tex, 'fig:events', r'Observed-image GHI estimation at Niushou on (a) 26 and (b) 27 September 2025 using the current three-frame models. Each row shows observed AGRI composites and retrieved COT at 10:30, 12:30 and 14:30 BJT on the left, with station-pixel COT and GHI trajectories on the right; vertical dotted guides mark snapshot times. Each snapshot is the last frame of the corresponding three-frame input. Composites use C03/C02/C01 with a common square-root transform of reflectance clipped to [0,1]; the native $16\times16$ pixels are retained. Crosses and dashed boxes mark the station and its $3\times3$ input. COT maps and curves share scales; GHI limits differ (0--150 and 0--500 W\,m$^{-2}$). Sparse CPP references are unconnected crosses. GHI observations are target-ending 15-minute means; displayed model curves average three seeds, while RMSE is calculated per seed and then averaged. Missing targets remain gaps. Selection follows Section~4.2 and represents contrasting cases, not their prevalence.')

zh_captions = {
    1: '图1：站点位置与真实图GHI估计的有效日期。(a)湖北、湖南研究区域及站点位置；(b)湖北三站局地放大。省级边界来自Natural Earth。(c)每条短线代表应用所述质量排除后至少有一个有效站点样本的日期，灰蓝色和橙色分别为训练和验证。间断表示该有效样本集中没有该日期，不代表零辐照度。图中日期为配对AGRI–GHI样本覆盖，不表示CPP连续可用。',
    4: '图4：湖北修正几何COT反演：相对CPP的有效像元计数（左）和离散参考厚度分层的平均偏差（右）。共67,187个有效像元、264帧、16个目标日期；CPP为检索参考，反演器与GHI的COT输入来源相同且固定。',
    5: '图5：湖北真实三帧同期GHI估计。左：逐站和等权平均RMSE，先分别计算各随机种子；灰色线段仅连接同一行的两种输入结果。右：加COT减基线的RMSE差及5,000次目标日期块重采样95%区间；负值有利COT。牛首、王寨、米庄分别663、607、635条验证站点—时次记录。',
    6: '图6：有效站点中心CPP参考COT分层的湖北GHI误差（左）和配对相对降幅（右）。灰色线段连接同一类别中有无COT的误差，正降幅表示改善。横向区间采用5,000次配对目标日期重采样，保持固定站点—种子组；标注为共同站点—时次和目标日期数，无CPP记录仍参与总体GHI评价。',
    7: '图7：当前三帧模型在牛首(a)2025年9月26日与(b)9月27日的真实图GHI估计。每行左侧展示北京时间10:30、12:30、14:30的真实AGRI合成图和反演COT，右侧展示站点COT和GHI轨迹；竖直点线标记快照时刻。每幅快照是对应三帧输入的末帧。合成图采用C03/C02/C01，将反射率统一截断至[0,1]后作平方根显示变换，保留原生16×16像元。十字与虚线框标记站点及3×3输入；COT场和曲线使用共同尺度，GHI纵轴分别为0–150和0–500 W/m²。稀疏CPP参考用不连线的叉号显示。GHI观测为以目标时刻结束的15分钟均值，模型曲线为三种子预测均值，RMSE则逐种子计算后平均；缺测目标保留断线。事件按4.2节选择，展示不同方向的案例而非发生频率。',
}
for n, body in zh_captions.items():
    zh, count = re.subn(r'\*\*图' + str(n) + r'：[^\n]*\*\*', lambda _: '**' + body + '**', zh)
    assert count == 1, (n, count)

(ROOT / 'main.tex').write_text(tex, encoding='utf-8')
(ROOT / 'main_zh.md').write_text(zh, encoding='utf-8')
print('Replaced six main figure assets and synchronized five changed captions; scientific results unchanged.')
