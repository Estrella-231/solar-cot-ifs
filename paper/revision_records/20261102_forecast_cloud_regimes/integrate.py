"""One-shot integration from the saved 2026-11-02 manuscript; no rescoring."""
from pathlib import Path
import csv, json, shutil, hashlib

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
ROOT = PAPER.parent
SOURCE = ROOT / 'refine-logs/forecast_thick_cloud_events_20261102'
before = HERE / 'before'
before.mkdir(exist_ok=True)
for name in ['main.tex', 'main_zh.md', 'FIGURE_PLAN.md', 'PAPER_PLAN.md', 'VERSION_INDEX.md']:
    dst = before / name
    if not dst.exists(): shutil.copy2(PAPER/name, dst)
tex = (before/'main.tex').read_text(encoding='utf-8')
tex = tex.replace(r'\setlength{\emergencystretch}{2em}', r'\setlength{\emergencystretch}{2em}'+'\n'+r'\makeatletter\setlength{\@fptop}{0pt}\makeatother')
zh = (before/'main_zh.md').read_text(encoding='utf-8')

def once(text, old, new):
    assert text.count(old) == 1, (old[:100], text.count(old))
    return text.replace(old, new, 1)

tex = once(tex, 'The largest reduction, 35.51\\%, occurs in the thick-cloud subset covered by the satellite reference product', 'The largest observed-image reduction, 35.51\\%, occurs in the thick-cloud subset covered by the satellite reference product')
tex = once(tex, 'Controlled input replacements further show that predicted COT contains useful cloud-evolution information, while GHI forecasts remain sensitive to errors in that representation.', 'Forecast gains depend on cloud regime: thick-cloud subgroup intervals include zero, and moderate-COT conditions in Hubei deteriorate. Input replacements and real cloud sequences indicate useful predicted evolution alongside unresolved errors in optical thickening and thinning.')
zh = once(zh, '卫星参考产品覆盖的厚云子组降幅最大，为35.51%', '真实图输入下，卫星参考产品覆盖的厚云子组降幅最大，为35.51%')
zh = once(zh, '控制输入替换进一步表明，预测COT保留了有用的云演变信息，但GHI预报仍对该表示中的误差敏感。', '预报收益依赖云况：两区厚云子组区间均包含零，湖北中等COT条件则出现退化。输入替换与真实云序列表明，预测COT含有有用的演变信息，同时仍未充分再现光学增厚与减薄。')

method_en = r'''For forecasting, each station--target time is assigned a reference class once, irrespective of how many forecast initializations include it. Consecutive 15-minute thick-reference times are joined; a missing reference, non-thick value, or temporal gap ends the segment. These are station reference segments, not independently tracked cloud objects. Within each region, the earliest multi-time segment in each of the improving and degrading groups is selected. Group assignment uses target-balanced errors over all available origins and three seeds. The displayed initialization maximizes segment coverage, with ties resolved by proximity to a 60-minute advance, without using its forecast error. Displayed curves use seed 42. Image snapshots are taken 30 minutes before the reference peak, at the peak, and at the last forecast target with valid GHI. Future observations are used only for this offline diagnosis. Reference COT at the interval end need not describe the entire 15-minute GHI interval.'''
tex = once(tex, r'\subsection{Effects of cloud-image prediction errors on COT and GHI}', method_en + '\n\n' + r'\subsection{Effects of cloud-image prediction errors on COT and GHI}')
method_zh = '''预报分析中，相同站点—目标时刻无论被多少次初始化覆盖，仅判定一次参考类别。将连续15分钟的厚云参考时刻连接；遇到缺参考、非厚云或时间中断即结束一段。这是站点参考连续段，并非独立追踪的物理云团。每区域分别从改善和退化两组中选择日历最早的多时刻段；分组依据全部可用起报、三个种子的目标时刻等权误差。图示起报优先覆盖最多的段内目标，再以最接近提前60分钟确定，不依据该起报的误差挑选。曲线固定采用种子42；图像快照为参考峰值前30分钟、峰值时刻及具有有效GHI的最后一个预报目标。未来观测仅用于离线诊断；区间末端CPP厚云不代表整个15分钟GHI区间持续厚云。'''
zh = once(zh, '### 4.3 云图预测误差对COT和GHI的影响', method_zh+'\n\n### 4.3 云图预测误差对COT和GHI的影响')
tex = once(tex, r'\pending{Complete the aligned COT trajectories and all four input combinations before assigning the image-route, COT-route, and interaction terms in Eq.~\ref{eq:propagation}.}', r'The current regional three-frame evaluation does not include the complete four-way substitution in Eq.~\ref{eq:propagation}; the reported COT-source contrasts are therefore input sensitivities rather than a quantitative allocation of error among the image route, COT route, and their interaction.')
tex = once(tex, r'\pending{Use the aligned cloud fields and GHI trajectories to identify whether cloud displacement, weakened optical-thickness evolution, or delayed growth/decay accounts for the error pattern. State the supported role of image extrapolation in limiting the forecast gain.}', '')
zh = once(zh, '【待填：完成对齐COT轨迹及四种输入组合后，再按式(10)量化图像路径、COT路径及交互作用。】', '当前区域三帧评价尚未包含式(10)的完整四组合替换，因此已报告的COT来源对照是输入敏感性分析，不能量化分配图像路径、COT路径及交互项各自的误差。')
zh = once(zh, '【待填：利用对齐云场与GHI轨迹，识别云团位移、厚度演变减弱或生消延迟对应的误差模式；写出证据所支持的底图外推对预报增益的限制。】', '')

result_en = r'''\subsection{Cloud-regime dependence of forecasting}
The overall forecast improvement does not extend uniformly to every reference thickness class (Fig.~\ref{fig:forecast_regimes}). Under thick reference COT, Hubei RMSE decreases from 72.479 to 61.281 W\,m$^{-2}$ (15.45\%), and Hunan from 92.134 to 89.464 W\,m$^{-2}$ (2.90\%). The paired target-date-block intervals are $[-25.927,+1.089]$ and $[-5.821,+0.690]$ W\,m$^{-2}$, respectively; both include zero. These subsets contain 139 distinct station--target observations on six dates in Hubei and 165 on 25 dates in Hunan. In contrast, Hubei moderate-COT RMSE increases from 86.965 to 94.089 W\,m$^{-2}$, with a difference interval of $[+4.320,+24.961]$ W\,m$^{-2}$.

Thick-cloud mean GHI bias decreases from $+31.986$ to $+16.072$ W\,m$^{-2}$ in Hubei and from $+34.456$ to $+22.821$ W\,m$^{-2}$ in Hunan. Thus the point improvement is associated with reduced positive bias, but the available reference dates do not resolve a stable thick-cloud forecast benefit. Only 32.6\% and 28.4\% of the regional forecast records have a quality-valid station CPP reference. The stratified results describe this covered subset; they neither replace the full-cohort forecast scores nor isolate the loss of benefit relative to observed-image estimation.
\begin{figure}[!htbp]\centering
\includegraphics[width=\linewidth]{figures/forecast_cloud_regimes_20261102/forecast_regimes.pdf}
\caption{GHI forecast errors conditional on target-time CPP optical thickness in Hubei and Hunan. Left: RMSE without and with predicted COT. Right: the paired difference (with minus without COT) and 95\% intervals from 5,000 target-date-block draws; negative values favour COT. Scores first average the three seed-specific RMSEs within each station and then weight stations equally. Repeated origins are retained in scoring; labels count distinct station--target observations and dates. Classes are $[0,0.1)$, $[0.1,5)$, $[5,30)$, and $[30,100]$. CPP is an offline retrieval reference, and near-zero does not by itself establish clear sky.}\label{fig:forecast_regimes}
\end{figure}
\FloatBarrier

The reference merging yields 75 Hubei and 137 Hunan station segments, but only 18 and seven, respectively, contain at least two consecutive thick-reference times. This fragmentation limits process-level inference. Figure~\ref{fig:forecast_events} contrasts one improving and one degrading multi-time segment, while all four selected cases and exact coverage are retained in Supplementary material S7. At Mizhuang on 26 September 2025, the forecast COT weakens before the sustained thick reference does, yet including it reduces part of the GHI overestimation. For the displayed initialization and seed, thick-segment RMSE decreases from 26.67 to 14.51 W\,m$^{-2}$ over nine targets.

At Sili on 15 June 2025, the forecast COT misses renewed local thickening and later remains elevated as the observed cloud becomes optically thinner. The displayed thick-segment GHI RMSE increases from 14.87 to 22.10 W\,m$^{-2}$ over three targets. Later in the forecast, both GHI models miss much of the observed irradiance recovery. The juxtaposition shows that a useful aggregate COT contribution can coexist with imperfect cloud evolution and adverse event corrections. The examples describe conditional responses; they do not uniquely separate image-prediction, retrieval, or station-mapping errors.
\begin{figure}[!htbp]\centering
\includegraphics[width=\linewidth,height=0.74\textheight,keepaspectratio]{figures/forecast_cloud_regimes_20261102/forecast_events.pdf}
\caption{Contrasting forecast cases: (a) Mizhuang, 26 September 2025, initialized at 06:15 BJT; (b) Sili, 15 June 2025, initialized at 11:15 BJT. Observed AGRI, CPP and predicted COT snapshots accompany station-COT and GHI trajectories. Composites use C03/C02/C01 and a common square-root transform of reflectance clipped to [0,1]. Native pixels are retained: Hubei fields are $16\times16$; the archived Hunan forecast COT is a magnified $3\times3$ crop, not the full observed field of view. Crosses mark the station; boxes locate its $3\times3$ neighbourhood. COT maps share a scale, while GHI axis limits differ. Grey maps and curve gaps indicate missing values. Shading marks the selected thick-reference segment, and dotted lines mark snapshots. GHI curves use one initialization and seed 42; observations are target-ending 15-minute means. Future observed inputs appear only as offline references. Cases follow the outcome-conditional selection in Section~4.2.}\label{fig:forecast_events}
\end{figure}
\FloatBarrier

'''
result_zh = '''### 5.6 预报的云况依赖

总体预报改善并不均匀延伸至每个参考厚度层（图9）。厚云参考条件下，湖北RMSE由72.479降至61.281 W/m²（15.45%），湖南由92.134降至89.464 W/m²（2.90%）。对应目标日期块配对区间分别为[−25.927,+1.089]和[−5.821,+0.690] W/m²，均包含零。湖北厚云子组含139个不同站点—目标时次、6个日期，湖南含165个、25个日期。相反，湖北中等COT组RMSE由86.965升至94.089 W/m²，差值区间为[+4.320,+24.961] W/m²。

厚云GHI平均偏差在湖北由+31.986降至+16.072 W/m²，在湖南由+34.456降至+22.821 W/m²。因此点估计改善伴随正偏差减小，但现有参考日期尚不能确定稳定的厚云预报收益。两区域分别仅32.6%和28.4%的预报记录具有质量有效的站点CPP参考。分层结果描述这一覆盖子集，不替代全样本预报成绩，也不能据此分离相对真实图估计的收益损失。

![图9](figures/forecast_cloud_regimes_20261102/forecast_regimes.png)

**图9：湖北与湖南按目标时刻CPP光学厚度分层的GHI预报误差。左：无COT与加入预测COT的RMSE；右：配对差（加COT减无COT）及5,000次目标日期块重采样95%区间，负值有利COT。先在每站平均三个种子分别计算的RMSE，再对站点等权。评分保留重复起报；标签统计不同站点—目标时次及日期数。分层为[0,0.1)、[0.1,5)、[5,30)、[30,100]。CPP是离线检索参考，近零本身不等于已确认晴空。**

参考归并得到湖北75段、湖南137段站点连续段，但分别仅18段和7段含至少两个相邻厚云参考时刻。这种碎片化限制过程解释。图10对照一例改善和一例退化的多时刻段，全部四例及准确覆盖保留在补充材料S7。2025年9月26日米庄案例中，预测COT比持续厚云参考提前减弱，但加入COT仍减轻了部分GHI高估。所展示起报和种子在9个厚云目标上的RMSE由26.67降至14.51 W/m²。

2025年6月15日四里案例中，预测COT未捕捉局地再次增厚，随后在观测云场光学减薄时仍保持较高。图示3个厚云目标的GHI RMSE由14.87升至22.10 W/m²。预报后期，两种GHI模型均未跟上观测辐照度的大幅恢复。这些并排证据说明，总体COT增量可与云演变失真及个例不利修正共存。个例描述条件响应，不能唯一分离云图预测、反演或站点映射误差。

![图10](figures/forecast_cloud_regimes_20261102/forecast_events.png)

**图10：不同方向的预报案例：(a)米庄，2025年9月26日06:15北京时间起报；(b)四里，2025年6月15日11:15起报。真实AGRI、CPP和预测COT快照与站点COT、GHI轨迹并排。合成图使用C03/C02/C01，反射率统一截断至[0,1]后取平方根。保留原生像元：湖北场为16×16，湖南已归档预测COT是放大的3×3裁块，不是完整观测视场。十字为站点，方框为3×3邻域；COT场使用共同色标，两例GHI纵轴范围不同。灰色场图与曲线断口表示缺测，背景阴影标记所选厚云参考段，竖直点线标快照。GHI曲线为单次起报、固定种子42；观测是以目标时刻结束的15分钟均值。未来真实输入仅作离线参考，案例遵循4.2节按误差方向分组后的选择规则。**

'''
tex = once(tex, r'\section{Discussion}', result_en+r'\section{Discussion}')
zh = once(zh, '## 6 讨论', result_zh+'## 6 讨论')

discuss_en = r'''The cloud-regime analysis adds a qualification to this aggregate benefit. Thick-cloud positive bias decreases, but both regional thick-subset intervals include zero, and Hubei moderate-COT errors increase. The Mizhuang example improves despite errors in predicted optical thickness, whereas the Sili example shows that adding COT does not recover the missing irradiance rise. COT accuracy and incremental GHI value are therefore related but not interchangeable: their connection depends on the joint spectral inputs and the irradiance mapping.'''
tex = once(tex, 'Errors in predicted cloud images can affect GHI directly', discuss_en+'\n\nErrors in predicted cloud images can affect GHI directly')
zh = once(zh, '云图预测误差既可以通过光谱和空间输入直接影响GHI', '云况分析进一步限定了总体收益的含义。厚云正偏差减小，但两区厚云子组区间均包含零，湖北中等COT误差反而增加。米庄案例在预测光学厚度仍有误差时改善，四里案例则表明加入COT未能恢复遗漏的辐照度上升。因此，COT精度与GHI增量相关却不能互相替代，两者关系取决于联合光谱输入及辐照度映射。\n\n云图预测误差既可以通过光谱和空间输入直接影响GHI')
limit_en = r'''The analysis uses development-validation dates, with the reserved test partitions unevaluated. Sparse CPP coverage limits cloud-regime and continuous-event conclusions: most reference segments contain a single time, and endpoint COT is not an interval-integrated cloud measurement. The cases and input sensitivities constrain possible explanations but do not attribute errors uniquely to cloud motion, optical evolution, retrieval, or radiative effects.'''
tex = once(tex, r'\pending{Consolidate the final evaluation scope and the principal residual limitation here, without repeating operational status in the results.}', limit_en)
zh = once(zh, '【待填：集中写最终评价范围及最主要剩余限制，不在结果中反复插入运行状态。】', '本分析使用开发验证日期，保留测试分区尚未评价。CPP覆盖稀疏限制云况和连续事件结论：多数参考段只有一个时刻，区间端点COT也不是区间积分云观测。个例和输入敏感性缩小了可能解释的范围，但尚不能将误差唯一归于云运动、光学演变、反演或辐射效应。')
tex = once(tex, 'These results extend the observed-input evidence to an overall forecasting benefit in both regions.', 'These results extend the observed-input evidence to an overall forecasting benefit in both regions, while unresolved thick-cloud subgroup intervals and moderate-COT degradation in Hubei demonstrate its cloud-regime dependence.')
zh = once(zh, '这些结果将真实图输入下的证据延伸至两区域的总体预报收益。', '这些结果将真实图输入下的证据延伸至两区域的总体预报收益，但厚云子组区间跨零、湖北中等COT退化，说明该收益依赖云况。')

# Supplemental exact tables are generated from saved metrics, never hand rescored.
rows = list(csv.DictReader((SOURCE/'strata_metrics.csv').open(encoding='utf-8')))
rows = [r for r in rows if r['scope']=='all' and r['stratum'] in ['Near-zero','Thin','Moderate','Thick']]
supp_en = r'''\clearpage
\subsection*{S7. Forecast cloud regimes and complete real-cloud cases}
The reference-covered forecast analysis retains the original GHI scoring records and applies CPP classes only offline. Table~\ref{tab:forecast_support} distinguishes forecast records, distinct station--target observations, and contiguous reference segments. Multiple origins do not create additional reference observations; segments do not count independent cloud systems.
\begin{table}[!htbp]\centering\small
\caption{CPP coverage within the regional forecast validation samples. A multi-time segment contains at least two consecutive 15-minute thick-reference times.}\label{tab:forecast_support}
\begin{tabularx}{\linewidth}{Xrr}\toprule
Quantity & Hubei & Hunan\\\midrule
All GHI forecast records & 24,451 & 66,086\\
CPP-valid forecast records & 7,976 & 18,750\\
CPP-valid distinct station--targets & 508 & 1,235\\
Thick-reference forecast records & 2,141 & 2,435\\
Thick-reference distinct station--targets & 139 & 165\\
Thick-reference target dates & 6 & 25\\
Reference segments / multi-time segments & 75 / 18 & 137 / 7\\\bottomrule
\end{tabularx}\end{table}
\begin{table}[!htbp]\centering\small
\caption{Forecast GHI RMSE by reference COT class. Units are W\,m$^{-2}$; differences are predicted-COT minus no-COT scores. Intervals use 5,000 paired target-date draws. $N/D$ counts distinct station--targets/dates.}\label{tab:forecast_strata}
\begin{tabularx}{\linewidth}{llrrXr}\toprule
Region & Class & No COT & + COT & $\Delta$ [95\% interval] & $N/D$\\\midrule
'''
supp_zh = '''\n\n### S7 预报云况分层与完整真实云案例

参考覆盖预报分析保留原GHI评分记录，仅在离线分析中按CPP分类。表S6区分预报记录、不同站点—目标时次与参考连续段；重复起报不增加参考观测数，段数也不代表独立云系统数。

**表S6：区域预报验证集的CPP覆盖。多时刻段至少包含两个相邻15分钟厚云参考时刻。**

|数量|湖北|湖南|
|---|---:|---:|
|全部GHI预报记录|24,451|66,086|
|CPP有效预报记录|7,976|18,750|
|CPP有效不同站点—目标时次|508|1,235|
|厚云参考预报记录|2,141|2,435|
|厚云参考不同站点—目标时次|139|165|
|厚云目标日期|6|25|
|参考段／多时刻段|75／18|137／7|

**表S7：CPP参考COT分层的预报GHI RMSE，单位W/m²。差值为加预测COT减无COT；区间来自5,000次配对目标日期抽样，N/D表示不同站点—目标时次/日期。**

|区域|类别|无COT|加COT|Δ [95%区间]|N/D|
|---|---|---:|---:|---:|---:|
'''
cn={'Hubei':'湖北','Hunan':'湖南','Near-zero':'近零','Thin':'薄云','Moderate':'中等','Thick':'厚云'}
for r in rows:
    lo,hi=json.loads(r['delta_ci95'])
    vals=[f"{float(r[k]):.3f}" for k in ['A_rmse','AC_rmse','delta_rmse']]
    ci=f'[{lo:+.3f}, {hi:+.3f}]'
    nd=f"{r['n_station_targets']}/{r['n_dates']}"
    supp_en+=f"{r['region']} & {r['stratum']} & {vals[0]} & {vals[1]} & ${vals[2]}$ [{lo:+.3f}, {hi:+.3f}] & {nd}\\\\\n"
    supp_zh+=f"|{cn[r['region']]}|{cn[r['stratum']]}|{vals[0]}|{vals[1]}|{vals[2]} {ci}|{nd}|\n"
supp_en+=r'''\bottomrule\end{tabularx}\end{table}
\par\FloatBarrier

As a sensitivity analysis, averaging squared errors over origins before scoring each distinct station--target gives thick-subset RMSEs of 72.559 versus 61.234 W\,m$^{-2}$ in Hubei and 90.838 versus 87.570 W\,m$^{-2}$ in Hunan. Leaving out one date at a time gives paired differences from $-14.616$ to $-6.830$ and from $-3.720$ to $-2.147$ W\,m$^{-2}$, respectively. These influence checks do not replace the main intervals that include zero.

For forecast COT itself, thick-cloud probability of detection, false-alarm ratio and critical success index are 0.717, 0.349 and 0.518 in Hubei, and 0.446, 0.365 and 0.355 in Hunan, at the same threshold of 30. Mean COT bias on thick-reference instances is $-10.997$ and $-15.280$, respectively. These dimensionless scores pool CPP-valid forecast instances rather than using the station--seed averaging of the GHI measures. Different regional reference coverage precludes interpreting them as a controlled regional model ranking.

The four cases below retain the earliest eligible improving and degrading multi-time segment per region. Each complete figure shows observed and forecast AGRI, CPP, COT retrieved from observed AGRI, COT retrieved from forecast AGRI, and aligned COT/GHI time series. Spatial arrays and processing match Fig.~\ref{fig:forecast_events}; missing observations remain missing. The Hunan forecast panels are explicitly magnified native $3\times3$ archived crops. The case scores below concern only the displayed initialization and seed 42, not all origins or the regional estimator.
'''
supp_zh+='''
作为敏感性分析，先对同一站点—目标时次的各起报平方误差平均，再评分，厚云RMSE在湖北为72.559与61.234 W/m²，在湖南为90.838与87.570 W/m²。逐一留出一个日期时，配对差范围分别为[−14.616,−6.830]和[−3.720,−2.147] W/m²。这些影响诊断不替代包含零的主区间。

对预测COT本身，阈值30下的厚云检出率、虚警比例与临界成功指数在湖北为0.717、0.349、0.518，在湖南为0.446、0.365、0.355。厚云参考实例上的COT平均偏差分别为−10.997与−15.280。这些无量纲指标汇总CPP有效预报实例，不采用GHI的站点—种子等权评分。两区参考覆盖不同，不能作为控制其他条件后的区域模型排名。

下列四例保留每区域日历最早的合格改善与退化多时刻段。每幅完整图展示真实/预测AGRI、CPP、真实AGRI反演COT、预测AGRI反演COT及对齐的COT/GHI序列。空间数组及显示处理与图10一致，缺测不补值。湖南预测场明确为放大的原生3×3归档裁块；下列个例误差只针对图示起报、种子42，并非全部起报或区域评分。
'''
case_metrics=list(csv.DictReader((SOURCE/'case_display_metrics.csv').open(encoding='utf-8')))
for i,r in enumerate(case_metrics,1):
    stem=f"case_{i:02d}_{r['event_id']}"
    # Package the preserved full evidence figures as supplement assets.
    dest=PAPER/'figures/forecast_cloud_regimes_20261102/supplement'
    dest.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','png']: shutil.copy2(SOURCE/'figures'/f'{stem}.{ext}',dest/f'{stem}.{ext}')
    sta=r['station'].capitalize(); date=r['start'][:10]; start=r['start'][11:16]; end=r['end'][11:16]; init=r['init'][11:16]
    a=float(r['shown_seed42_origin_thick_A']); c=float(r['shown_seed42_origin_thick_AC'])
    cap=f"{sta}, {r['region']}, {date}; initialized at {init} BJT. The selected thick-reference segment is {start}--{end}. Displayed-seed segment RMSE is {a:.2f} without COT and {c:.2f} W\\,m$^{{-2}}$ with predicted COT. The future observed fields are offline references, not forecast inputs."
    supp_en+='\n'+r'\clearpage\begin{figure}[!htbp]\centering'+'\n'+r'\includegraphics[width=\linewidth,height=0.83\textheight,keepaspectratio]{figures/forecast_cloud_regimes_20261102/supplement/'+stem+'.pdf}\n'+r'\caption{'+cap+'}'+r'\label{fig:forecast_case_'+str(i)+'}\n'+r'\end{figure}'+'\n'
    names={'Mizhuang':'米庄','Niushou':'牛首','Zhujia':'竺家','Sili':'四里'}
    supp_zh+=f"\n![图S{9+i}](figures/forecast_cloud_regimes_20261102/supplement/{stem}.png)\n\n**图S{9+i}：{cn[r['region']]}{names[sta]}，{date}，{init}北京时间起报。所选厚云参考段为{start}–{end}；图示种子在段内的RMSE为无COT {a:.2f}、加预测COT {c:.2f} W/m²。未来真实场仅作离线参考，不是预报输入。**\n"
tex=once(tex,r'\end{document}',supp_en+'\n'+r'\end{document}')
zh+=supp_zh
(PAPER/'main.tex').write_text(tex,encoding='utf-8')
(PAPER/'main_zh.md').write_text(zh,encoding='utf-8')
(HERE/'source_manifest.json').write_text(json.dumps({
    'source_dir':str(SOURCE),'before_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in before.iterdir()},
    'analysis_changed':False,'model_or_data_changed':False,'paper_authorized':True,
    'main_figures':['forecast_regimes','forecast_events'],'supplement_cases':4,
    'selection':'all reference classes; case1 and case4 main; all four preserved in S7',
    'still_not_completed':'Full four-way current-model input substitution and independent test',
},ensure_ascii=False,indent=2),encoding='utf-8')
print('Integrated bilingual manuscript, S7 and preserved original source snapshots.')
