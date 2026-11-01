"""Replace the historical main-text event with the current paired models."""
from pathlib import Path

P=Path(__file__).resolve().parents[2]
en=(P/'main.tex').read_text(encoding='utf-8')
zh=(P/'main_zh.md').read_text(encoding='utf-8')


def one(s,old,new):
    assert s.count(old)==1,(old[:90],s.count(old))
    return s.replace(old,new,1)


old='Continuous cases combine actual AGRI composites, retrieved and reference COT, and interval-mean GHI on a common time axis. Event definitions use cloud evolution and data completeness rather than improvement by one model. Cloud passage, growth, and dissipation are examined together with cases where COT does not improve GHI. These cases interpret the aggregate results rather than replace them.'
new=r'''Continuous cases combine observed AGRI composites, retrieved and reference COT, and interval-mean GHI. At Niushou, the first station in the fixed study order, candidate validation dates must contain at least twelve of the seventeen 15-minute targets between 10:30 and 14:30 BJT, a station-pixel retrieved COT peak of at least 30, and a COT range of at least 5. We select the earliest eligible date in each of the improving and non-improving RMSE groups. This conditional selection illustrates contrasting responses without selecting the largest changes; the cases do not estimate their frequency. The retrieved-COT condition is separate from the CPP-based thickness strata.'''
en=one(en,old,new)
zh=one(zh,'连续事件在同一时间轴组合真实AGRI合成图、反演与参考COT、区间平均GHI。事件按云演变与数据完整性定义，不按模型增益选取。分析云团经过、增长、消散，以及COT无改善或退化的情况。个例用于解释总体结果，不代替总体评价。','连续事件组合真实AGRI合成图、反演与参考COT及区间平均GHI。按固定研究顺序的首站牛首筛选验证日期：北京时间10:30–14:30的17个15分钟目标中至少12个有效，站点像元反演COT峰值至少30、极差至少5。随后分别在RMSE改善与未改善两组中取日历最早的合格日期。这种条件选例展示不同方向的响应，不挑选最大变化，也不估计此类事件的频率。反演COT筛选与基于CPP的厚度分层分开定义。')

a=en.index(r'\subsection{Cloud events and irradiance response}')
b=en.index(r'\subsection{Effect of COT on forecasting',a)
en=en[:a]+r'''\subsection{Cloud events and irradiance response}
The two Niushou cases show that COT changes the irradiance response in both directions, rather than imposing a uniform attenuation (Fig.~\ref{fig:events}). On 26 September 2025, station-pixel retrieved COT ranges from 33.5 to 48.2 while observed GHI remains between 50.4 and 107.1 W\,m$^{-2}$. Adding COT raises underestimated late-morning values and reduces early-afternoon overestimation. Averaging the three seed-specific RMSEs gives 27.8 W\,m$^{-2}$ without COT and 19.3 W\,m$^{-2}$ with COT on fifteen common intervals. Retrieved COT is generally lower than the available CPP reference, yet the added representation still improves GHI in this case.

On 27 September, retrieved COT ranges from 18.4 to 31.7 and is generally higher than the CPP reference. At 14:00, observed interval-mean GHI rises to 484.2 W\,m$^{-2}$ while both models remain much lower. The COT-assisted model gives the lower prediction and increases RMSE from 92.7 to 100.9 W\,m$^{-2}$ over sixteen common intervals. The juxtaposition of the cloud field, station COT and GHI shows that a local thickness representation can improve a low-irradiance period but does not guarantee that rapid irradiance changes are captured. These adjacent-date examples are not independent repetitions and do not isolate retrieval error from spatial or temporal representativeness.

\begin{figure}[!htbp]\centering
\includegraphics[width=\linewidth,height=0.73\textheight,keepaspectratio]{figures/current_events_20261102/current_observed_events.pdf}
\caption{Observed-image GHI estimation at Niushou on (a) 26 and (b) 27 September 2025 using the current three-frame models. Each column shows observed AGRI composites and retrieved COT at 10:30, 12:30 and 14:30 BJT, followed by station-pixel COT and GHI trajectories. Composites use C03/C02/C01 with a common square-root transform of reflectance clipped to [0,1]; the native $16\times16$ pixels are retained. Crosses and dashed boxes mark the station and its $3\times3$ input. COT maps and curves share scales; GHI limits differ (0--150 and 0--500 W\,m$^{-2}$). Sparse CPP references are unconnected crosses. GHI observations are target-ending 15-minute means; displayed model curves average three seeds, while RMSE is calculated per seed and then averaged. Missing targets remain gaps. Selection follows Section~4.2 and represents contrasting cases, not their prevalence.}\label{fig:events}
\end{figure}
\FloatBarrier

'''+en[b:]
a=zh.index('### 5.4 云事件与辐照度响应');b=zh.index('### 5.5 ',a)
zh=zh[:a]+'''### 5.4 云事件与辐照度响应

牛首两例表明，COT可以向两个方向改变辐照度响应，并非施加统一衰减（图6）。2025年9月26日，站点像元反演COT为33.5–48.2，实测GHI为50.4–107.1 W/m²。加入COT上调了上午较晚时段的低估值，并降低午后较早时段的高估。15个共同区间的三种子RMSE均值从27.8降至19.3 W/m²。虽然反演COT通常低于可用CPP参考，该表示在此例中仍改善了GHI。

9月27日，反演COT为18.4–31.7，通常高于CPP参考。14:00的实测区间平均GHI升至484.2 W/m²，两种模型的估计均明显偏低。含COT模型给出更低的预测，使16个共同区间的RMSE从92.7升至100.9 W/m²。云场、站点COT与GHI并排展示了局地厚度表示可以改善低辐照过程，但不保证捕捉快速辐照度变化。这两个相邻日期并非独立重复，也不能分离反演误差与空间、时间代表性问题。

![图6](figures/current_events_20261102/current_observed_events.png)

**图6：当前三帧模型在牛首(a)2025年9月26日与(b)9月27日的真实图GHI估计。每列依次展示北京时间10:30、12:30、14:30的真实AGRI合成图和反演COT，以及站点COT和GHI轨迹。合成图采用C03/C02/C01，将反射率统一截至[0,1]后作平方根显示变换，保留原生16×16像元。十字与虚线框标记站点及3×3输入；COT场和曲线使用共同尺度，GHI纵轴分别为0–150和0–500 W/m²。稀疏CPP参考用不连线的叉号显示。GHI观测为以目标时刻结束的15分钟均值，模型曲线为三种子预测均值，RMSE则逐种子计算后平均；缺测目标保留断线。事件按4.2节选择，展示不同方向的案例而非发生频率。**

'''+zh[b:]
en=one(en,'The continuous cases connect these processes to the conditions in which the added representation succeeds or fails.','The continuous cases illustrate contrasting irradiance responses to the added representation, but do not separate these physical processes.')
zh=one(zh,'连续事件将这些过程与表示有效或失效的条件联系起来。','连续事件展示了加入该表示后的不同辐照度响应，但未分离上述物理过程。')
en=one(en,'Figure~\\ref{fig:three_station_event} retains the three station trajectories for the event in Fig.~\\ref{fig:events}. The common initialization and valid-row masks are preserved, with gaps retained where a station lacks a valid 15-minute GHI target. The station curves expand the spatial context of one event; they are not three independently selected events or evidence of an aggregate COT benefit.',r'''Figure~\ref{fig:three_station_event} retains an earlier forecast-source diagnostic initialized at 12:00 BJT on 6 August 2025. This was the earliest eligible correction-development date with all three stations, with initialization chosen nearest noon. It uses a historical-COT baseline and a fixed residual mapping, rather than the current synchronous models in Fig.~\ref{fig:events}. Gaps remain where a station lacks a valid 15-minute GHI target. The station curves describe one event, not three independent repetitions or an aggregate COT benefit.''')
en=one(en,'All source, aggregation and interpretation conventions follow the main event figure.','Displayed model curves average the three frozen-baseline sources of this historical diagnostic; actual future AGRI-derived COT is used only offline.')
zh=one(zh,'图 S2 保留图6事件的三个站点轨迹，初始化和有效样本掩膜一致；缺少有效15分钟 GHI 标签时保留断线。三站轨迹扩展同一事件的空间背景，并不是三个独立选取的事件，也不能证明总体 COT 收益。','图S2保留此前预报来源诊断的三站轨迹，初始化为2025年8月6日北京时间12:00。该日为三站均覆盖的最早合格修正开发日期，初始化取最接近正午的有效时刻。该诊断使用历史COT基线和固定残差映射，与图6的当前同期模型不同。无有效15分钟GHI标签时保留断线。三站曲线描述同一事件，不是三个独立重复，也不代表总体COT收益。')
zh=one(zh,'来源、汇总与解释约定同主事件图。','展示曲线为该历史诊断三份冻结基线来源的均值；实际未来AGRI反演COT仅用于离线诊断。')
(P/'main.tex').write_text(en,encoding='utf-8')
(P/'main_zh.md').write_text(zh,encoding='utf-8')
print('Current synchronous cases integrated bilingually; historical S3 identity corrected.')
