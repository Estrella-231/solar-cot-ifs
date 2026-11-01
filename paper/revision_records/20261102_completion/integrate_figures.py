"""One-time integration into the existing bilingual manuscript; do not replay."""
from pathlib import Path

P = Path(__file__).resolve().parents[2]
en = (P/'main.tex').read_text(encoding='utf-8')
zh = (P/'main_zh.md').read_text(encoding='utf-8')


def replace_once(s, old, new):
    assert s.count(old)==1, (old[:100],s.count(old))
    return s.replace(old,new,1)


def replace_figure(s, label, new):
    i=s.index(r'\label{'+label+'}')
    a=s.rfind(r'\begin{figure}',0,i)
    b=s.index(r'\end{figure}',i)+len(r'\end{figure}')
    return s[:a]+new+s[b:],s[a:b]


en,_=replace_figure(en,'fig:regions',r'''\begin{figure}[!htbp]\centering
\includegraphics[width=\linewidth]{figures/data_coverage_20261102/study_regions_coverage.pdf}
\caption{Station locations and usable dates for observed-image GHI estimation. Panels (a,b) show registration coordinates at different regional scales, without administrative boundaries. In (c), each tick marks a date with at least one valid station sample after the stated quality exclusions; grey and orange ticks distinguish training and validation. Gaps indicate dates absent from the respective usable sample set, rather than zero irradiance. The display concerns paired AGRI--GHI samples, not continuous CPP availability.}\label{fig:regions}
\end{figure}''')
zh=replace_once(zh,'![图1](figures/integrated_20261101/fig01_regions.png)','![图1](figures/data_coverage_20261102/study_regions_coverage.png)')
a=zh.index('**图1：'); b=zh.index('**',a+2)+2
zh=zh[:a]+'**图1：站点位置与真实图GHI估计的有效日期。(a,b)按不同区域尺度展示配准坐标，不绘制行政边界；(c)每条短线代表应用所述质量排除后至少有一个有效站点样本的日期，灰色和橙色分别为训练和验证。间断表示该有效样本集中没有该日期，不代表零辐照度。图中日期为配对AGRI–GHI样本覆盖，不表示CPP连续可用。**'+zh[b:]

caption=r'''Lead-dependent GHI forecast errors for the independently trained regional DyDiff predictors and fixed observed-image-trained irradiance models. (a,b) RMSE without and with predicted COT; (c,d) their paired difference (with minus without COT), with pointwise 95\% intervals from 5,000 target-date-block bootstrap draws. Scores average station- and seed-specific RMSEs equally. Hubei uses 24,451 records from 17 dates and Hunan 66,086 records from 51 dates; counts vary by lead. Grey shading marks the first two leads, which include known historical imagery; all three frames are predicted from +45 min onward. Both models include solar geometry.'''
new=r'''\begin{figure}[!htbp]\centering
\includegraphics[width=\linewidth]{figures/regional_forecast_20261102/regional_forecast_lead.pdf}
\caption{'''+caption+r'''}\label{fig:forecast_latest}
\end{figure}'''
en,old_forecast=replace_figure(en,'fig:forecast_latest',new)
en=replace_once(en,'The lead-dependent contrasts are shown in Fig.~\\ref{fig:forecast_latest}.','The image-predictor contrasts are retained in Fig.~\\ref{fig:predictor_comparison}.')
en=replace_once(en,'Thus the point improvements are not confined to the first two leads containing observed historical frames, although the all-predicted Hubei subset has greater uncertainty.',r'''Thus the point improvements are not confined to the first two leads containing observed historical frames, although the all-predicted Hubei subset has greater uncertainty.

The point estimates favour COT at all sixteen leads in both regions (Fig.~\ref{fig:forecast_latest}). At +60 and +240 min, relative reductions are 6.07\% and 2.06\% in Hubei, and 2.09\% and 1.19\% in Hunan. Some lead-specific date-block intervals include zero, particularly at longer horizons in Hubei; the overall improvements therefore do not establish an individually resolved benefit at every lead.''')
old_forecast=old_forecast.replace(r'\label{fig:forecast_latest}',r'\label{fig:predictor_comparison}')
en=replace_once(en,r'\end{document}',r'''\subsection*{S6. Image-predictor comparison on common Hunan samples}
The comparison retains 63,480 common validation records across 51 target dates for the two image predictors and the same observed-image-trained three-frame irradiance models. It is separate from the expanded regional DyDiff comparison in Table~\ref{tab:regional_forecast} and Fig.~\ref{fig:forecast_latest}. Both methods show a negative point RMSE difference after adding predicted COT; this comparison does not establish that one image predictor has a larger COT benefit.
'''+old_forecast+'\n\\FloatBarrier\n\\end{document}')
zh=replace_once(zh,'分时效差异见图7。','两种底图的分时效对照保留于补充图S9。')
zh=replace_once(zh,'因此，点估计改善并不限于包含真实历史帧的前两个时效，但湖北全预测输入子集的不确定性更大。','''因此，点估计改善并不限于包含真实历史帧的前两个时效，但湖北全预测输入子集的不确定性更大。

两区域16个时效的点估计均显示加入COT后误差降低（图7）。湖北在+60和+240分钟的相对RMSE降幅分别为6.07%和2.06%，湖南对应为2.09%和1.19%。部分时效的日期块区间包含零，湖北长时效尤其如此；总体改善不等于每个时效的收益均能单独与零区分。''')
zh=replace_once(zh,'![图7](figures/verified_results_20261101/forecast_lead.png)','''![图7](figures/regional_forecast_20261102/regional_forecast_lead.png)

**图7：两区域独立训练的DyDiff与真实图训练的固定辐照度模型产生的逐时效GHI预报误差。(a,b)为不加COT与加入预测COT的RMSE；(c,d)为配对差值（加COT减不加COT），阴影表示5,000次目标日期块重采样得到的逐时效95%区间。先逐站逐种子计算RMSE，再等权平均。湖北为17个日期的24,451条记录，湖南为51个日期的66,086条记录，各时效记录数不同。灰色背景标示仍含已知历史图的前两个时效；+45分钟起三帧均为预测图。两组均包含太阳几何。**''')
zh+='''

### S6 湖南共同样本上的底图比较

两种底图使用相同的真实图训练三帧辐照度模型，比较保留51个目标日期的63,480条共同验证记录。它与表4、图7扩展后的区域DyDiff比较分开报告。两种方法加入预测COT后的RMSE差值点估计均为负，但该比较不能确定哪种底图的COT增益更大。

![图S9](figures/verified_results_20261101/forecast_lead.png)

**图S9：湖南两种底图加入预测COT后的逐时效GHI RMSE变化，评价范围为51个日期的63,480条共同验证记录。每个点为同一站点、同一时效下三个种子RMSE差值的均值；负值表示COT改善。曲线为验证点估计，本图不显示逐时效置信区间。**
'''
(P/'main.tex').write_text(en,encoding='utf-8')
(P/'main_zh.md').write_text(zh,encoding='utf-8')
print('Integrated real date coverage and current regional forecast figures; retained predictor comparison in S6.')
