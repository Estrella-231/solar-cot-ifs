"""Fill verified manuscript sections and draw figures from saved validation scores."""
from pathlib import Path
import json, shutil, hashlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parent
R=P.parent
D=P/'revision_records/20261101_verified_results'
D.mkdir(parents=True,exist_ok=True)
F=P/'figures/verified_results_20261101'
F.mkdir(parents=True,exist_ok=True)
for name in ['main.tex','main_zh.md']:
    if not (D/(name+'.before')).exists(): shutil.copy2(P/name,D/(name+'.before'))
hb=R/'refine-logs/hubei_cloud_probability_20261020/results_fourarm_attempt2'
hn=R/'audits/sili_invalid_day_20261019'
fc=R/'refine-logs/hunan_dydiff_20261028/hunan_full_validation_20261030/recovered/evaluation_attempt1'
hs=pd.read_csv(hb/'metrics_by_station_arm.csv')
bins=pd.read_csv(hb/'metrics_by_cpp_cot_clp_stratum.csv')
boot=json.loads((hb/'paired_two_day_block_bootstrap.json').read_text())
clean=json.loads((hn/'RESULTS.json').read_text())['experiments']['single_frame_A_AC_AP_AL']['after']
leads=pd.read_csv(fc/'ghi_lead_metrics_fp64.csv')
rc=json.loads((fc/'LOCAL_INDEPENDENT_GHI_RECHECK.json').read_text())
assert rc['status']=='PASS'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
blue='#0072B2'; orange='#D55E00'
def save(fig,name):
    for ext in ['pdf','png']: fig.savefig(F/(name+'.'+ext),dpi=220)
    plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(7.05,2.7),layout='constrained')
for ax,reg,names in [(axs[0],'Hubei',['Niushou','Wangzhai','Mizhuang']),(axs[1],'Hunan',['sili','zhujia'])]:
    for j,arm in enumerate(['A','AC']):
        vals=[float(hs[(hs.station==s)&(hs.arm==arm)].rmse_mean_seeds.iloc[0]) if reg=='Hubei' else clean['by_station'][s]['rmse'][arm] for s in names]
        ax.bar(np.arange(len(names))+(j-.5)*.34,vals,.34,color=[blue,orange][j],hatch=['','//'][j],label=['AGRI + geometry','AGRI + geometry + COT'][j])
    ax.set_xticks(range(len(names)),[s.title() for s in names]); ax.set_ylim(0,130); ax.set_title(reg); ax.grid(axis='y',alpha=.18); ax.set_axisbelow(True)
axs[0].set_ylabel('GHI RMSE (W m$^{-2}$)')
fig.legend(*axs[0].get_legend_handles_labels(),loc='outside upper center',ncol=2,frameon=False)
save(fig,'estimation_station')

keys=['CPP_COT_[0,0.1)','CPP_COT_[0.1,5)','CPP_COT_[5,30)','CPP_COT_[30,100]']
sel=[next(x for x in boot if x['stratum']==k and x['ref']=='A' and x['candidate']=='AC') for k in keys]
fig,axs=plt.subplots(1,2,figsize=(7.05,3),layout='constrained')
for j,a in enumerate(['A','AC']):
    v=[bins[(bins.stratum==k)&(bins.arm==a)].rmse_equal_station_seed.iloc[0] for k in keys]
    axs[0].plot(range(4),v,['o-','s--'][j],color=[blue,orange][j],label=['Without COT','With COT'][j])
axs[0].set_ylabel('GHI RMSE (W m$^{-2}$)');axs[0].set_ylim(0,140);axs[0].legend(frameon=False,fontsize=8)
vals=np.array([x['gain_percent_point'] for x in sel]); ci=np.array([x['gain_percent_ci95'] for x in sel])
axs[1].errorbar(range(4),vals,yerr=np.array([vals-ci[:,0],ci[:,1]-vals]),fmt='o',color=blue,capsize=4)
axs[1].axhline(0,color='.4',lw=.8);axs[1].set_ylabel('RMSE reduction (%)');axs[1].set_ylim(-10,60)
for i,x in enumerate(sel): axs[1].annotate(f"n={x['unique_station_time_n']}\nblocks={x['n_blocks']}",(i,ci[i,1]),xytext=(0,6),textcoords='offset points',ha='center',fontsize=7)
for ax in axs:
    ax.set_xticks(range(4),['Near-zero','Thin','Moderate','Thick']);ax.set_xlim(-.5,3.5);ax.grid(axis='y',alpha=.18);ax.set_xlabel('CPP reference COT class')
save(fig,'thickness')

fig,axs=plt.subplots(1,2,figsize=(7.05,3),layout='constrained')
for ax,s in zip(axs,['sili','zhujia']):
    for method,col,mark in [('simvp',blue,'o'),('dydiff',orange,'s')]:
        sub=leads[(leads.station==s)&(leads.method==method)].groupby(['lead_minutes','arm']).rmse.mean().unstack()
        ax.plot(sub.index,sub.M10-sub.M00,marker=mark,markersize=3,color=col,label=method.title())
    ax.axhline(0,color='.4',lw=.8);ax.set_title(s.title());ax.set_xlabel('Forecast lead (min)');ax.set_xticks([15,60,120,180,240]);ax.grid(alpha=.18)
axs[0].set_ylabel('RMSE with COT $-$ without COT\n(W m$^{-2}$)');axs[1].legend(frameon=False)
save(fig,'forecast_lead')

en=(P/'main.tex').read_text(encoding='utf-8')
zh=(P/'main_zh.md').read_text(encoding='utf-8')
def replace(eold,enew,zold,znew):
    global en,zh
    assert en.count(eold)==1,(eold[:80],en.count(eold))
    assert zh.count(zold)==1,(zold[:80],zh.count(zold))
    en=en.replace(eold,enew);zh=zh.replace(zold,znew)

replace(r'\pending{Regional RMSEs with and without COT, relative reductions, and the main difference between cloud conditions.} \pending{Forecasting results and the measured contribution of cloud-image prediction errors.}',
r'On the regional validation sets, adding COT reduces single-image GHI RMSE from 77.29 to 69.14 W\,m$^{-2}$ in Hubei (10.54\%) and from 108.18 to 103.99 W\,m$^{-2}$ in Hunan (3.87\%). The largest reduction in the Hubei reference-covered subsets occurs for thick clouds, although coverage is limited. With independently trained Hunan DyDiff imagery and three-frame irradiance models, predicted COT reduces 15--240 min GHI RMSE from 124.96 to 122.79 W\,m$^{-2}$ (1.74\%).',
'【待填：分区域有无COT的RMSE、相对降幅及主要云况差异。】【待填：预报结果及云图预测误差的影响。】',
'在区域验证集上，加入COT后，单帧GHI反演RMSE在湖北由77.29降至69.14 W/m²（10.54%），在湖南由108.18降至103.99 W/m²（3.87%）。湖北参考产品覆盖的子组中，厚云条件的降幅最大，但覆盖有限。采用湖南独立训练的DyDiff云图及三帧辐照度模型，预测COT使15～240分钟GHI预报RMSE由124.96降至122.79 W/m²（1.74%）。')

replace('The main comparison measures the change in GHI error after adding COT to AGRI and solar geometry in Hubei; the Hunan comparison examines regional consistency. '+r'\pending{Insert matched regional RMSEs, relative reductions, paired intervals, and station-level directions.}',
r'''Adding COT improves single-image GHI estimation in both regional validation sets (Fig.~\ref{fig:gain}). In Hubei, the station- and seed-averaged RMSE decreases from 77.285 to 69.142 W\,m$^{-2}$, a reduction of 10.54\%. The paired difference is $-8.143$ W\,m$^{-2}$ (95\% two-day-block interval: $[-14.999,-3.361]$). MAE decreases from 56.616 to 46.723 W\,m$^{-2}$, and positive bias decreases from 24.799 to 13.236 W\,m$^{-2}$. The evaluation contains 2,195 station--time records in ten two-day blocks, with three random initializations. All three station means improve: Niushou, 76.851 to 63.978; Wangzhai, 70.539 to 63.137; and Mizhuang, 84.466 to 80.312 W\,m$^{-2}$.

The complementary Hunan single-image comparison contains 4,440 station--time records across 51 target dates. RMSE decreases from 108.181 to 103.995 W\,m$^{-2}$ (3.87\%), with a paired difference of $-4.187$ W\,m$^{-2}$ and a 95\% date-block interval of $[-6.652,-1.855]$. Sili improves from 103.558 to 99.877 W\,m$^{-2}$ and Zhujia from 112.805 to 108.113 W\,m$^{-2}$. These independently trained regional comparisons indicate a repeatable benefit for the evaluated configurations, without requiring equal absolute errors or equal percentage gains between regions.''',
'主比较衡量湖北在AGRI与太阳几何之外加入COT带来的GHI误差变化，湖南检验区域一致性。【待填：配对分区域RMSE、相对降幅、区间及逐站方向。】',
'''两个区域验证集的单帧GHI反演均因加入COT而改善（图4）。湖北的站点与种子等权RMSE由77.285降至69.142 W/m²，降低10.54%；配对差为−8.143 W/m²，95%两日日期块区间为[−14.999,−3.361]。MAE由56.616降至46.723 W/m²，正偏差由24.799降至13.236 W/m²。评价包含10个两日日期块内2,195条站点—时次记录及三次随机初始化。三站平均误差均下降：牛首由76.851降至63.978，王寨由70.539降至63.137，米庄由84.466降至80.312 W/m²。

湖南单帧补充比较包含51个目标日期内4,440条站点—时次记录。RMSE由108.181降至103.995 W/m²（3.87%），配对差为−4.187 W/m²，95%日期块区间为[−6.652,−1.855]。四里由103.558降至99.877 W/m²，竺家由112.805降至108.113 W/m²。两区域独立训练的比较说明，这些配置下的收益可在区域间重复，而绝对误差和相对降幅不必相同。''')

replace(r'\pending{Insert the equal-capacity and retrieval-feature contrasts. State whether the observed gain is associated with the cloud-supervised representation and whether explicit COT differs from latent retrieval features. Preserve relevant null or adverse findings.}',
r'''The Hunan representation controls provide context for this improvement (Table~\ref{tab:controls}). Additional image features yield an RMSE of 108.014 W\,m$^{-2}$, close to the baseline, whereas retrieval-model features yield 103.456 W\,m$^{-2}$, comparable to explicit COT. Thus, cloud-supervised features can be useful without expressing that information solely as optical thickness. The result supports COT as a useful representation in this setting, but does not establish that explicit thickness is uniquely responsible for the entire gain. In the Hubei cloud-probability comparison, continuous cloud probability alone yields 74.053 W\,m$^{-2}$; combining probability with COT yields 68.815 W\,m$^{-2}$. Its additional reduction relative to COT alone is 0.47\%, with a 95\% interval of $[-1.23,2.62]$\%, providing no clear overall incremental benefit.''',
'【待填：容量匹配和反演器特征对照。说明增益与云监督表示的关系，以及显式COT是否与隐特征存在差异；保留相关无增益或退化结论。】',
'湖南表示对照为上述改善提供了背景（表3）。额外图像特征的RMSE为108.014 W/m²，与基线接近；反演模型特征的RMSE为103.456 W/m²，与显式COT相近。因此，云监督特征不必仅以光学厚度输出的形式发挥作用。结果支持COT是该配置下有用的表示，但不能确定全部收益独属于显式厚度。湖北连续云概率对照的RMSE为74.053 W/m²；同时加入云概率与COT为68.815 W/m²。相对单独加入COT，额外降幅为0.47%，95%区间为[−1.23,2.62]%，未显示明确的总体增量。')
start=en.index(r'\begin{table}',en.index(r'\subsection{Effect of COT on irradiance estimation}'))
end=en.index(r'\end{table}',start)+len(r'\end{table}')
en=en[:start]+r'''\begin{table}[!htbp]\centering\small
\caption{Hunan single-image representation controls on the same 4,440 validation records. RMSE is averaged equally across two stations and three initializations.}\label{tab:controls}
\begin{tabularx}{\linewidth}{Xrr}\toprule
Input representation & RMSE (W\,m$^{-2}$) & Reduction (\%)\\\midrule
AGRI + solar geometry & 108.181 & ---\\
Additional image features & 108.014 & 0.15\\
Explicit COT & 103.995 & 3.87\\
Retrieval-model features & 103.456 & 4.37\\\bottomrule
\end{tabularx}\end{table}'''+en[end:]
start=zh.index('**表3：');end=zh.index('**图4：',start)
zh=zh[:start]+'''**表3：湖南单帧表示对照，采用相同4,440条验证记录。RMSE对两站和三次随机初始化等权平均。**

| 输入表示 | RMSE（W/m²） | 降幅（%） |
|---|---:|---:|
| AGRI＋太阳几何 | 108.181 | — |
| 额外图像特征 | 108.014 | 0.15 |
| 显式COT | 103.995 | 3.87 |
| 反演模型特征 | 103.456 | 4.37 |

'''+zh[end:]

replace(r'\pending{Insert the common-sample GHI errors and gains for the four COT bins, including date counts and uncertainty. Identify the conditions with the largest improvement and those with no improvement or degradation.}',
r'''The Hubei reference-covered subsets show a marked dependence on optical thickness (Fig.~\ref{fig:thickness}). For near-zero, thin, moderate, and thick reference COT, RMSE decreases from 88.704 to 85.145, from 116.483 to 106.130, from 77.587 to 75.281, and from 57.139 to 34.527 W\,m$^{-2}$, respectively. The corresponding reductions are 4.01\%, 8.89\%, 2.97\%, and 39.57\%. The thick-cloud subset contains 173 station--time records in four two-day blocks; its 95\% reduction interval is $[28.04,50.05]$\%. Near-zero, thin, and moderate subsets contain 268, 71, and 211 records across six, three, and five blocks, respectively, and their reduction intervals include zero. The overall GHI result includes records without CPP and therefore is not a weighted average of these four subsets.''',
'【待填：四档共同样本上的GHI误差与增益、日期数及不确定性；明确最大收益、无改善及退化条件。】',
'湖北参考产品覆盖的子组呈现明显的厚度依赖（图5）。近零、薄云、中等厚度和厚云条件的RMSE分别由88.704降至85.145、116.483降至106.130、77.587降至75.281、57.139降至34.527 W/m²，对应降幅为4.01%、8.89%、2.97%和39.57%。厚云子组包含4个两日日期块内173条站点—时次记录，降幅95%区间为[28.04,50.05]%。近零、薄云和中等厚度子组分别包含268、71、211条记录，覆盖6、3、5个日期块，其降幅区间均包含零。总体GHI评价还包含无CPP参考的记录，因此并非这四个子组的加权平均。')
replace(r'\pending{Relate the signed GHI error to the COT strata, and compare the pattern across stations and solar-geometry conditions. Distinguish an observed subgroup association from a universal monotonic thickness response.}',
r'''In the thick-cloud subset, mean bias decreases from $+36.097$ to $+6.093$ W\,m$^{-2}$, consistent with correcting excessive transmitted irradiance under strong attenuation. This does not imply a universal downward correction: moderate-cloud bias changes from $+14.490$ to $-11.023$ W\,m$^{-2}$, while near-zero bias increases from $+6.218$ to $+14.284$ W\,m$^{-2}$. At Mizhuang, near-zero-COT RMSE increases from 114.775 to 118.948 W\,m$^{-2}$. The strongest aggregate improvement therefore coexists with adverse responses in particular conditions and stations. The non-monotonic reductions and limited reference coverage preclude a general claim that benefit increases continuously with thickness.''',
'【待填：GHI有符号误差与COT分层的关系，及其跨站点、太阳几何条件变化；区分实际分层关联与普遍单调规律。】',
'厚云子组的平均偏差由+36.097降至+6.093 W/m²，与纠正强衰减条件下过高的透射辐照度相符。但这并不意味着统一向下修正：中等厚度子组的偏差由+14.490变为−11.023 W/m²，近零子组则由+6.218增至+14.284 W/m²。米庄近零COT子组的RMSE由114.775升至118.948 W/m²。因此，总体收益可与特定站点和云况下的退化并存。各档降幅并不单调，加上参考覆盖有限，不能推导收益随厚度连续增大的普遍规律。')

replace(r'\pending{Insert the same-sample forecasting RMSEs with and without COT, the relative gain and its uncertainty, and the lead-dependent pattern. Report results separately for irradiance models trained on observed imagery and those trained on forecast imagery where both are evaluated.}',
r'''Predicted COT retains a positive effect when the Hunan irradiance models, trained on observed imagery, are applied without parameter updates to independently trained Hunan DyDiff forecasts. Across 63,480 common GHI records and 51 target dates, the station- and seed-averaged RMSE decreases from 124.958 to 122.788 W\,m$^{-2}$ (1.74\%). The paired difference is $-2.171$ W\,m$^{-2}$, with a 95\% date-block interval of $[-3.685,-0.735]$. Sili improves from 137.222 to 133.815 W\,m$^{-2}$ (2.48\%) and Zhujia from 112.695 to 111.761 W\,m$^{-2}$ (0.83\%); all six station--initialization pairs improve. These are validation results for fixed irradiance models, rather than results from forecast-adapted retraining.

For +45 to +240 min, all three input frames are predicted. On the corresponding 55,673 records, RMSE decreases from 128.555 to 126.308 W\,m$^{-2}$ (1.75\%), with a paired interval of $[-3.776,-0.808]$ W\,m$^{-2}$. The benefit therefore persists beyond the first two leads, which include observed historical frames. Using the same samples and irradiance models with SimVP imagery gives 127.421 versus 125.340 W\,m$^{-2}$ (1.63\%). The lead-dependent contrasts are shown in Fig.~\ref{fig:forecast}. Although DyDiff has lower point RMSEs, paired intervals for its difference from SimVP include zero for both irradiance models; this comparison does not establish superiority of one image predictor.''',
'【待填：同样本有无COT预报RMSE、相对增益、区间与分时效模式；如两者均有结果，基于观测云图训练与基于预测云图训练的辐照度模型分别报告。】',
'''将基于真实云图训练的湖南辐照度模型直接用于湖南独立训练的DyDiff预测云图、保持参数不变时，预测COT仍具有正向作用。在51个目标日期的63,480条共同GHI记录上，站点与种子等权RMSE由124.958降至122.788 W/m²（1.74%）。配对差为−2.171 W/m²，95%日期块区间为[−3.685,−0.735]。四里由137.222降至133.815 W/m²（2.48%），竺家由112.695降至111.761 W/m²（0.83%），六个站点—初始化配对均改善。这些是固定辐照度模型的验证结果，未针对预测输入重新训练。

在+45～+240分钟，三张输入全部为预测图。对应55,673条记录的RMSE由128.555降至126.308 W/m²（1.75%），配对区间为[−3.776,−0.808] W/m²。因此，收益并不限于包含真实历史帧的前两个时效。相同样本及辐照度模型使用SimVP云图时，RMSE为127.421和125.340 W/m²（降低1.63%）。分时效差异见图7。尽管DyDiff点估计误差较低，两种辐照度模型下DyDiff与SimVP差异的配对区间均包含零，因此不能确定哪一种底图模型更优。''')

for label,base,caption,zcaption in [
('gain','estimation_station','Station-wise GHI estimation errors using observed imagery, averaged over three initializations. Hubei uses 2,195 records; Hunan uses 4,440. The regional samples and models are independent.','图4：真实云图输入下逐站GHI反演误差，按三次随机初始化平均。湖北2,195条、湖南4,440条记录；两区域样本与模型独立。'),
('thickness','thickness','Hubei GHI errors (left) and relative RMSE reductions after adding COT (right), conditional on station-centre CPP COT. Error bars show 95% paired two-day-block intervals. Counts denote station--time records and date blocks, not independent pixels.','图5：湖北站点中心CPP COT分层的GHI误差（左）及加入COT后的RMSE降幅（右）。误差棒为95%配对两日日期块区间；标注为站点—时次记录数及日期块数，并非独立像元数。'),
('forecast','forecast_lead','Lead-dependent change in Hunan GHI RMSE after adding predicted COT for two image predictors. Each point averages three initialization-specific RMSE differences at the same station and lead; negative values favour COT. Curves are validation point estimates without lead-wise confidence intervals.','图7：两种底图下加入预测COT后的湖南分时效GHI RMSE变化。每点对同站点、同时效的三次初始化RMSE差等权平均；负值表示COT有益。曲线为验证点估计，未显示逐时效置信区间。')]:
    a=en.index(r'\figslot{',en.index({'gain':r'\subsection{Effect of COT on irradiance estimation}','thickness':r'\subsection{Dependence on optical thickness','forecast':r'\subsection{Effect of COT on forecasting'}[label]))
    b=en.index('\n',a)
    # Captions have literal percent characters only when escaped.
    caption=caption.replace('%',r'\%')
    en=en[:a]+r'\begin{figure}[!htbp]\centering'+'\n'+rf'\includegraphics[width=\linewidth]{{figures/verified_results_20261101/{base}.pdf}}'+'\n'+rf'\caption{{{caption}}}\label{{fig:{label}}}\end{{figure}}'+en[b:]
    n={'gain':'4','thickness':'5','forecast':'7'}[label]
    a=zh.index('**图'+n+'：');b=zh.index('\n',a)
    zh=zh[:a]+'**'+zcaption+'**\n\n'+f'![{zcaption}](figures/verified_results_20261101/{base}.png)'+zh[b:]

replace(r'\pending{Integrate the final control results with the overall and cloud-condition findings.}',
r'The Hunan retrieval-feature contrast suggests that cloud-product supervision contributes a useful representation, while the Hubei thickness analysis identifies a substantial correction of positive bias in the sampled thick-cloud conditions. These findings support a practical value of COT without demonstrating that a scalar optical-thickness field contains uniquely useful information beyond all alternative cloud-supervised features.',
'【待填：将最终控制结果与总体、云况结果结合解释。】',
'湖南反演特征对照提示云产品监督可形成有用表示；湖北厚度分析则显示，在已采样厚云条件下，COT明显纠正了正偏差。这些结果支持COT的实际价值，但不说明标量光学厚度场包含超出所有其他云监督特征的独有信息。')
replace(r'\pending{State the final estimation gain and its regional and cloud-condition dependence.} \pending{State the final forecasting gain under the specified configuration and horizon.}',
r'On the evaluated single-image validation sets, COT reduces regional GHI RMSE by 10.54\% in Hubei and 3.87\% in Hunan. The Hubei reference-covered thick-cloud subset shows a larger reduction, accompanied by a decrease in positive bias, while near-zero and moderate conditions include adverse responses. With Hunan DyDiff predictions and fixed three-frame irradiance models, the 15--240 min RMSE reduction is 1.74\%, and remains 1.75\% when all three input frames are predicted.',
'【待填：最终反演增益及其区域、云况依赖。】【待填：明确配置与时效下的最终预报增益。】',
'在所评价的单帧验证集上，COT使湖北和湖南的区域GHI RMSE分别降低10.54%和3.87%。湖北参考覆盖的厚云子组降幅更大，同时正偏差减小；近零和中等厚度条件下仍存在不利响应。湖南DyDiff预测云图与固定三帧辐照度模型下，15～240分钟RMSE降低1.74%；三张输入全部为预测图时，降幅仍为1.75%。')
for name,txt in [('main.tex',en),('main_zh.md',zh)]: (P/name).write_text(txt,encoding='utf-8')
sources=[hb/'RESULTS.md',hb/'metrics_by_station_arm.csv',hb/'metrics_by_cpp_cot_clp_stratum.csv',hb/'paired_two_day_block_bootstrap.json',hn/'RESULTS.json',fc/'LOCAL_INDEPENDENT_GHI_RECHECK.json',fc/'ghi_lead_metrics_fp64.csv']
(D/'SOURCES.json').write_text(json.dumps({str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2),encoding='utf-8')
(D/'STATUS.md').write_text('''# 2026-11-01 已核实结果回填

同步更新main.tex/main_zh.md：摘要、§5.2总体与表示对照、§5.3湖北COT分层、§5.5湖南预报、§6.1解释与结论已有数值。图4、5、7从保存评分绘制，未使用示意云图。

来源与哈希见SOURCES.json；本脚本用于复现本次修订，后续正文修改后不要直接重跑。修订前两份正文保存在同目录.before文件中。

湖北10.54%是修正R单帧同期结果；湖南3.87%是旧单帧权重按统一无效日排除规则重评分，原选模未重做；容量/隐特征数值仅绑定该湖南队列。未沿用旧错时湖北AL控制或旧3.02%与22.45%分层。湖南1.74%为独立DyDiff第150轮EMA、固定同期训练三帧头的完整验证；不是预报适配重训或独立test。仅+45以后三帧全部预测。

尚保留明确占位：数据与模型详细设置表、正确R的完整COT质量汇总、真实连续事件图、同样本四输入替换与误差归因。已有相关分散分析需要逐项绑定当前模型后再填，不编造归因。尚未完成独立test，未读取test载荷，未新增训练。
''',encoding='utf-8')
print('Updated bilingual manuscript; generated three source-linked figures.')
