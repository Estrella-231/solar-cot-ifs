from pathlib import Path
import re

root = Path(__file__).resolve().parent
en_path, zh_path = root/'main.tex', root/'main_zh.md'
en, zh = en_path.read_text(encoding='utf-8'), zh_path.read_text(encoding='utf-8')
archive = root/'revision_records/20261018'
archive.mkdir(parents=True, exist_ok=True)
for p in (en_path, zh_path):
    q = archive/(p.name+'.before')
    if not q.exists(): q.write_bytes(p.read_bytes())

def span(text, start, end, new):
    a=text.index(start); b=text.index(end,a)
    return text[:a]+new+'\n\n'+text[b:]

abstract = r'''Cloud optical thickness (COT) describes cloud attenuation, but its value for irradiance estimation is uncertain when the underlying multispectral imagery is already available. We assess COT-assisted global horizontal irradiance (GHI) estimation at five stations in Hunan and Hubei, with separate regional training. A cloud-product-supervised retriever supplies local COT fields to a compact network receiving AGRI imagery and solar geometry. Across three random initializations, adding COT reduced station-averaged validation RMSE from 123.980 to 120.240 W m$^{-2}$ in Hunan and from 74.333 to 70.993 W m$^{-2}$ in Hubei, reductions of 3.02\% and 4.49\%, respectively. The reduction reached 22.45\% in the Hunan thick-cloud subset defined by the CPP reference product. Daily irradiation mean absolute error also decreased from 0.5491 to 0.5032 kWh m$^{-2}$ over 30 complete Hunan observation days. However, COT did not outperform matched retrieval features or improve a stronger AGRI residual model. In a separate short-term forecasting comparison, adding forecast-derived COT changed GHI RMSE from 77.150 to 78.155 W m$^{-2}$, with a paired confidence interval spanning zero. These validation results identify a benefit of COT in compact satellite irradiance estimators, concentrated in some cloud conditions, but do not establish an advantage over richer image representations or a corresponding forecasting gain.'''
en=span(en, r'\begin{abstract}',r'\end{abstract}',r'\begin{abstract}'+'\n'+abstract)
zh=span(zh,'## 摘要','## 1 引言','''## 摘要

云光学厚度（COT）描述云对辐射的衰减，但在多光谱图像已经可用时，显式COT能否进一步改善辐照度估计仍需检验。本文在湖南和湖北五个站点开展区域独立训练，评价COT辅助全球水平辐照度（GHI）估计。通过云产品监督训练的反演器生成局地COT场，与AGRI图像及太阳几何共同输入紧凑估计网络。三次随机初始化结果显示，加入COT后，湖南站点平均验证RMSE从123.980降至120.240 W m⁻²，湖北从74.333降至70.993 W m⁻²，分别降低3.02%和4.49%。在CPP参考产品定义的湖南厚云子组中，降幅为22.45%；在30个观测完整的湖南日期上，日辐照量平均绝对误差从0.5491降至0.5032 kWh m⁻²。然而，COT未优于匹配的反演特征，也未改善更强的AGRI残差模型。另在短临预报比较中，加入预测图像反演的COT后，GHI RMSE从77.150变为78.155 W m⁻²，配对差值置信区间跨零。验证结果表明，COT可改善紧凑卫星辐照度估计器，且收益集中于部分云况；相对于更丰富的图像表示及短临预报，其额外收益尚未得到支持。''')

# Methods describe comparisons, not their outcomes.
en=span(en,'The matched forecasting comparison trains',r'\FloatBarrier',r'''The forecasting comparison uses predicted AGRI alone, predicted AGRI with COT, and predicted AGRI with an additional image-feature branch. All models are trained on forecast inputs and evaluated at identical stations, initialization times, and target times. The image-feature branch projects 32 channels to four, yielding 24,613 trainable parameters compared with 24,489 for the COT model. A separate sensitivity analysis holds the synchronous GHI model fixed while replacing AGRI and COT inputs individually with their forecast counterparts. This distinguishes the response to each input from the performance of models trained for forecasting.''')
zh=span(zh,'匹配预报对照在预测输入上分别训练','## 4 实验设计','''预报比较包括仅用预测AGRI、预测AGRI与COT融合，以及预测AGRI与辅助图像特征融合三种模型。各模型均使用预测输入训练，并在相同站点、初始化时刻和目标时刻评价。辅助图像分支将32个特征通道投影为4个，模型可训练参数为24,613，COT模型为24,489。另固定同期GHI模型，分别将AGRI和COT替换为其预测值，分析模型对两类输入的敏感性；该分析与重新训练后的预报模型评价分别进行。''')

# Preserve exploratory results as bilingual supplementary analyses.
e_start=en.index('A separate single-seed screen asked')
e_end=en.index('In a separate fixed-head sensitivity diagnostic',e_start)
supp_en=en[e_start:e_end].strip()
z_start=zh.index('另一个单种子') if '另一个单种子' in zh else -1
if z_start<0:
    candidates=[m for m in re.finditer(r'[^\n]*单种子[^\n]*',zh) if '3,140' in m.group()]
    assert len(candidates)==1, len(candidates)
    z_start=candidates[0].start()
z_end=zh.index('另一项固定头敏感性诊断',z_start)
supp_zh=zh[z_start:z_end].strip()

forecast_en=r'''\subsection{Application to short-term forecasting}
The COT advantage in synchronous estimation did not carry over to the matched Hubei forecasting comparison. Across three stations, three random initializations, and 22 target dates, GHI RMSE was 77.150 W m$^{-2}$ for forecast AGRI alone, 78.155 for forecast AGRI with COT, and 78.980 for the additional image-feature control. The COT-minus-imagery difference was $+1.005$ W m$^{-2}$ (95\% paired date-block interval $[-0.722,+3.780]$). Differences for the image-feature control relative to imagery alone and relative to COT were $+1.830$ ($[-0.517,+4.778]$) and $+0.825$ ($[-2.657,+4.402]$) W m$^{-2}$, respectively. None of these comparisons resolved an improvement; the intervals also do not establish equivalence. A two-day-block analysis gave a similar interval for the COT comparison, $[-0.681,+3.813]$ W m$^{-2}$.

Forecast COT nevertheless contained information about cloud evolution. In Hunan, SimVP followed by COT retrieval achieved a station-and-lead-averaged RMSE of 10.664 on the available local $3\times3$ CPP reference pixels, compared with 20.487 for last-frame persistence. Thick-cloud CSI was 0.423 and 0.306, respectively (Fig.~\ref{fig:forecast}). Equal-budget MSE, temporal-change, and spatial-gradient fine-tuning gave RMSEs of 10.750, 10.756, and 10.759. Direct forecasting of historical COT also failed to improve these scores; the exploratory comparisons and cloud-event diagnostics are reported in Appendix~\ref{sec:forecast-supp}.

Input-replacement analysis helps explain why COT forecast skill need not translate into GHI improvement. For a Hunan model trained on observed inputs, RMSE over 2,817 sequences and 52 dates was 118.14 W m$^{-2}$ with observed AGRI and its retrieved COT, 125.42 when only COT was forecast-derived, 126.46 when only AGRI was forecast, and 139.39 when both inputs were forecast-derived. Replacing COT increased RMSE by 7.28 W m$^{-2}$ (95\% interval $[4.52,10.54]$) with observed AGRI and by 12.93 ($[9.73,16.58]$) with forecast AGRI. These hindsight comparisons reveal sensitivity to input error and distribution shift; they do not isolate the cause of the matched forecasting result in Hubei.'''
en=span(en,r'\subsection{Application to short-term forecasting}',r'\begin{figure}[!htbp]\centering'+'\n'+r'\includegraphics[width=0.94\linewidth]',forecast_en)
forecast_zh='''### 5.6 短临预报应用

同期估计中的COT收益没有在湖北匹配预报比较中延续。在三个站点、三次随机初始化和22个目标日期上，仅用预测AGRI的GHI RMSE为77.150 W m⁻²，加入COT后为78.155，增加辅助图像特征后为78.980。COT相对仅用图像的差值为+1.005 W m⁻²（配对日期块95%区间[−0.722, +3.780]）。辅助图像特征相对仅用图像、相对COT的差值分别为+1.830（[−0.517, +4.778]）和+0.825（[−2.657, +4.402]）W m⁻²。各项比较均未显示明确改善，区间也不确立模型等效。连续两日块分析给出的COT比较区间相近，为[−0.681, +3.813] W m⁻²。

预测COT仍包含云演变信息。在湖南可用的局地3×3 CPP参考像元上，SimVP预测图像经COT反演后的站点及时效平均RMSE为10.664，末帧持续性为20.487；厚云CSI分别为0.423和0.306（图7）。等预算MSE、时间变化和空间梯度微调的RMSE分别为10.750、10.756和10.759。直接预测历史COT同样未改善这些分数；探索性比较和云事件诊断见附录B。

输入替换分析进一步说明，COT预报技巧不一定转化为GHI改善。对于使用观测输入训练的湖南模型，在2,817个序列、52个日期上，观测AGRI及其反演COT对应的RMSE为118.14 W m⁻²；仅COT来自预测图像时为125.42，仅AGRI采用预测值时为126.46，两者均来自预测时为139.39。在观测AGRI条件下替换COT使RMSE增加7.28 W m⁻²（95%区间[4.52, 10.54]），在预测AGRI条件下增加12.93（[9.73, 16.58]）。这些事后比较反映输入误差和分布变化的影响，不能单独解释湖北匹配预报结果的成因。'''
zh=span(zh,'### 5.6 ',r'\begin{figure}[!htbp]\centering',forecast_zh)

discussion_en=r'''\section{Discussion}
Retrieved COT improved the compact GHI estimator in both regions, but the comparison with richer representations shows that its value depends on the predictor. A similarly sized image-derived branch did not reproduce the full gain, whereas features from the COT retriever performed comparably to explicit COT in Hubei. The stronger AGRI residual model also showed no additional benefit from COT. Together, these results suggest that cloud-product supervision can provide a useful representation for a compact estimator, without establishing that the scalar optical-thickness output is superior to the information retained in learned features.

The largest Hunan improvement occurred in the CPP thick-cloud subset, where adding COT reduced positive GHI bias. This is consistent with better representation of cloud attenuation, although cloud thickness alone does not determine station irradiance. Cloud position relative to the solar beam, diffuse radiation, and changes within the 15-minute averaging interval can also affect the relationship. The event examples show that even a pronounced COT change does not guarantee a better GHI trajectory. Moreover, the retriever sees a $16\times16$ patch whereas the GHI imagery branch sees $3\times3$; the comparison therefore does not separate cloud-property supervision from wider spatial context.

For solar-resource assessment, the lower Hunan daily irradiation MAE is encouraging, but the more negative daily bias shows that an improvement in absolute error can coexist with systematic underestimation. Forecasting introduces a further limitation: predicted COT outperformed persistence, yet adding it did not improve the matched GHI forecasts. This difference is consistent with sensitivity to errors in cloud evolution and to a changed input distribution. The present comparisons cannot attribute it uniquely to the extrapolation model, the retriever, or their interaction with the GHI estimator.

The study is limited to five stations in two separately trained regions. CPP is a retrieval reference with sparse temporal coverage, and the cloud-event examples cover summer only. In addition, model selection and the reported comparisons use validation data; independent testing is needed to assess generalization. These limitations delimit the result: COT is useful within the tested compact estimation framework, but its incremental value in stronger predictors and forecasting remains unresolved.'''
conclusion_en=r'''\section{Conclusion}
Adding retrieved COT to a compact satellite GHI estimator reduced validation RMSE by 3.02\% in Hunan and 4.49\% in Hubei. In Hunan, the largest reduction occurred in the CPP thick-cloud subset, and daily irradiation MAE decreased over complete observation days. However, COT did not improve the stronger AGRI residual model or outperform matched retrieval features. Its benefit therefore depends on how the satellite information is represented. The corresponding forecasting experiment showed no resolved GHI improvement despite COT skill relative to persistence. The results support COT-assisted synchronous estimation in this setting and identify the retention of that benefit under forecast-input errors as an open problem.'''
en=span(en,r'\section{Discussion}',r'\FloatBarrier'+'\n'+r'\appendix',discussion_en+'\n\n'+conclusion_en+'\n')
zh=span(zh,'## 6 讨论','## 附录A','''## 6 讨论

反演COT改善了两个区域的紧凑GHI估计器，但与更丰富表示的比较表明，其价值取决于预测模型。规模相近的图像派生分支没有复现全部收益，而湖北的反演器特征与显式COT表现相近。更强的AGRI残差模型同样未从COT获得额外收益。这些结果表明，云产品监督可为紧凑估计器提供有用表示，但不能确立单一光学厚度输出优于学习特征中保留的信息。

湖南最大的改善出现在CPP厚云子组，加入COT减弱了GHI正偏差。这与更好地表达云衰减作用一致，但站点辐照度并非仅由云厚度决定。云相对于太阳光路的位置、散射辐射，以及15分钟平均区间内的变化都会影响这种关系。事件案例表明，明显的COT变化并不保证更准确的GHI曲线。此外，反演器读取16×16裁块，GHI图像分支读取3×3邻域，因此现有比较未将云属性监督与更大空间上下文的作用分离。

对太阳能资源评估而言，湖南日辐照量MAE下降具有积极意义，但日偏差更负，说明绝对误差改善可以与系统性低估并存。预报还引入另一层限制：预测COT优于持续性，加入它却没有改善匹配的GHI预报。这与云演变误差和输入分布变化带来的敏感性相符；现有比较无法将其唯一归因于外推模型、反演器或它们与GHI估计器的相互作用。

本研究限于两个独立训练区域的五个站点。CPP为时间覆盖稀疏的反演参考产品，连续云事件案例仅覆盖夏季。此外，模型选择和所报比较均使用验证数据，泛化能力仍需独立测试。这些限制明确了结论范围：COT在所测试的紧凑估计框架中有用，在更强预测器和预报中的增量价值仍未确定。

## 7 结论

在紧凑卫星GHI估计器中加入反演COT后，湖南和湖北验证RMSE分别降低3.02%和4.49%。湖南最大的降幅出现在CPP厚云子组，观测完整日期的日辐照量MAE也有所下降。然而，COT未改善更强的AGRI残差模型，也未优于匹配的反演特征，其收益因卫星信息的表示方式而异。在相应预报实验中，尽管COT优于持续性，GHI仍未显示明确改善。结果支持本研究设置下的COT辅助同期估计；如何在预测输入存在误差时保留这种收益，仍是有待解决的问题。

''')
en=en.replace(r'\FloatBarrier'+'\n'+r'\begin{thebibliography}',r'\section{Additional cloud-forecast comparisons}\label{sec:forecast-supp}'+'\n'+supp_en+'\n'+r'\FloatBarrier'+'\n'+r'\begin{thebibliography}')
zh=zh.replace('## 参考文献','## 附录B 云属性预报的补充比较\n\n'+supp_zh+'\n\n## 参考文献')

# Remove administrative vocabulary, retaining definitions and numerical claims.
en=en.replace('source-keyed common support','common valid pixels').replace('source-keyed CPP support','time- and location-matched CPP pixels')
en=en.replace('sampling contracts','sampling procedures').replace('label contract','label definition').replace('CPP-supervision contracts','CPP reference samples')
en=en.replace('The implemented predictor','The predictor').replace('S-to-R prediction bank','imagery and COT predictions').replace('completed GHI cohort','Hunan GHI dataset').replace('completed Hunan validation comparison','Hunan validation comparison')
en=en.replace('Hubei synchronous A/AC/AP and AL heads','Hubei synchronous heads').replace('Hubei forecast-input FA/FAC/FAP heads','Hubei forecast-input heads')
en=en.replace('Hubei same-time A/AC/AP primary','Hubei synchronous primary').replace('common-support AL sensitivity','retrieval-feature comparison on common samples')
en=en.replace('local feature branch from the frozen retriever\'s 16-channel decoder feature map (AL)','local features from the frozen retriever\'s 16-channel decoder feature map')
en=en.replace('for AL (COT minus AL','for AGRI + retrieval features (COT minus retrieval features').replace('COT and AL heads','COT and retrieval-feature heads')
en=en.replace('The error-blind event audit links','The event comparison links').replace('hash-selected case','reproducibly sampled case').replace('deterministic-hash event','reproducibly sampled event')
en=en.replace('using the audited station coordinates','using station coordinates').replace('the diagram does not imply that forecast skill has been established','both pathways produce 15-minute interval-mean GHI')
en=en.replace('The Hubei source field is interpreted under the original engineering W m$^{-2}$ operating convention; instrument unit and calibration metadata were not independently verified.','Hubei irradiance values are reported in W m$^{-2}$ as supplied; independent instrument calibration information was unavailable.')
en=en.replace('All intervals are conditional on validation-based checkpoint selection and are not independent test confirmation.','')
en=en.replace('Region-equal RMSE reduction','Within-region, station-equal RMSE reduction')
en=en.replace('The Hunan capacity control','The Hunan capacity control')
zh=zh.replace('验证队列','验证数据').replace('来源键匹配的共同有效支持','时空匹配的共同有效像元').replace('来源键匹配的CPP支持','时空匹配的CPP参考像元').replace('CPP监督合同','CPP参考样本')
zh=zh.replace('辐照度终点','辐照度评价指标').replace('配对GHI终点','配对GHI评价').replace('区域终点为','区域指标为').replace('已实现的预测器','预测器')
zh=zh.replace('湖北同期A/AC/AP及AL头','湖北同期模型').replace('湖北预报输入FA/FAC/FAP头','湖北预报模型').replace('湖北同期A/AC/AP主区间','湖北同期主要区间').replace('共同支持上的AL敏感性','共同样本上的反演特征对照')
zh=zh.replace('局地特征分支（AL）','局地特征分支').replace('AGRI + COT和AL','AGRI + COT和AGRI + retrieval features').replace('（COT−AL）','（COT减反演特征）')
zh=zh.replace('已完成的GHI样本包括','湖南GHI数据包括').replace('已完成的湖南验证比较','湖南验证比较').replace('冻结的全训练集S→R预测bank','全训练集模型生成的AGRI及COT预测')
zh=zh.replace('湖北原始字段按原工程W m⁻²操作口径解释，但仪器单位与校准元数据未独立核验。','湖北辐照度沿用供数记录的W m⁻²单位，未获得独立仪器校准资料。')
zh=zh.replace('误差盲选事件审计将','事件比较将').replace('哈希案例','可重复抽样案例').replace('确定性哈希事件','可重复抽样事件')
zh=zh.replace('经核验的站点坐标','站点坐标').replace('流程图不代表预测技能已得到证实','两条路径均输出15分钟区间平均GHI')
zh=zh.replace('上述区间均以验证集选模为条件，不是独立测试确认。','').replace('数量对应已完成的同期GHI样本；“待补”不表示零观测。','数量对应各区域的同期GHI样本。')
zh=zh.replace('区域等权RMSE降低量','各区域内站点等权RMSE降低量')
# Replace accidental raw LaTeX figure in the Markdown counterpart.
zh=re.sub(r'\\begin\{figure\}\[!htbp\]\\centering\s*\\includegraphics.*?\\end\{figure\}',lambda m:'![图7 COT预报误差与厚云检出率](F:/CODE_Classify/辐照度预报/solar-cot-ifs/paper/figures/fig7_forecast_cot_supervision.png)\n\n**图7 COT预报误差与厚云检出率。** (a)(b)四里和竺家局地3×3邻域的RMSE；(c)(d)厚云CSI（COT≥30）。各曲线采用相同CPP参考像元，比较原始SimVP、三种等预算微调和末帧持续性。总体RMSE按站点和时效等权平均，CSI由混淆计数汇总计算。COT无量纲。',zh,flags=re.S)
en_path.write_text(en,encoding='utf-8'); zh_path.write_text(zh,encoding='utf-8')
print('Bilingual revision saved; exploratory forecast comparisons retained in Appendix B.')
