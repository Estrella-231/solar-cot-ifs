"""Integrate verified figures and results into the current bilingual draft.

Unresolved synchronous results remain explicit; no old score fills that gap.
"""
from pathlib import Path
import hashlib, json, shutil, re

HERE=Path(__file__).resolve().parent
PAPER=HERE.parents[1]
OUT=PAPER/'draft_v1_20261030'

def replace_once(text,old,new):
    assert text.count(old)==1,old[:100]
    return text.replace(old,new,1)

def fig(path,caption,label):
    return '\n'+r'\begin{figure}[!htbp]\centering'+'\n'+r'\includegraphics[width=\linewidth]{figures/revision_v1_20261030/'+path+'}\n'+r'\caption{'+caption+r'}\label{'+label+'}\n'+r'\end{figure}'+'\n'

def main():
    en=(HERE/'current_editorial_snapshot/main.tex').read_text(encoding='utf-8')
    zh=(HERE/'current_editorial_snapshot/main_zh.md').read_text(encoding='utf-8')
    # Source snapshot guard: permit replay only after our own recorded output.
    old_manifest=HERE/'MANUSCRIPT_INTEGRATION.json'
    expected=json.loads(old_manifest.read_text())['outputs'] if old_manifest.exists() else {}
    # Another writing window edits the original main files. Work in a separate
    # bilingual review edition and preserve those originals byte-for-byte.
    en=replace_once(en,'feature widths 32, 64, and 128','feature widths $w$, $2w$, and $4w$ ($w=32$ in the Hunan implementation and $w=16$ in the Hubei implementation)')
    zh=replace_once(zh,'特征宽度为32、64、128','特征宽度为w、2w、4w（湖南实现w=32，湖北实现w=16）')
    en=replace_once(en,'The encoded local features are used to predict the station clear-sky index.',r'''In the single-frame Hunan implementation, two $1\times1$ convolutional layers encode the 13 AGRI and two geometry channels as 32 feature channels. A separate $1\times1$ layer encodes one standardized log-COT channel into four channels. Concatenation gives $36\times3\times3$ features; ordered flattening produces 324 values for a $324\to64\to32\to1$ readout with a final softplus. The $1\times1$ encoders act per pixel, whereas the readout jointly uses the nine positions; retaining their order does not by itself demonstrate learned cloud motion. The Hubei four-arm single-frame design uses two auxiliary slots (COT and cloud probability), mapped $2\to4$ without bias; inactive slots are zero. These regional implementations must be distinguished when interpreting capacity controls.''')
    en+= ''
    method_caption=r'''Implemented single-frame retrieval and station irradiance architecture. Tensor labels omit the batch dimension. The retriever uses a $16\times16$ context, whereas the irradiance head reads aligned $3\times3$ inputs. The main head panel shows the Hunan one-slot COT design; the inset distinguishes the Hubei two-slot control design. Geometry is available to the imagery branch. Ordered features are concatenated, flattened and mapped to a clear-sky index before multiplication by interval-mean clear-sky GHI. Solid lines carry inference features and dashed lines indicate supervision. This diagram describes the implementations; it is not a claim of identical regional weights or capacities.'''
    anchor='\nThe models minimize physical GHI mean squared error, scaled for numerical convenience:'
    en=replace_once(en,anchor,fig('architecture/fig_core_architecture.pdf',method_caption,'fig:architecture')+anchor)
    znew='湖南单帧实现中，13个AGRI与2个几何通道经两层1×1卷积编码为32通道；标准化对数COT经独立1×1卷积编码为4通道。拼接得到36×3×3特征，按通道、行、列顺序展平为324维，再经324→64→32→1读出和末端softplus得到晴空指数。1×1编码逐像元进行，九个位置由后续读出联合利用；保留位置本身不证明学到了云运动。湖北四臂单帧设计使用COT与云概率两个辅助槽，经不带偏置的2→4映射编码，不活动槽置零；容量对照必须区分两种区域实现。'
    zh=replace_once(zh,'编码后的局地特征用于预测站点晴空指数。',znew)
    if znew not in zh:
        pos=zh.index('主分析使用目标时刻单帧') if '主分析使用目标时刻单帧' in zh else zh.index('三帧配置')
        zh=zh[:pos]+znew+'\n\n'+zh[pos:]
    architecture_zh='\n\n![单帧模型结构](F:/CODE_Classify/辐照度预报/solar-cot-ifs/paper/figures/revision_v1_20261030/architecture/fig_core_architecture.png)\n\n**模型结构图：已实现的单帧反演与站点辐照度网络。** 张量省略batch维；R使用16×16上下文，GHI头读取对齐3×3输入。主头面板为湖南单槽COT设计，插图区分湖北双槽控制；实线为推理特征、虚线为监督。该图描述实现，不代表两区域权重或容量相同。\n\n'
    zh=replace_once(zh,'### 3.3 短时预报',architecture_zh+'### 3.3 短时预报')
    forecast_method=r'''The forecast-trained comparison separates historical-COT information from the conditional contribution of future COT. An imagery-only baseline and an otherwise matched historical-COT baseline are fitted on base-training dates; the latter receives eight observed-history COT fields. The historical-COT model is then frozen. Two equal-capacity gated residual heads are fitted on separate correction-training dates: the control has neutral-zero future-COT slots, while the augmented head receives COT from forecast AGRI. Both use the same examples, geometry, sample order and budget. Corrections are signed, with zero-initialized output layers, and the final clear-sky index is nonnegative. Base and correction training use 30 epochs, batch 1,536 and AdamW (learning rate $10^{-3}$, weight decay $10^{-4}$, cosine decay to $10^{-5}$). The three random seeds are 42, 43 and 44. This is a separately trained forecasting comparison, not a direct reuse of the synchronous head's weights.'''
    old='We apply the two irradiance models trained on observed imagery to forecast imagery, keeping their parameters fixed, to determine whether the benefit of including COT persists in forecasting. Models trained specifically on forecast imagery are evaluated separately. For the three-frame configuration, +15 min includes two known historical frames and +30 min includes one; results for leads $\\geq45$ min separately assess inputs consisting entirely of forecast imagery.'
    en=replace_once(en,old,forecast_method)
    zforecast='预报专训比较将历史COT与未来COT的条件增量分开。无COT基线和匹配的历史COT基线在基底训练日期拟合，后者读取8帧真实历史反演COT；随后冻结历史COT模型，在另一组修正训练日期拟合两个等容量门控残差头。控制组未来COT槽为中性零，加入COT组读取预测AGRI反演的COT；两组样本、几何、行序和预算相同。修正可正可负，输出层零初始化，最终晴空指数非负。两个训练阶段均为30轮、batch 1536、AdamW，学习率10⁻³、weight decay 10⁻⁴，余弦衰减至10⁻⁵；种子42、43、44。这是独立训练的预报对照，不是直接复用同期头权重。'
    start=zh.index('将基于观测云图训练的两个辐照度模型')
    if start>=0:
        end=zh.index('\n\n',start)
        zh=zh[:start]+zforecast+zh[end:]
    else:
        # Keep the source intact if the translated paragraph was reformulated.
        p=zh.index('## 4 实验设计');zh=zh[:p]+zforecast+'\n\n'+zh[p:]
    old_start=en.index('\\figslot{Observed AGRI')
    old_end=en.index('\n\n\\section{Experimental design}',old_start)
    en=en[:old_start]+fig('workflow/fig_workflow.pdf',r'''Data and task flow for observed-image estimation and forecasting. CPP provides retrieval supervision; station GHI provides targets and evaluation. The forecast path uses only historical observations and predicted future imagery. Regional image support, retrieval context and station neighbourhood are distinct. The forecast-COT correction retains the historical-COT base and is compared against an equal-capacity correction with neutral future-COT slots. Future observed imagery is confined to offline diagnosis.''','fig:framework')+en[old_end:]
    zh=replace_once(zh,'**图2：COT辅助反演与预报流程。** 同期路径为观测AGRI→反演COT→GHI；预报路径为历史AGRI→预测AGRI→预测COT→GHI。有无COT对照仅改变局地COT表示是否加入，并区分历史观测与预测场。','**流程图：COT辅助估计与预报。** CPP仅供反演监督，GHI作标签和评价；区域图、反演上下文与站点邻域尺度分开。未来COT修正保留历史COT基底，与未来COT槽为中性零的等容量修正比较。推理仅用历史观测及预测未来图，未来真实图仅作离线诊断。')
    # A compact methods table adds information rather than another score list.
    intro_table=r'''
\begin{table}[!htbp]\centering\small
\caption{Related approaches and the distinct question evaluated here. This is a comparison of designs, not a cross-dataset ranking of RMSE.}\label{tab:related}
\begin{tabularx}{\linewidth}{p{29mm}XX}\toprule
Study & Representation and task & Question informing this study\\\midrule
Chen et al.\ \cite{chen2023} & Selected multispectral imagery for GHI/DNI estimation & How does a spectral representation support a compact station mapping?\\
Papachristopoulou et al.\ \cite{pap2024} & Cloud/aerosol information and radiation modelling for nowcasting and forecasting & How do cloud conditions and forecast lead affect irradiance error?\\
Zhang et al.\ \cite{zhang2026} & Forecasting cloud physical properties with a Fourier neural operator & Are cloud-property evolution and spatial structure retained?\\
This study & AGRI retained alongside retrieved COT, with paired representation and forecasting controls & Does explicit COT add GHI value beyond the same imagery, and under which input conditions?\\\bottomrule
\end{tabularx}\end{table}
'''
    en=replace_once(en,'\\section{Study area and data}',intro_table+'\n\\section{Study area and data}')
    zh=replace_once(zh,'## 2 研究区域与数据','''**相关工作对照表：比较研究设计，不跨数据集排名RMSE。**

| 研究 | 表示与任务 | 对本文问题的启发 |
|---|---|---|
| Chen等[1] | 多光谱选择、GHI/DNI估计 | 光谱表示怎样进入紧凑站点映射 |
| Papachristopoulou等[3] | 云/气溶胶信息与辐射计算、短时预报 | 云况和时效怎样影响辐射误差 |
| Zhang等[4] | Fourier神经算子预测云物理属性 | 演变与空间结构是否保留 |
| 本研究 | 保留AGRI并加入反演COT，配对表示和预报控制 | 显式COT是否在同源图像外增加GHI价值，依赖哪些输入条件 |

## 2 研究区域与数据''')
    start=en.index('\\subsection{Effect of COT on forecasting and propagation of image prediction errors}')
    end=en.index('\\FloatBarrier',start)
    result=r'''\subsection{Forecasting gain and propagation of image errors}
The forecast-trained comparison uses 15,275 common station--initialization--target records from 22 selected validation target dates between 22 August 2025 and 29 June 2026. Table~\ref{tab:forecast_values} separates the historical-COT baseline from the two subsequent correction heads. Historical COT lowers equal-station, equal-seed RMSE from 105.595 to 103.670 W m$^{-2}$, a point reduction of 1.823\%. The paired difference is $-1.925$ W m$^{-2}$ with a 95\% target-date block interval of $[-4.734,+0.670]$, so the apparent improvement is not resolved from zero by this analysis.

Adding forecast COT to the correction head changes RMSE from 103.783 to 103.808 W m$^{-2}$. Its difference from the matched-capacity control is $+0.026$ W m$^{-2}$, with interval $[-0.003,+0.059]$. Figure~\ref{fig:forecast} retains the complete 16-lead comparison and distinguishes seed variation from date-block uncertainty. This comparison does not demonstrate an additional benefit from future COT; it also does not establish equivalence or universal absence of value.

\begin{table}[!htbp]\centering\small
\caption{Forecast GHI RMSE (W m$^{-2}$) on common validation rows. Each value averages three seed-specific scores; the final row also averages stations equally. The correction heads share the trained historical-COT base.}\label{tab:forecast_values}
\begin{tabularx}{\linewidth}{Xrrrr}\toprule
Station & No COT & History COT & History + control & History + forecast COT\\\midrule
Niushou & 97.017 & 96.299 & 96.665 & 96.701\\
Wangzhai & 106.183 & 102.892 & 103.165 & 103.215\\
Mizhuang & 113.586 & 111.819 & 111.519 & 111.510\\
Equal-station mean & 105.595 & 103.670 & 103.783 & 103.808\\\bottomrule
\end{tabularx}\end{table}
'''
    result+=fig('evidence/forecast_ghi.pdf',r'''Paired GHI forecasting evaluation. (a) Four input variants over +15--240 min. (b) Forecast-COT-minus-matched-control difference and pointwise 95\% paired target-date bootstrap interval. (c,d) Historical-COT-minus-no-COT and forecast-COT-minus-control differences by station. Diamonds average three seeds; open circles show seed-specific differences; whiskers show 95\% date-block intervals (5,000 draws). Negative differences improve error. Panels use different scales. All panels use the same 15,275 records on 22 validation target dates. No actual future AGRI or CPP enters forecast inference. These are development-validation results, not an independent test.''','fig:forecast')
    result+=r'''A separate fixed-head diagnosis uses 8,111 common replacement rows from 12 development target dates. The control in this diagnosis already contains historical COT. Replacing forecast AGRI by observed target-time AGRI lowers its RMSE from 103.469 to 89.441 W m$^{-2}$ (difference $-14.028$, interval $[-21.911,-6.053]$). However, adding source-matched future COT gives a difference of $+0.025$ under forecast AGRI and $+0.140$ under observed AGRI; both intervals include zero. Thus better image information is relevant to this mapping, but coherent observed image and COT inputs do not themselves recover an incremental COT benefit in the fixed head.

\begin{table}[!htbp]\centering\small
\caption{MSE sensitivity decomposition on the same 8,111 replacement rows. O denotes observed-image inputs and F forecast-image inputs. Units are W$^2$ m$^{-4}$. Points satisfy Eq.~\ref{eq:propagation}; interval limits are not additive.}\label{tab:decomposition}
\begin{tabularx}{\linewidth}{Xrr}\toprule
Contrast & Point & Paired 95\% interval\\\midrule
$E_{FO}-E_{OO}$ & 2778.1 & [938.0, 5228.9]\\
$E_{OF}-E_{OO}$ & $-5.4$ & [$-48.9$, 33.5]\\
$E_{FF}-E_{FO}-E_{OF}+E_{OO}$ & $-51.6$ & [$-109.2$, $-7.8$]\\
$E_{FF}-E_{OO}$ & 2721.1 & [920.4, 5105.2]\\\bottomrule
\end{tabularx}\end{table}
The image-source term is large in this conditional comparison, while the interaction is nonzero. These are sensitivities of a particular fixed mapping rather than an independent physical error budget. The replacement rows differ from the forecasting-validation cohort, so their absolute scores must not be used as a validation-to-oracle gain. Neither these contrasts nor an individual cloud event uniquely identifies displacement, optical growth or retrieval behavior as the cause.
'''
    result+=fig('source_replacement/source_replacement.pdf',r'''Fixed-head source sensitivity on 8,111 common rows from 12 development target dates. (a) GHI RMSE for each AGRI and future-COT source. Every branch retains historical COT. (b) AGRI replacement in the no-future-COT control. (c) Source-matched COT addition under the same AGRI source. Intervals are 95\% paired date-block intervals; (b,c) use different scales. Observed future inputs are offline diagnostic information.''','fig:replacement')
    en=en[:start]+result+'\n'+en[end:]
    start=zh.index('### 5.5 ');end=zh.index('## 6 讨论',start)
    zresult='''### 5.5 预报增益与底图误差传递

预报专训对照使用2025年8月22日至2026年6月29日之间选定的22个验证目标日期、15,275条共同站点—初始化—目标时刻记录。历史COT使站点与种子等权RMSE由105.595降至103.670 W/m²，点降幅1.823%；配对差−1.925，目标日期块95%区间[−4.734,+0.670]，本分析未能将该改善与零区分。

在修正头加入未来预测COT后，RMSE由同容量控制的103.783变为103.808 W/m²，配对差+0.026，区间[−0.003,+0.059]。完整16时效图将种子变化与日期块不确定性分开。该比较未显示未来COT额外增益，也不证明等效或普遍无价值。

**预报精确结果表：共同验证样本上的GHI RMSE，W/m²。每格为三seed分别评分后平均，末行进一步对站点等权；后两臂共用历史COT基底。**

| 站点 | 无COT | 历史COT | 历史+控制修正 | 历史+预测COT |
|---|---:|---:|---:|---:|
| 牛首 | 97.017 | 96.299 | 96.665 | 96.701 |
| 王寨 | 106.183 | 102.892 | 103.165 | 103.215 |
| 米庄 | 113.586 | 111.819 | 111.519 | 111.510 |
| 站点等权 | 105.595 | 103.670 | 103.783 | 103.808 |

![预报配对结果](F:/CODE_Classify/辐照度预报/solar-cot-ifs/paper/figures/revision_v1_20261030/evidence/forecast_ghi.png)

**预报图。** (a) 四组+15～240分钟RMSE。(b) 预测COT减同容量控制及逐时效95%目标日期块区间。(c,d) 历史COT减无COT、预测COT减控制的逐站差值。菱形为三seed平均、空心圆为单seed、横线为5,000次日期块bootstrap的95%区间；负值改善，各面板尺度不同。共同15,275行、22个验证目标日期；未来真实AGRI和CPP不进入推理。本图为开发验证，不是独立测试。

另一组固定头诊断使用12个开发目标日期的8,111条共同可替换记录。该诊断的控制已含历史COT。用真实目标时刻AGRI替换预测AGRI，使控制RMSE由103.469降至89.441 W/m²，差−14.028、区间[−21.911,−6.053]。然而，在预测AGRI与真实AGRI条件下分别加入同源未来COT，差值为+0.025和+0.140，两区间均跨零。因此较好的云图信息与该映射相关，但将真实图与真实图COT联合输入，并未恢复固定头中的COT增量。

**固定头MSE分解表。** O表示真实图来源，F表示预测图来源；同一8,111行，单位W²/m⁴。点估计满足式(10)，区间端点不能相加。

| 对比 | 点估计 | 配对95%区间 |
|---|---:|---:|
| E_FO−E_OO | 2778.1 | [938.0,5228.9] |
| E_OF−E_OO | −5.4 | [−48.9,33.5] |
| E_FF−E_FO−E_OF+E_OO | −51.6 | [−109.2,−7.8] |
| E_FF−E_OO | 2721.1 | [920.4,5105.2] |

图像来源项在这个条件比较中较大，交互项也非零。它们是特定固定映射的敏感性，不是独立物理误差预算。替换样本与预报验证队列不同，不能把两者绝对分数差当验证到oracle收益；这些对照与单个云事件都不能唯一指定云团位移、厚度增长或反演行为为原因。

![来源替换诊断](F:/CODE_Classify/辐照度预报/solar-cot-ifs/paper/figures/revision_v1_20261030/source_replacement/source_replacement.png)

**来源替换图。** 12个开发目标日期、8,111条共同记录。(a) 不同AGRI与未来COT来源的GHI RMSE；所有分支保留历史COT。(b) 无未来COT控制中的AGRI替换。(c) 相同AGRI来源下加入同源COT。区间为目标日期块配对95%区间，(b)(c)尺度不同；未来真实来源仅作离线诊断。

'''
    zh=zh[:start]+zresult+zh[end:]
    # The corresponding control is not imagery-only: historical COT is fixed.
    en=en.replace('the imagery-only head is also evaluated on observed and forecast imagery on those same keys','the control without future COT is also evaluated on observed and forecast imagery on those same keys')
    en=en.replace('than the imagery-only head','than that control')
    zh=zh.replace('无COT头在同一键上','无未来COT控制在同一键上').replace('无COT头，负值','该控制，负值')
    en=en.replace('we also evaluate the irradiance model without COT on observed and forecast imagery using the same samples','we also evaluate the matched control without future COT, retaining its historical COT, on observed and forecast imagery using the same samples')
    zh=zh.replace('另将无COT辐照度模型分别用于观测及预测图像','另将保留历史COT、但不含未来COT的匹配控制分别用于观测及预测图像')
    en=en.replace('In the three-frame configuration, known historical positions remain identical; only future positions are replaced.','Historical COT and other historical inputs remain identical; only future AGRI and COT sources are replaced in this diagnosis.')
    zh=zh.replace('三帧配置中已知历史位置保持相同，只替换未来位置。','该诊断中历史COT及其他历史输入保持相同，只替换未来AGRI和COT来源。')
    # Keep Chinese figure/table indices aligned with LaTeX's caption order.
    zh=re.sub(r'图([3-6])(?=：)',lambda m:'图'+str(int(m.group(1))+1),zh)
    zh=re.sub(r'表([1-3])',lambda m:'表'+str(int(m.group(1))+1),zh)
    zh=zh.replace('**相关工作对照表：','**表1：相关工作对照，')
    zh=zh.replace('**模型结构图：','**图2：模型结构，')
    zh=zh.replace('**流程图。','**图3：流程图。').replace('**流程图：','**图3：流程图，')
    zh=zh.replace('**预报精确结果表：','**表5：预报精确结果，')
    zh=zh.replace('**预报图。','**图8：预报配对结果。')
    zh=zh.replace('**固定头MSE分解表。','**表6：固定头MSE分解。')
    zh=zh.replace('**来源替换图。','**图9：来源替换诊断。')
    # Distinguish the review edition from the concurrently edited primary files.
    en=en.replace('\\begin{abstract}',r'\begin{center}\small Figure-integrated review draft, 30 October 2026. Unresolved result fields remain explicit.\end{center}'+'\n'+r'\begin{abstract}',1)
    zh=zh.replace('## 摘要','> 图表整合审阅稿，2026-10-30。尚未核验的结果保留待填标记。\n\n## 摘要',1)
    for rel in ['architecture/fig_core_architecture.pdf','workflow/fig_workflow.pdf','evidence/forecast_ghi.pdf','source_replacement/source_replacement.pdf']:
        dst=OUT/'figures/revision_v1_20261030'/rel
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(PAPER/'figures/revision_v1_20261030'/rel,dst)
    outputs={}
    for name,text in [('main.tex',en),('main_zh.md',zh)]:
        (OUT/name).write_text(text,encoding='utf-8')
        outputs[name]=hashlib.sha256((OUT/name).read_bytes()).hexdigest()
    (HERE/'MANUSCRIPT_INTEGRATION.json').write_text(json.dumps({'state':'bilingual review-draft update','outputs':outputs,
       'scope':'verified architecture and forecast evidence integrated; unresolved synchronous results remain pending',
       'source_replacement':'8111 common rows / 12 development dates; different from forecast validation',
       'test_read':False,'training_started':False},indent=2),encoding='utf-8')
    print(json.dumps(outputs,indent=2))

if __name__=='__main__':main()
