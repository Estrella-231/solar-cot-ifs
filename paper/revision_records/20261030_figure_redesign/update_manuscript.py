"""Limited bilingual editorial update; result cohorts are not silently replaced."""
from pathlib import Path
import hashlib, json, shutil
HERE=Path(__file__).resolve().parent
PAPER=HERE.parents[1]
backup=HERE/'before'
backup.mkdir(exist_ok=True)
for n in ['main.tex','main_zh.md','main.pdf','FIGURE_PLAN.md']:
    if not (backup/n).exists(): shutil.copy2(PAPER/n,backup/n)

en=r'''Clouds redistribute solar radiation between direct and diffuse components, while solar geometry and surface reflection influence the relation between satellite radiance and station irradiance. Cloud optical thickness (COT) summarizes optical attenuation, but does not uniquely determine global horizontal irradiance (GHI). Its radiative effect depends on the scene and illumination conditions \cite{pap2024}. The relevant question is therefore not simply whether COT is correlated with irradiance, but whether it provides a useful representation beyond multispectral imagery and known geometry.

Existing approaches introduce cloud information in different ways. Image-based models learn the spectral--spatial mapping to irradiance directly; Chen et al. compared selected spectral inputs and attention-based representations under different sky conditions \cite{chen2023}. Explicit physical constraints can address ambiguities associated with surface reflection and thin clouds, as in Sasaki et al. \cite{sasaki2024}. Cloud-property approaches instead estimate or predict optical depth and connect it to radiation: Peng et al. combine optical-depth and cloud-mask forecasts with a radiative-transfer-based irradiance estimator \cite{peng2026}. These studies motivate a controlled comparison of representations rather than a claim that introducing COT is itself novel.

A COT field retrieved from the same AGRI image is not an independent observation. It may nevertheless help a finite-capacity estimator by organizing spectral information under cloud-product supervision. An apparent improvement could also arise from extra trainable capacity, features learned by the retriever, or its wider spatial context. In our setting, the retriever sees a $16\times16$ patch, whereas the GHI network receives aligned $3\times3$ fields. Comparing COT with a capacity-matched image branch and retrieval features can distinguish some of these explanations; retaining pixel order alone cannot establish that cloud structure causes the gain.

Forecasting places a separate requirement on this representation. Cloud-FNO evaluates six-hour forecasts of cloud physical properties, including COT \cite{zhang2026}; a recent preprint predicts COT from Meteosat imagery and evaluates its radiative effect through the cloud modification factor \cite{drosoglou2026}. These tasks differ from station-level interval-mean GHI forecasting. Improvements in cloud-property error need not be preserved after the irradiance mapping, and COT obtained from actual future imagery is unavailable operationally. A downstream evaluation must therefore distinguish forecast-derived COT from an observed-image reference used only for offline diagnosis.

We examine three questions: whether explicit COT reduces GHI error beyond imagery and matched representation controls; how the increment depends on cloud conditions; and which information is retained when predicted imagery replaces observations. Separate experiments in Hubei and Hunan assess regional repetition without pooling their observational records. Paired station--time comparisons quantify the increment, while aligned cloud fields and GHI trajectories examine its limits. The scientific focus is the conditional value of the representation, not a new cloud-extrapolation architecture.

'''
zh=r'''云会改变太阳辐射的直接与散射分量，而太阳几何及地表反射也会影响卫星辐亮度与站点辐照度之间的关系。云光学厚度（COT）概括光学衰减程度，但不能唯一确定全球水平辐照度（GHI），其辐射作用还取决于云场和照明条件[7]。因此，需要回答的不只是COT是否与辐照度相关，而是它能否在多光谱图像及已知几何之外提供有用的表示。

已有方法以不同方式引入云信息。图像模型直接学习光谱—空间信息到辐照度的映射；Chen等比较了所选光谱输入及注意力表示在不同天空条件下的表现[5]。明确的物理约束则可处理地表反射与薄云造成的混淆，例如Sasaki等的方法[6]。云属性方法先估计或预测光学厚度，再与辐射联系：Peng等将光学厚度、云掩膜预报与基于辐射传输的辐照度估计结合[8]。这些研究说明，本文需要受控比较不同表示，而不能把“引入COT”本身作为创新。

从同一AGRI图像反演的COT场不是独立观测，但云产品监督可能帮助有限容量的估计器组织光谱信息。表面收益也可能来自额外可训练容量、反演器学习的特征或更大的空间上下文。本文反演器读取16×16裁块，GHI网络接收对齐的3×3场。将COT与容量匹配的图像分支及反演特征比较，可以区分其中部分解释；仅保留像元顺序，并不足以证明收益由云空间结构产生。

预报对这种表示提出了另一项要求。Cloud-FNO评价包含COT在内的云物理属性六小时预报[10]；一项近期预印本由Meteosat图像预测COT，并通过云修正因子评价其辐射影响[9]。这些任务不同于站点区间平均GHI预报。云属性误差的改善不一定经过辐照度映射后仍然保留，真实未来图像反演的COT也不具备业务可用性。因此，下游评价必须区分预测图像反演COT与仅用于离线诊断的真实图像参考。

本文研究三个问题：显式COT能否在图像及匹配的表示对照之外降低GHI误差；这种增量如何依赖云况；预测图像替代观测后，哪些信息仍能保留。湖北与湖南分别开展实验，检验区域重复性，不合并两区域观测记录。通过站点—时刻配对比较量化增量，并用对齐的云场与GHI轨迹分析其边界。科学重点是这种表示的条件性价值，而非提出新的云图外推结构。

'''
tex=(PAPER/'main.tex').read_text(encoding='utf-8')
start=tex.index('\\section{Introduction}')+len('\\section{Introduction}')
end=tex.index('\\section{Study Regions and Data}')
tex=tex[:start]+'\n'+en+tex[end:]
tex=tex.replace('figures/fig2_cot_estimation_forecast_workflow.pdf','figures/redesign_20261030/method/method_workflow.pdf')
caption=r'''COT-assisted station irradiance estimation and forecast input availability. (a) Observed AGRI supplies both the local imagery branch and the frozen $16\times16$ COT retriever. The two $3\times3$ branches preserve pixel order until concatenation and flattening. AGRI channels and $\log(1+\mathrm{COT})$ use training-set normalization; the GHI geometry channels retain their cosine values. CPP provides retrieval supervision only. The predicted clear-sky index is scaled by interval-mean clear-sky GHI. (b) Eight observed frames ending at $t$ are used to forecast sixteen AGRI frames; their retrieved COT and predicted imagery enter separately trained forecast-input GHI heads. Target geometry is determined from time and location for SimVP, retrieval and GHI estimation. Each target is the end of a 15-minute averaging interval, covering $+15$ to $+240$ minutes. The grid panels are spatial-support schematics, not experimental cloud images.'''
marker='\\label{fig:framework}'
idx=tex.index(marker); cstart=tex.rfind('\\caption{',0,idx)
tex=tex[:cstart]+'\\caption{'+caption+'}\n'+tex[idx:]
(PAPER/'main.tex').write_text(tex,encoding='utf-8')
md=(PAPER/'main_zh.md').read_text(encoding='utf-8')
start=md.index('## 1 引言')+len('## 1 引言');end=md.index('## 2 研究区域与数据')
md=md[:start]+'\n\n'+zh+md[end:]
md=md.replace('paper/figures/fig2_cot_estimation_forecast_workflow.png','paper/figures/redesign_20261030/method/method_workflow.png')
start=md.index('**图2 ');end=md.index('### 3.1',start)
md=md[:start]+'''**图2 COT辅助站点辐照度估计与预报输入可用性。** (a) 真实AGRI同时进入局地图像分支和冻结的16×16 COT反演器。两条3×3分支在拼接与展平之前保留像元顺序。AGRI通道和$\\log(1+\\mathrm{COT})$使用训练集归一化；GHI几何通道保留余弦原值。CPP仅提供反演训练监督。预测晴空指数乘以区间平均晴空GHI，得到辐照度。(b) 以$t$结束的8帧观测用于预测16帧AGRI；预测图像及其反演COT输入另行训练的预报GHI头。SimVP、反演和GHI估计所用的目标几何由时间与位置确定。每个目标时刻是15分钟平均区间的结束时刻，覆盖+15至+240分钟。图中的网格为采样范围示意，并非实际云图。

'''+md[end:]
(PAPER/'main_zh.md').write_text(md,encoding='utf-8')
manifest={'scope':'Introduction and Fig.2 only; new result figures remain separately captioned until their cohorts and methods are integrated.',
 'changed_files':{n:hashlib.sha256((PAPER/n).read_bytes()).hexdigest() for n in ['main.tex','main_zh.md']},
 'unchanged_science':'No model, mask, selection, saved result, or training changed. Obsolete result sections are NOT certified by this editorial pass.'}
(HERE/'manuscript_edit_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Updated bilingual introduction and Fig.2; preserved pre-edit snapshots.')
