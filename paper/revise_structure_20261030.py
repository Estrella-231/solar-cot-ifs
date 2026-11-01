"""Write a bilingual, results-pending scientific draft; preserve the preceding draft."""
from pathlib import Path
import re
import shutil

root = Path(__file__).resolve().parent
archive = root / 'revision_records' / '20261030_structure_draft' / 'before'
archive.mkdir(parents=True, exist_ok=True)
for name in ('main.tex', 'main_zh.md'):
    dest = archive / name
    if not dest.exists():
        shutil.copy2(root / name, dest)
old = (archive / 'main.tex').read_text(encoding='utf-8')
entries = dict(re.findall(r'\\bibitem\{([^}]+)\}\s*(.*?)(?=\\bibitem|\\end\{thebibliography\})', old, re.S))
entries['liao2024'] = ('Z. Liao and C. F. M. Coimbra.\n'
    'Hybrid solar irradiance nowcasting and forecasting with the SCOPE method and convolutional neural networks.\n'
    r'\textit{Renewable Energy}, 232:121055, 2024.' '\n'
    r'\url{https://doi.org/10.1016/j.renene.2024.121055}.' '\n')

tex = r'''\documentclass[11pt]{article}
\usepackage[margin=25mm]{geometry}
\usepackage{amsmath,amssymb,graphicx,booktabs,tabularx,placeins,url}
\setlength{\emergencystretch}{2em}
\newcommand{\pending}[1]{\textit{[To be completed: #1]}}
\newcommand{\figslot}[3]{%
\begin{figure}[!htbp]\centering
\fbox{\begin{minipage}[c][32mm][c]{0.92\linewidth}\centering #1\end{minipage}}
\caption{#2}\label{#3}\end{figure}}
\title{Cloud Optical Thickness for Satellite-Based Solar Irradiance Estimation and Short-Term Forecasting}
\author{}\date{}
\begin{document}
\begin{center}\LARGE\bfseries
Cloud Optical Thickness for Satellite-Based Solar Irradiance Estimation and Short-Term Forecasting
\end{center}
\begin{abstract}
Multispectral satellite imagery contains information about clouds, but learning its relation to surface solar irradiance does not necessarily make explicit use of cloud attenuation. This study examines whether cloud optical thickness (COT), retrieved from the same imagery under cloud-product supervision, improves global horizontal irradiance (GHI) estimation and remains useful in short-term forecasting. A frozen retriever supplies spatially aligned local COT fields to a compact irradiance model alongside AGRI channels and solar geometry. Paired experiments use Hubei as the primary study region and Hunan as a complementary region. We examine overall estimation accuracy, dependence on cloud optical thickness, and continuous cloud events. The framework is then applied to forecast imagery over 15--240 min. Input-replacement experiments trace how image-extrapolation errors propagate through the COT and irradiance mappings. \pending{Insert final regional estimation gains, the principal cloud-condition finding, and the matched forecasting gain.} \pending{Insert the error-propagation finding and the conclusion warranted by these comparisons.} The study connects the practical value of an explicit cloud-property representation with the quality of the evolving cloud fields available to irradiance forecasting.
\end{abstract}

\section{Introduction}
Clouds modulate the solar radiation reaching the surface, and changes in their location and optical thickness can produce substantial variability at a ground station. Geostationary multispectral imagery offers frequent observations of these changes. Data-driven irradiance models exploit spectral and spatial patterns, yet a spectral description and an explicit cloud-property representation need not be equally useful to a finite-capacity estimator. The question addressed here is whether providing retrieved COT in addition to the original imagery improves station GHI estimation.

Previous studies have improved satellite irradiance estimation through spectral selection and spatial learning \cite{chen2023}. Cloud-related physical features can also supplement image-based models: Liao and Coimbra compared CNNs with and without SCOPE-derived information and extended the approach to short-term forecasting \cite{liao2024}. These studies motivate examining the incremental value of an explicit optical-thickness field when the original multispectral observations are already retained. Because COT is inferred from those observations, the additional information takes the form of a representation learned with cloud-property supervision rather than an independent measurement.

The usefulness of that representation may vary with cloud conditions. Errors in cloud attenuation can affect irradiance magnitude, while cloud boundaries, solar geometry, and evolution within the averaging interval influence the timing and amplitude of station-level fluctuations. An overall score alone cannot show whether an improvement is broadly distributed or concentrated in particular optical-thickness regimes. Matched comparisons across cloud conditions and continuous events are therefore needed alongside the aggregate evaluation.

Short-term forecasting introduces an additional challenge: the imagery supplied to the retrieval and irradiance models is itself predicted. Cloud-motion-based irradiance studies have evaluated this distinction between estimation errors and forecast errors \cite{pap2024}; Cloud-FNO further demonstrates the prediction of evolving cloud physical properties \cite{zhang2026}. In an image-extrapolation chain, errors in cloud position, boundaries, or growth and decay can reach GHI both through the image input and through retrieved COT. Understanding these pathways is necessary to explain how the value of COT changes from estimation to forecasting.

We address three questions: (1) how much does local COT improve GHI estimation beyond AGRI imagery and solar geometry; (2) under which optical-thickness conditions and cloud events does the improvement occur; and (3) how much of that benefit remains in 15--240 min forecasts, and how is it constrained by cloud-image extrapolation errors? Hubei provides the primary station comparisons, while independently trained Hunan models provide complementary regional evidence. The work focuses on the contribution of cloud-property representation and its error propagation; developing an improved image-extrapolation model is a subsequent research question.

\section{Study area and data}
\subsection{Regions, satellite observations, and cloud reference}
The primary region contains Niushou, Wangzhai, and Mizhuang in Hubei. Sili and Zhujia in Hunan form the complementary region. Models and normalization are fitted independently within each region. Regional results are reported separately because station observations have different native temporal sampling.

The AGRI inputs comprise thirteen channels, C01--C06 and C09--C15, with region-specific geolocation. The same station indices register the imagery, geometry, and COT fields. The COT retriever reads a $16\times16$ patch; the irradiance model uses its central $3\times3$ neighbourhood. CPP provides COT and cloud-phase information for retrieval supervision and cloud-condition analysis. It is a satellite retrieval reference, with uncertainty and incomplete temporal coverage. Its availability does not determine inclusion in the main GHI evaluation.

\pending{Table 1: final station coordinates, observation periods, native sampling, and numbers of dates and samples in each partition.}
\begin{table}[!htbp]\centering\small
\caption{Regional observations and temporal partitions. Coordinates, periods, native sampling, and partition counts will be filled from the final data summary.}\label{tab:data}
\begin{tabularx}{\linewidth}{lXX}\toprule
Region & Stations & Observation and partition details\\\midrule
Hubei & Niushou; Wangzhai; Mizhuang & \pending{coordinates, periods, sampling, counts}\\
Hunan & Sili; Zhujia & \pending{coordinates, periods, sampling, counts}\\\bottomrule
\end{tabularx}\end{table}
\figslot{Station map and observation coverage}{Study regions and temporal coverage of AGRI, station GHI, and CPP reference observations. Coverage is shown separately from the span between the first and last records.}{fig:regions}

\subsection{Irradiance targets and solar geometry}
Let $\tau$ denote the end of a 15-minute target interval. The target is the interval-mean GHI, $\bar G_{s,\tau}$, at station $s$. Hunan five-minute interval-end records are aggregated as
\begin{equation}
\bar G_{s,\tau}=\frac13\sum_{j=0}^{2}G^{(5)}_{s,\tau-5j\,\mathrm{min}}.
\label{eq:interval}
\end{equation}
Hubei labels are formed from the valid one-minute bins in the same right-closed interval; incomplete windows are excluded. The clear-sky irradiance $\bar G^{cs}_{s,\tau}$ is averaged over that interval using the Ineichen--Perez model \cite{ineichen,pvlibclear}. The learning target and reconstruction are
\begin{equation}
k_{s,\tau}=\frac{\bar G_{s,\tau}}{\bar G^{cs}_{s,\tau}},\qquad
\widehat G_{s,\tau}=\widehat k_{s,\tau}\bar G^{cs}_{s,\tau}.
\label{eq:kt}
\end{equation}
The clear-sky index is not capped at one. GHI observations serve as labels, not model inputs. Solar geometry is computed from UTC, latitude, and longitude and registered with each image. The cosine of solar zenith and the cosine of the solar--viewing relative azimuth provide the geometric inputs; the retriever also receives a daytime indicator.

\subsection{Temporal sampling}
Training, validation, and final evaluation dates are separated at the date-block level. For Hunan, sufficiently covered months reserve separate two-day blocks for validation and testing, with one-day buffers; months with fewer than twenty eligible days contribute only training data. Each forecasting example contains eight historical and sixteen future frames at 15-minute intervals, all within the same partition. Normalization is fitted on training data alone. \pending{Complete Table 1 with the fixed Hubei partition and the final evaluated cohorts.}

\section{Methods}
\subsection{Supervised COT retrieval}
For imagery $A_\tau$ and geometry $g_\tau$, the retriever produces a nonnegative log-COT field,
\begin{equation}
L_\tau=R_\theta(A_\tau,g_\tau),\qquad C_\tau=\exp(L_\tau)-1.
\label{eq:retrieval}
\end{equation}
The retriever is a U-Net with two downsampling stages, feature widths 32, 64, and 128, and skip-connected upsampling to the original $16\times16$ grid. Convolutional blocks use batch normalization and SiLU; the final softplus produces nonnegative $L_\tau$. It is trained against $\log(1+C^{CPP})$ with a masked Huber objective,
\begin{equation}
\mathcal L_R=\frac{\sum_{i,p}m_{i,p}\rho_{0.1}\!\left(L_{i,p}-\log(1+C^{CPP}_{i,p})\right)}{\sum_{i,p}m_{i,p}},
\label{eq:retrieval-loss}
\end{equation}
where $m$ identifies valid common input and reference pixels. Valid zero-COT references are retained. Training normalization is retained at inference; inference validity uses available inputs and geometry, without requiring a future CPP reference. Reference-quality criteria apply to supervised training and reference-based evaluation. The fitted retriever is frozen during GHI training and forecasting.

\subsection{Local COT augmentation of the irradiance model}
Let $\mathcal P_s$ select the station-centred $3\times3$ field. AGRI channels and log-COT are standardized with training statistics:
\begin{equation}
X_{s,\tau,c}=\frac{\mathcal P_sA_{\tau,c}-\mu_c}{\sigma_c},\qquad
Z_{s,\tau}=\frac{\mathcal P_sL_\tau-\mu_L}{\sigma_L}.
\label{eq:standardization}
\end{equation}
COT retains its nine-pixel arrangement and between-sample magnitude; no patch mean or percentile replaces the field. The paired mappings are
\begin{equation}
\widehat k^A_{s,\tau}=H_A(X_{s,\tau},g_{s,\tau}),\qquad
\widehat k^{AC}_{s,\tau}=H_{AC}(X_{s,\tau},g_{s,\tau},Z_{s,\tau}).
\label{eq:heads}
\end{equation}
Local features are encoded before a station readout predicts the clear-sky index. The main estimation comparison uses one target-time frame. A three-frame configuration additionally uses $\tau-30$, $\tau-15$, and $\tau$ minutes and is treated separately. Input frame count is held fixed within each COT comparison.

The models minimize physical GHI mean squared error, scaled for numerical convenience:
\begin{equation}
\mathcal L_H=\frac1N\sum_i\left[\frac{\widehat k_i\bar G^{cs}_i-\bar G_i}{1000\ \mathrm{W\,m^{-2}}}\right]^2.
\label{eq:ghi-loss}
\end{equation}
Paired training uses the same examples, random seeds, sample order, optimization budget, and selection rule. Table~\ref{tab:training} specifies the final architecture and settings for each evaluated configuration.
\begin{table}[!htbp]\centering\small
\caption{Model configurations and training settings, reported separately for the main single-frame comparison and the temporal extension.}\label{tab:training}
\begin{tabularx}{\linewidth}{lXX}\toprule
Component & Fixed input definition & Settings to report\\\midrule
COT retriever & 13 AGRI + 3 geometry/daytime channels; $16\times16$ & \pending{normalization, optimizer, batch, epochs, selection}\\
GHI estimator & $3\times3$ AGRI + geometry, with/without COT & \pending{widths, parameters, frame count, batch, updates}\\
Image extrapolator & 8 historical frames; 16 future frames & \pending{regional model, normalization, checkpoint selection}\\\bottomrule
\end{tabularx}\end{table}

\subsection{Short-term forecasting}
With initialization $t$ defined by the last historical image, a fixed image predictor $S_\psi$ generates
\begin{equation}
\widehat A_{t+k\Delta}=S_\psi(A_{t-7\Delta:t},g)_{k},\qquad
\widehat C_{t+k\Delta}=\exp R_\theta(\widehat A_{t+k\Delta},g_{t+k\Delta})-1,
\label{eq:forecast}
\end{equation}
for $k=1,\ldots,16$ and $\Delta=15$ min. SimVP provides the fixed baseline \cite{simvp,simvpv2}. Additional image predictors, where evaluated, use the same downstream comparison. Future solar geometry is deterministic. Future observed imagery and CPP are excluded from forecasting inputs.

The same pair of frozen irradiance mappings is applied to imagery with and without forecast-derived COT to measure retention of the estimation representation in forecasting. Heads trained specifically on forecast imagery form a separate comparison. For a three-frame head, +15 min includes two known historical frames and +30 min includes one; results for leads $\geq45$ min separately assess inputs consisting entirely of forecast imagery.
\figslot{Observed AGRI $\rightarrow$ retrieved COT $\rightarrow$ GHI\\[4pt]
Historical AGRI $\rightarrow$ forecast AGRI $\rightarrow$ forecast COT $\rightarrow$ GHI}{COT-assisted estimation and forecasting. The paired GHI inputs differ only in the inclusion of a local COT representation. The forecasting branch preserves the distinction between historical observations and predicted future fields.}{fig:framework}

\section{Experimental design}
\subsection{Paired evaluation of COT value}
Each comparison uses identical station, initialization where applicable, target time, GHI label, and valid daytime mask. Results are computed for each station and random seed before equal-weight averaging within region. We report physical RMSE, MAE, bias, and the relative RMSE reduction
\begin{equation}
\eta=100\frac{\mathrm{RMSE}_{A}-\mathrm{RMSE}_{AC}}{\mathrm{RMSE}_{A}}.
\label{eq:gain}
\end{equation}
Uncertainty is evaluated by paired resampling of date blocks, retaining overlapping forecast records within the same resampled block. The final tables identify the evaluation partition and date count. Equal-capacity imagery features and frozen-retriever features provide complementary controls for model capacity and cloud-supervised representation.

\subsection{Cloud conditions and continuous events}
GHI errors are stratified by station-centre CPP reference COT using fixed near-zero, thin, moderate, and thick bins: $[0,0.1)$, $[0.1,5)$, $[5,30)$, and $[30,100]$. These comparisons use common reference-supported rows and report sample and date counts. The all-weather GHI comparison retains samples without CPP. Solar-geometry and regional stratification test whether an apparent cloud-thickness pattern reflects other conditions.

Continuous cases combine actual AGRI composites, retrieved and reference COT, and interval-mean GHI on a common time axis. Event definitions use cloud evolution and data completeness rather than improvement by one model. Cloud passage, growth, and dissipation are examined together with cases where COT does not improve GHI. These cases interpret the aggregate results rather than replace them.

\subsection{Image-error propagation through COT and GHI}
The mechanism analysis fixes the retriever, the COT-augmented GHI head, and the evaluated sample keys. Let $A^o$ and $A^f$ denote observed and forecast imagery, and let $C^o=R_C(A^o,g)$ and $C^f=R_C(A^f,g)$ denote the corresponding physical COT fields from the same retriever. The four input combinations $(A^o,C^o)$, $(A^f,C^o)$, $(A^o,C^f)$, and $(A^f,C^f)$ isolate sensitivity to the image and COT input routes. The first three use future observations and serve only as offline mechanism experiments.

All replacements use the same station--initialization--target keys, labels, validity support, weights, and the head's fixed preprocessing, including log-COT standardization. In the three-frame configuration, known historical positions remain identical; only future positions are replaced. For $a,b\in\{0,1\}$, define $E_{ab}$ as mean squared GHI error, with 0 denoting observed-derived inputs and 1 forecast-derived inputs. Then
\begin{equation}
E_{11}-E_{00}=\underbrace{(E_{10}-E_{00})}_{D_A}
+\underbrace{(E_{01}-E_{00})}_{D_C}
+\underbrace{(E_{11}-E_{10}-E_{01}+E_{00})}_{I}.
\label{eq:propagation}
\end{equation}
The interaction $I$ allows the GHI response to joint input errors to differ from the sum of separate replacements. This is an exact decomposition of prediction-error sensitivity, not a decomposition into independent physical causes. It uses MSE, not differences between incompatible RMSE cohorts. COT trajectory errors, cloud-boundary displacement, and timing of station irradiance changes provide physical interpretation of the replacement contrasts.

To distinguish a general forecast-error increase from a reduction in the value of COT, the imagery-only head is also evaluated on observed and forecast imagery on those same keys. Let its MSEs be $B_o$ and $B_f$. The COT advantages are $V_o=B_o-E_{00}$ and $V_f=B_f-E_{11}$, so
\begin{equation}
V_o-V_f=(E_{11}-E_{00})-(B_f-B_o).
\label{eq:value-loss}
\end{equation}
A positive value indicates that the COT-augmented head loses more performance under extrapolated inputs than the imagery-only head. A negative value indicates a larger COT advantage in forecasting. This comparison establishes the direction of change before attributing an attenuation of benefit.

\FloatBarrier
\section{Results}
\subsection{Quality of the retrieved COT representation}
The input-quality analysis compares retrieved COT with the CPP reference at the station pixel and local neighbourhood. It establishes the magnitude and optical-thickness dependence of retrieval errors before evaluating their relevance to GHI. \pending{Insert regional RMSE, MAE, bias, thick-cloud detection, and the common reference sample/date counts. Describe both agreement and systematic residuals.}
\figslot{COT density comparisons and thickness-dependent bias}{COT retrieval quality and its dependence on reference optical thickness. Regional panels use their respective common reference samples.}{fig:retrieval}

\subsection{Incremental value for irradiance estimation}
The main comparison measures the change in GHI error after adding COT to AGRI and solar geometry in Hubei; the Hunan comparison examines regional consistency. \pending{Insert matched regional RMSEs, relative reductions, paired intervals, and station-level directions.}

\pending{Insert the equal-capacity and retrieval-feature contrasts. State whether the observed gain is associated with the cloud-supervised representation and whether explicit COT differs from latent retrieval features. Preserve relevant null or adverse findings.}
\begin{table}[!htbp]\centering\small
\caption{Representation controls on common samples. Numerical entries will report matched GHI errors and paired differences.}\label{tab:controls}
\begin{tabularx}{\linewidth}{XXl}\toprule
Comparison & Question & Result\\\midrule
COT vs imagery features & Effect beyond added head capacity & \pending{values}\\
COT vs retrieval features & Explicit thickness vs cloud-supervised latent features & \pending{values}\\
COT in a stronger imagery model & Dependence on predictor representation & \pending{values}\\\bottomrule
\end{tabularx}\end{table}
\figslot{Station-wise paired RMSE and regional error differences}{Incremental value of COT for synchronous GHI estimation. Hubei is the primary region and Hunan is complementary; absolute errors are not pooled across regions.}{fig:gain}

\subsection{Dependence on optical thickness and cloud conditions}
\pending{Insert the common-sample GHI errors and gains for the four COT bins, including date counts and uncertainty. Identify the conditions with the largest improvement and those with no improvement or degradation.}

\pending{Relate the signed GHI error to the COT strata, and compare the pattern across stations and solar-geometry conditions. Distinguish an observed subgroup association from a universal monotonic thickness response.}
\figslot{GHI error and paired improvement across COT strata}{Cloud-condition dependence of COT-assisted irradiance estimation. Sample and date counts accompany each stratum; gains are evaluated within common support.}{fig:thickness}

\subsection{Cloud events and irradiance response}
\pending{Describe a continuous cloud-passage or growth/decay event: cloud location, COT change, measured GHI change, and the corresponding difference between irradiance models. Include a contrasting event with little gain or an adverse response.}
\figslot{AGRI and COT sequence above aligned GHI and COT trajectories}{Cloud evolution and station irradiance response. All curves use the same 15-minute target intervals. Cases illustrate both effective and ineffective use of the COT representation.}{fig:events}

\subsection{Forecasting gain and propagation of image errors}
\pending{Insert the same-sample forecasting RMSEs with and without COT, the relative gain and its uncertainty, and the lead-dependent pattern. Report the frozen-head and forecast-trained comparisons separately where both are available.}

\pending{Insert observed-derived versus forecast-derived COT trajectory errors and the four fixed-head input-replacement scores. Quantify the image route, COT route, and interaction using Eq.~\ref{eq:propagation}, and the change in COT advantage using Eq.~\ref{eq:value-loss}.}

\pending{Use the aligned cloud fields and GHI trajectories to identify whether cloud displacement, weakened optical-thickness evolution, or delayed growth/decay accounts for the error pattern. State the supported role of image extrapolation in limiting the forecast gain.}
\figslot{Lead-dependent GHI gain and fixed-head input-replacement contrasts}{Forecasting value of COT and sensitivity to cloud-image extrapolation errors. Forecast comparisons and offline observed-input replacements occupy separate panels.}{fig:forecast}

\FloatBarrier
\section{Discussion}
\subsection{Why an explicit cloud-property representation can help}
COT provides a physically defined coordinate for cloud attenuation within a learned irradiance mapping. Its contribution is practical if cloud-product supervision makes relevant information easier for a compact GHI model to use. This interpretation does not require COT to be an independent observation or the only useful representation. Capacity and retrieval-feature comparisons clarify how much of the measured benefit is shared with a richer representation of the same imagery. \pending{Integrate the final control results with the overall and cloud-condition findings.}

Thickness-dependent gains should be interpreted through both attenuation and irradiance variability. A local COT error can affect the inferred radiation level, while cloud edges, diffuse radiation, and changes during a 15-minute interval complicate the mapping to station GHI. The continuous cases connect these processes to the conditions in which the added representation succeeds or fails.

\subsection{Cloud-image extrapolation as a constraint on forecasting benefit}
The forecasting chain exposes two routes by which errors in extrapolated imagery can affect GHI. One changes the spectral and spatial input to the irradiance model directly. The other changes the COT inferred from that same imagery. Cloud displacement can shift the inferred attenuation away from the station or to the wrong target time. Spatial smoothing can weaken cloud boundaries, and errors in cloud growth or dissipation can suppress or delay changes in retrieved optical thickness. Consequently, an effective COT representation in observed imagery need not retain the same benefit when supplied with extrapolated fields.

\pending{Insert the evidence-backed attribution paragraph: relate the change in COT benefit on common samples to the image/COT replacement contrasts and event errors. Identify image extrapolation as the principal limitation only where the contrasts support that ranking; otherwise describe the measured contribution without assigning an unsupported fraction.}

This analysis motivates a distinct next step: improving cloud-image forecasts so that they preserve the location, boundaries, and optical evolution relevant to irradiance. An appropriate future test would hold the retriever and irradiance mapping fixed while improving the upstream image prediction, and evaluate whether better cloud evolution translates into larger GHI gains. The present study establishes the role of COT and examines its sensitivity to forecast inputs; the subsequent study can address how to reduce those upstream errors.

\subsection{Scope of the interpretation}
Regional sampling, reference-product uncertainty, and the finite number of independent weather periods delimit the conclusions. CPP-based error measures assess agreement with a retrieval reference, whereas GHI performance is assessed against station observations. Representation controls and adverse cases remain relevant to the interpretation even when the average COT effect is positive. \pending{Consolidate the final evaluation scope and the principal residual limitation here, without repeating operational status in the results.}

\section{Conclusions}
This study evaluates explicit COT as a local representation for satellite GHI estimation and follows its effect into short-term forecasting. \pending{State the final estimation gain and its regional and cloud-condition dependence.} \pending{State the final forecasting gain under the specified configuration and horizon.} \pending{Summarize the demonstrated influence of image-extrapolation errors on COT and GHI, and distinguish that influence from residual retrieval or mapping errors.} The resulting evidence provides a basis for improving irradiance-relevant cloud evolution in future image-extrapolation models.
\FloatBarrier
'''
keys = re.findall(r'\\cite\{([^}]+)\}', tex)
ordered = list(dict.fromkeys(k for group in keys for k in group.split(',')))
tex += '\\begin{thebibliography}{99}\\small\n'
tex += '\n'.join('\\bibitem{' + k + '}\n' + entries[k].strip() for k in ordered)
tex += '\n\\end{thebibliography}\n\\end{document}\n'

zh = r'''# 云光学厚度辅助的卫星辐照度反演及短时预报

## 摘要

多光谱卫星图像包含丰富的云信息，但从光谱学习地表辐照度的映射未必能显式利用云的辐射衰减特征。本文研究从同源图像中、通过云产品监督反演的云光学厚度（COT），能否改善总水平辐照度（GHI）估计，并在短时预报中保持作用。冻结反演器生成与AGRI和太阳几何空间对齐的局地COT场，输入紧凑辐照度模型。以湖北为主要研究区域、湖南为补充区域，通过配对实验分析总体反演精度、云厚度条件依赖及连续云事件，再将该框架应用于15～240分钟预测云图。输入替换实验用于追踪底图外推误差如何经过COT反演和辐照度映射传递。【待填：最终分区域反演增益、主要云况结论及配对预报增益。】【待填：误差传递发现及其对应结论。】本文将显式云物理表示的实际价值与辐照度预报所依赖的云场演变质量联系起来。

## 1 引言

云调制到达地面的太阳辐射，其位置和光学厚度的变化会引起站点辐照度波动。静止卫星多光谱图像提供对这些变化的高频观测。数据驱动模型可以利用光谱和空间模式，但对于容量有限的估计器，光谱描述与显式云物理表示未必同样易于利用。本文关注的问题是：在原始图像之外加入反演COT，能否改善站点GHI估计。

已有研究通过光谱选择与空间特征学习改进卫星辐照度估计[1]。云相关物理特征也可以补充图像模型：Liao和Coimbra比较有无SCOPE信息的CNN，并将方法扩展到短时预报[2]。这些研究促使我们进一步检验：保留原始多光谱观测时，显式光学厚度场还能增加多少价值。由于COT由这些观测反演得到，它增加的是通过云属性监督学习得到的表示，而非独立观测。

这种表示的作用可能随云况变化。云衰减误差影响辐照度幅值；云边界、太阳几何和平均区间内的演变则影响站点波动的时刻与强度。总体指标无法说明改善是否广泛出现，还是集中于特定厚度条件。因此，需要将配对总体评价与云况分层和连续事件分析结合。

短时预报引入额外困难：反演器和辐照度模型接收的云图本身是预测结果。基于云运动的研究已经分别分析同期估计误差与预报误差[3]，Cloud-FNO进一步研究多种云物理属性的演变预报[4]。在云图外推链中，云团位置、边界和生消过程的误差，既可以通过图像输入直接影响GHI，也可以经过COT反演间接传递。理解这两条路径，有助于解释COT作用从反演到预报的变化。

本文回答三个问题：（1）在AGRI与太阳几何之外，局地COT能改善多少GHI估计；（2）改善发生在哪些光学厚度条件和云事件中；（3）这种作用在15～240分钟预报中能保留多少，又如何受到云图外推误差限制。湖北开展主要站点比较，湖南独立训练模型提供补充区域证据。本篇重点研究云物理表示的价值及误差传递，更优云图外推模型的开发作为后续研究问题。

## 2 研究区域与数据

### 2.1 区域、卫星观测与云参考产品

主要区域包含湖北牛首、王寨和米庄，湖南四里和竺家构成补充区域。模型和归一化分别在各区域内拟合。由于地面观测原生时间采样不同，各区域结果分别报告。

AGRI输入为C01～C06、C09～C15共13个通道，使用各区域地理定位网格。图像、几何和COT由相同站点索引配准。COT反演器读取16×16裁块，辐照度模型使用其中心3×3邻域。CPP提供COT和云相参考，用于反演监督及云况分析；它本身是具有不确定性和非连续覆盖的卫星检索产品。主要GHI评价不以CPP是否存在作为样本筛选条件。

【表1待填：最终站点坐标、观测时期、原生采样、各划分日期数和样本数。】

**表1：区域观测与时间划分。坐标、时期、原生采样及划分数量由最终数据汇总填入。**

| 区域 | 站点 | 观测与划分细节 |
|---|---|---|
| 湖北 | 牛首、王寨、米庄 | 【待填：坐标、时期、采样、数量】 |
| 湖南 | 四里、竺家 | 【待填：坐标、时期、采样、数量】 |

**图1：研究区域与AGRI、GHI、CPP观测覆盖。** 可用记录的实际覆盖与首末记录跨度分别表达。

### 2.2 辐照度标签与太阳几何

令τ为15分钟目标区间结束时刻，站点s的目标为区间平均GHI。湖南5分钟区间末观测聚合为

$$\bar G_{s,\tau}=\frac13\sum_{j=0}^{2}G^{(5)}_{s,\tau-5j\,\mathrm{min}}.\tag{1}$$

湖北在相同右闭区间内按有效一分钟箱构造标签，不完整窗口排除。晴空辐照度使用Ineichen–Perez模型，并在相同区间取平均[5,6]。学习目标与还原为

$$k_{s,\tau}=\frac{\bar G_{s,\tau}}{\bar G^{cs}_{s,\tau}},\qquad\widehat G_{s,\tau}=\widehat k_{s,\tau}\bar G^{cs}_{s,\tau}.\tag{2}$$

晴空指数不截断到1。实测GHI仅作标签，不作为输入。太阳几何依据UTC、经纬度计算并与每幅图像配准。几何输入为太阳天顶角余弦、太阳与卫星相对方位角余弦；反演器另接收白天标记。

### 2.3 时间采样

训练、验证与最终评价按日期块分离。湖南在覆盖充足月份分别留出连续2天验证、2天测试，并设置前后1天缓冲；合格日少于20天的月份仅用于训练。每个预报样本包含间隔15分钟的8帧历史及16帧未来，全部帧属于同一划分。归一化只用训练数据拟合。【待填：表1中固定的湖北划分和最终评价样本。】

## 3 方法

### 3.1 监督COT反演

对于图像A和几何g，反演器输出非负对数COT场：

$$L_\tau=R_\theta(A_\tau,g_\tau),\qquad C_\tau=\exp(L_\tau)-1.\tag{3}$$

采用两级下采样U-Net，特征宽度为32、64、128，经跳连上采样恢复16×16网格；卷积块使用批归一化与SiLU，输出softplus保证L非负。使用CPP的log(1+COT)监督，损失为

$$\mathcal L_R=\frac{\sum_{i,p}m_{i,p}\rho_{0.1}\!\left(L_{i,p}-\log(1+C^{CPP}_{i,p})\right)}{\sum_{i,p}m_{i,p}}.\tag{4}$$

其中m表示输入与参考均有效的像元，保留有效零COT参考。推理沿用训练归一化，有效性依据可获得的输入和几何判定，不要求未来CPP参考存在。参考质量条件用于监督训练及参考产品评价。训练完成的反演器在GHI训练和预报中冻结。

### 3.2 辐照度模型的局地COT增强

以P_s选取站点中心3×3场，AGRI和对数COT按训练统计标准化：

$$X_{s,\tau,c}=\frac{\mathcal P_s A_{\tau,c}-\mu_c}{\sigma_c},\qquad Z_{s,\tau}=\frac{\mathcal P_s L_\tau-\mu_L}{\sigma_L}.\tag{5}$$

COT保留九像元排列及样本间幅值，不使用全图均值或分位数代替。两组映射为

$$\widehat k^A_{s,\tau}=H_A(X_{s,\tau},g_{s,\tau}),\qquad\widehat k^{AC}_{s,\tau}=H_{AC}(X_{s,\tau},g_{s,\tau},Z_{s,\tau}).\tag{6}$$

局地特征编码后，通过站点读出预测晴空指数。主反演比较使用目标时刻单帧；三帧配置另使用目标前30分钟、前15分钟及目标时刻，并单列为时间信息扩展。同一次有无COT比较保持帧数一致。

模型优化缩放后的物理GHI均方误差：

$$\mathcal L_H=\frac1N\sum_i\left[\frac{\widehat k_i\bar G^{cs}_i-\bar G_i}{1000\ \mathrm{W\,m^{-2}}}\right]^2.\tag{7}$$

配对训练保持样本、随机种子、样本顺序、优化预算和选模规则一致。最终结构与训练设置见表2。

**表2：模型结构及训练设置，单帧主比较与时间扩展分别报告。**

| 模块 | 固定输入定义 | 待填设置 |
|---|---|---|
| COT反演器 | 13个AGRI＋3个几何／白天通道；16×16 | 归一化、优化器、batch、轮数、选模 |
| GHI估计器 | 3×3 AGRI＋几何，有／无COT | 宽度、参数量、帧数、batch、更新数 |
| 底图外推 | 8帧历史、16帧未来 | 区域模型、归一化、权重选取规则 |

### 3.3 短时预报

以最后一张历史图像时刻t为初始化，固定外推模型产生

$$\widehat A_{t+k\Delta}=S_\psi(A_{t-7\Delta:t},g)_k,\qquad\widehat C_{t+k\Delta}=\exp R_\theta(\widehat A_{t+k\Delta},g_{t+k\Delta})-1.\tag{8}$$

k=1,…,16，Δ=15分钟。SimVP作为固定基线[7,8]；若评价其他底图，则使用相同下游对照。未来太阳几何可确定性计算；未来真实图像与CPP不进入预报输入。

将同一对冻结辐照度映射用于预测云图及其有无COT输入，检验反演表示在预报中的保留程度。专门使用预测云图训练的GHI头单列比较。三帧头的+15分钟包含2张已知历史帧，+30分钟包含1张，因此另报告≥45分钟、全部由预测图构成的输入结果。

**图2：COT辅助反演与预报流程。** 同期路径为观测AGRI→反演COT→GHI；预报路径为历史AGRI→预测AGRI→预测COT→GHI。有无COT对照仅改变局地COT表示是否加入，并区分历史观测与预测场。

## 4 实验设计

### 4.1 COT价值的配对评价

每次比较使用相同站点、初始化时间（如适用）、目标时间、GHI标签及有效白天掩膜。先逐站逐种子计算，再在区域内等权平均。报告物理RMSE、MAE、bias及相对RMSE降幅：

$$\eta=100\frac{\mathrm{RMSE}_A-\mathrm{RMSE}_{AC}}{\mathrm{RMSE}_A}.\tag{9}$$

区间由日期块配对重采样估计，重叠预报记录保留在相应日期块内。最终表格标明评价划分与日期数。容量匹配的图像特征及冻结反演器特征，分别用于检验额外容量与云监督表示的作用。

### 4.2 云况与连续事件

按站点中心CPP参考COT固定分为近零、薄、中等、厚四档：[0,0.1)、[0.1,5)、[5,30)、[30,100]。分层比较采用共同参考有效样本，并报告样本数和日期数；总体GHI比较保留缺CPP的样本。结合太阳几何与区域分层，检查厚度条件差异是否与其他条件相关。

连续事件在同一时间轴组合真实AGRI合成图、反演与参考COT、区间平均GHI。事件按云演变与数据完整性定义，不按模型增益选取。分析云团经过、增长、消散，以及COT无改善或退化的情况。个例用于解释总体结果，不代替总体评价。

### 4.3 底图误差经过COT和GHI的传递

固定反演器、含COT的GHI头及样本键。A^o与A^f分别为真实和预测云图；C^o=R_C(A^o,g)、C^f=R_C(A^f,g)为同一反演器产生的物理COT。比较(A^o,C^o)、(A^f,C^o)、(A^o,C^f)、(A^f,C^f)四种组合，分别检验图像与COT输入路径的敏感性。前三种包含未来观测，只用于离线机制分析。

全部替换使用同一站点—初始化—目标键、标签、有效支持、权重和头部固定预处理，包括对数COT标准化。三帧配置中已知历史位置保持相同，只替换未来位置。以E_ab表示GHI均方误差，0代表真实图派生输入，1代表预测图派生输入，则

$$E_{11}-E_{00}=\underbrace{(E_{10}-E_{00})}_{D_A}+\underbrace{(E_{01}-E_{00})}_{D_C}+\underbrace{(E_{11}-E_{10}-E_{01}+E_{00})}_{I}.\tag{10}$$

交互项I允许两种输入误差同时作用时不同于分别替换之和。该式是预测误差敏感性的精确分解，不是彼此独立的物理原因分解。使用MSE，不将不同样本的RMSE差拼接。结合COT轨迹误差、云边界位移及站点辐照度变化时刻，解释输入替换差异。

为区分一般的预报误差增加与COT价值下降，还在相同键上将无COT头分别用于真实及预测图像，记MSE为B_o与B_f。COT优势为V_o=B_o−E_00、V_f=B_f−E_11，则

$$V_o-V_f=(E_{11}-E_{00})-(B_f-B_o).\tag{11}$$

正值表示含COT头在预测输入下损失的性能多于无COT头，负值表示预报中的COT优势更大。先确定变化方向，再归因收益是否衰减。

## 5 结果

### 5.1 COT表示的反演质量

将反演COT与CPP参考在站点像元和局地邻域比较，先明确输入误差及其厚度依赖，再评价其对GHI的意义。【待填：分区域RMSE、MAE、bias、厚云检出指标及共同参考样本／日期数；同时写一致性与系统残差。】

**图3：COT密度对比与厚度依赖偏差。** 各区域使用各自共同参考样本。

### 5.2 对辐照度反演的增量价值

主比较衡量湖北在AGRI与太阳几何之外加入COT带来的GHI误差变化，湖南检验区域一致性。【待填：配对分区域RMSE、相对降幅、区间及逐站方向。】

【待填：容量匹配和反演器特征对照。说明增益与云监督表示的关系，以及显式COT是否与隐特征存在差异；保留相关无增益或退化结论。】

**表3：共同样本上的表示对照。数值项填写配对GHI误差与差值。**

| 比较 | 回答的问题 | 结果 |
|---|---|---|
| COT与图像特征 | 是否超出增加头部容量的作用 | 【待填】 |
| COT与反演器特征 | 显式厚度与云监督隐特征的差异 | 【待填】 |
| 更强图像模型中的COT | 对预测器表示能力的依赖 | 【待填】 |

**图4：逐站配对RMSE与分区域误差差值。** 湖北为主、湖南为补充，不混合两区域绝对误差。

### 5.3 光学厚度及云况依赖

【待填：四档共同样本上的GHI误差与增益、日期数及不确定性；明确最大收益、无改善及退化条件。】

【待填：GHI有符号误差与COT分层的关系，及其跨站点、太阳几何条件变化；区分实际分层关联与普遍单调规律。】

**图5：不同COT条件下的GHI误差与配对改善。** 各档标注样本与日期数，在共同支持内比较。

### 5.4 云事件与辐照度响应

【待填：连续云团经过或生消事件中，云的位置、COT变化、实测GHI变化与两种模型的差异；包含一个收益小或退化的对照事件。】

**图6：AGRI与COT序列，以及对齐的GHI／COT轨迹。** 采用同一15分钟目标区间，说明COT表示有效及失效的情况。

### 5.5 预报增益与底图误差传递

【待填：同样本有无COT预报RMSE、相对增益、区间与分时效模式；如两者均有结果，冻结头与预报专训头分别报告。】

【待填：真实图与预测图反演COT的轨迹误差，四种固定头输入替换分数；按式(10)量化图像路径、COT路径及交互作用，按式(11)量化COT优势的变化。】

【待填：利用对齐云场与GHI轨迹，识别云团位移、厚度演变减弱或生消延迟对应的误差模式；写出证据所支持的底图外推对预报增益的限制。】

**图7：分时效GHI增益与固定头输入替换对照。** 预报结果与包含真实未来输入的离线对照分面表达。

## 6 讨论

### 6.1 显式云物理表示为何可能有用

COT在学习得到的辐照度映射中提供了有明确物理定义的云衰减表征。若云产品监督使紧凑GHI模型更容易利用相关信息，其贡献就具有实际价值。这不要求COT是独立观测，也不要求它是唯一有效的表示。容量和反演器隐特征对照说明收益有多少可以由同源图像的更丰富表示共同解释。【待填：将最终控制结果与总体、云况结果结合解释。】

厚度依赖的收益应同时从辐射衰减和辐照度变化理解。局地COT误差影响辐射量级；云边界、散射辐射及15分钟内的变化使站点GHI映射更复杂。连续事件将这些过程与表示有效或失效的条件联系起来。

### 6.2 底图外推对预报增益的限制

底图外推误差可以通过两条路径影响GHI：一条直接改变辐照度模型接收的光谱和空间输入，另一条改变由同一预测图反演出的COT。云团位移可能使衰减信号偏离站点或出现在错误时刻；空间平滑可能削弱云边界；云团增长和消散误差可能压低或延迟光学厚度变化。因此，在真实云图中有效的COT表示，进入预测云场后未必保留相同程度的增益。

【待填有证据的归因段落：将同一样本上的COT收益变化，与图像／COT替换差异及事件误差联系。对照支持主次排序时，将底图外推识别为主要限制；否则量化已观测贡献，不填写未经证实的占比。】

由此引出独立的下一步问题：如何让云图预报更好地保留辐照度所需的云位置、边界和光学演变。后续研究可以固定反演器与辐照度映射，改进上游底图，并检查云演变改善能否转化为更大的GHI收益。本篇明确COT的作用及其对预报输入误差的敏感性，下一篇进一步研究如何降低这些上游误差。

### 6.3 解释范围

区域采样差异、参考产品不确定性和有限的独立天气过程决定结论的适用范围。CPP指标衡量与检索参考的一致性，GHI精度以站点观测评价。即使总体COT增益为正，表示对照及失败事件仍是解释的一部分。【待填：集中写最终评价范围及最主要剩余限制，不在结果中反复插入运行状态。】

## 7 结论

本文评价显式COT作为卫星GHI局地表示的作用，并追踪其在短时预报中的效果。【待填：最终反演增益及其区域、云况依赖。】【待填：明确配置与时效下的最终预报增益。】【待填：已证实的底图外推误差对COT和GHI的影响，与剩余反演／映射误差区分。】这些证据为后续改善与辐照度有关的云场演变预报提供依据。

## 参考文献

1. Chen et al. Global and direct solar irradiance estimation using deep learning and selected spectral satellite images. Applied Energy, 352, 121979, 2023. https://doi.org/10.1016/j.apenergy.2023.121979
2. Liao & Coimbra. Hybrid solar irradiance nowcasting and forecasting with the SCOPE method and convolutional neural networks. Renewable Energy, 232, 121055, 2024. https://doi.org/10.1016/j.renene.2024.121055
3. Papachristopoulou et al. Effects of clouds and aerosols on downwelling surface solar irradiance nowcasting and short-term forecasting. Atmospheric Measurement Techniques, 17, 1851–1877, 2024. https://doi.org/10.5194/amt-17-1851-2024
4. Zhang et al. Short-term forecasting of cloud physical properties based on Fourier neural operator method. Geophysical Research Letters, 53, e2025GL119553, 2026. https://doi.org/10.1029/2025GL119553
5. Ineichen & Perez. A new airmass independent formulation for the Linke turbidity coefficient. Solar Energy, 73, 151–157, 2002.
6. pvlib python contributors. Ineichen/Perez clear-sky model documentation. https://pvlib-python.readthedocs.io/en/stable/_modules/pvlib/clearsky.html
7. Gao et al. SimVP: Simpler Yet Better Video Prediction. CVPR, 3170–3180, 2022. https://openaccess.thecvf.com/content/CVPR2022/html/Gao_SimVP_Simpler_Yet_Better_Video_Prediction_CVPR_2022_paper.html
8. Tan et al. SimVPv2: Towards Simple yet Powerful Spatiotemporal Predictive Learning. arXiv:2211.12509. https://arxiv.org/abs/2211.12509
'''
(root / 'main.tex').write_text(tex, encoding='utf-8')
(root / 'main_zh.md').write_text(zh, encoding='utf-8')
print('Wrote bilingual structural draft; preceding manuscripts retained in', archive)
