"""Rebuild a standalone, bilingual visual-review manuscript from frozen draft sources.

Only assets explicitly listed in FIGURE_ASSETS.json are embedded. Missing data
panels are labelled templates, never synthetic observations. Quantitative plots
are copied byte-for-byte from source exports; this script does not rescore data.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil

ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parent
BASE = PAPER / "draft_v1_20261030"
SNAP = ROOT / "source_snapshot"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rep(text, old, new):
    if old not in text:
        raise ValueError(f"Expected source text missing: {old[:110]}")
    return text.replace(old, new, 1)


def replace_figure(tex, label, content):
    # A reserved slot or a complete figure environment, addressed by stable label.
    slot = re.compile(r"\\figslot\{[^\n]*\}\{[^\n]*\}\{" + re.escape(label) + r"\}")
    tex, count = slot.subn(lambda m: content, tex)
    if count == 1:
        return tex
    figures = re.compile(r"\\begin\{figure\}\[[^\]]*\][\s\S]*?\\end\{figure\}")
    count = 0
    def change(m):
        nonlocal count
        if r"\label{" + label + "}" in m.group(0):
            count += 1
            return content
        return m.group(0)
    tex = figures.sub(change, tex)
    if count != 1:
        raise ValueError(f"Figure {label}: expected one occurrence, found {count}")
    return tex


def fig_tex(path, caption, label, height="0.64"):
    return (r"\begin{figure}[!htbp]\centering" + "\n"
            + rf"\includegraphics[width=\linewidth,height={height}\textheight,keepaspectratio]{{{path}}}" + "\n"
            + rf"\caption{{{caption}}}\label{{{label}}}" + "\n" + r"\end{figure}")


CAPTIONS_EN = {
"fig:regions": "Study regions and observation coverage. The layout reserves separate Hubei and Hunan panels and distinguishes AGRI, station GHI and CPP reference coverage. This is a design template: geographic positions, coverage dates and valid record counts await verified data; no observation availability is inferred from empty panels.",
"fig:architecture": r"Single-frame COT-assisted GHI estimation, illustrated for the Hunan implementation. A $16\times16$ AGRI--geometry context enters a U-Net retriever; its nonnegative log-COT output is converted to physical COT. The GHI model reads aligned $3\times3$ AGRI, geometry and standardized log-COT fields. Ordered features are concatenated and flattened, then mapped to the clear-sky index and multiplied by interval-mean clear-sky GHI. Dashed paths denote supervision. Cloud textures are schematic, not observations. This AI-assisted visual preview depicts the method; regional model sizes and the Hubei auxiliary-slot variant are specified in Table~\ref{tab:model_sizes}.",
"fig:framework": r"Forecast comparison and causal input flow. Eight observed historical AGRI frames supply the image predictor and historical COT. SimVP generates 16 future AGRI frames, from which the fixed retriever derives future COT. The four comparisons separate no COT, historical COT, a matched correction control, and a forecast-COT correction. The latter two share a frozen historical-COT base; neutral future-COT slots in the control preserve correction capacity. Signed corrections are zero-initialized and gated; unavailable future COT falls back to the base. Future observed imagery and CPP are excluded from this inference flow. Tensor drawings are illustrative, not measurements; input definitions are listed in Table~\ref{tab:forecast_inputs}.",
"fig:retrieval": "Reserved presentation of COT retrieval quality. Hubei and Hunan density comparisons against CPP references and thickness-dependent bias occupy separate panels. Axes, one-to-one reference lines and COT bins define the intended comparisons. The panels contain no measured density, bias estimates or sample counts; these will be inserted from the matched reference evaluation.",
"fig:gain": r"Reserved presentation of the contribution of COT to GHI estimation from observed AGRI. Station-wise RMSE and paired COT-minus-baseline differences are separated by region; negative differences denote lower error. The five stations specify the study design, not five demonstrated improvements. No result points or confidence intervals are plotted; values and paired date-block intervals await the final common-sample comparison.",
"fig:thickness": r"Reserved presentation of GHI errors and COT-minus-baseline differences across fixed CPP reference COT bins: $[0,0.1)$, $[0.1,5)$, $[5,30)$ and $[30,100]$. Each regional panel will use identical valid reference samples within a bin and report sample and date support. Empty plotting areas indicate missing results, not zero error or zero COT effect.",
"fig:events": "CAPTION_PENDING_REAL_EVENT",
"fig:forecast": r"Paired GHI forecast evaluation on 15,275 common station--initialization--target records from 22 validation target dates. (a) Four model variants over +15--240 min from the last observed frame. (b) Forecast-COT correction minus the matched correction control, with pointwise 95\% paired target-date bootstrap intervals. (c,d) Historical-COT-minus-no-COT and forecast-COT-minus-control differences for Niushou, Wangzhai, Mizhuang and their equal-station mean; both panels share this station order. Diamonds average three seed-specific scores; open circles show individual seeds; whiskers are 95\% intervals from 5,000 paired date-block draws. Negative differences improve RMSE. Panel scales differ. The correction models share the frozen historical-COT base. Future observed AGRI/CPP is not an inference input. These are development-validation results, not an independent test.",
"fig:replacement": "CAPTION_PENDING_SOURCE_REPLACEMENT",
"fig:probe": "CAPTION_PENDING_PROBE",
"fig:three_station_event": "CAPTION_PENDING_THREE_STATION_EVENT",
}

CAPTIONS_ZH = {
"fig:regions": "研究区域与观测覆盖。分别预留湖北、湖南区域面板及 AGRI、站点 GHI、CPP 参考覆盖。此图是版式模板：地理位置、覆盖日期和有效记录数待核验后填入；空白面板不表示观测连续或缺测情况。",
"fig:architecture": "单帧 COT 辅助 GHI 估计，以湖南实现为示意。16×16 AGRI 与几何上下文进入 U-Net，非负对数 COT 输出转换为物理 COT。GHI 模型读取对齐的3×3 AGRI、几何和标准化对数 COT，经有序拼接、展平与读出得到晴空指数，再乘区间平均晴空 GHI。虚线表示监督，云图纹理为示意而非观测。本图为 AI 辅助视觉预览；区域模型规模与湖北辅助槽变体见表 S1。",
"fig:framework": "预报对照及因果输入流程。8帧真实历史 AGRI 用于图像外推和历史 COT；SimVP 产生16帧未来 AGRI，固定反演器生成未来 COT。四臂分别为无 COT、历史 COT、匹配控制修正及预测 COT 修正。后两臂共享冻结历史 COT 基底；控制臂将未来 COT 槽置中性零以保持修正容量。修正有正有负，输出零初始化并由门控调节；未来 COT 不可用时回退基底。未来真实图像与 CPP 不进入该推理流程。张量图为示意，不是测量；输入定义见表 S2。",
"fig:retrieval": "COT 反演质量的预留版式。湖北、湖南分别展示相对于 CPP 参考的密度比较与厚度分层偏差。坐标、1:1参考线和 COT 档位规定拟比较内容；图内没有实测密度、偏差或样本数，需由共同参考样本评价结果填入。",
"fig:gain": "观测 AGRI 估计 GHI 时 COT 增量的预留版式。逐站 RMSE 与加 COT 减基线的配对差按地区分开，负值表示误差减小。五站表示研究设计，不表示五站均已改善。图内不含结果点或置信区间，等待最终同样本比较与日期块区间。",
"fig:thickness": "固定 CPP 参考 COT 档位 [0,0.1)、[0.1,5)、[5,30)、[30,100] 下的 GHI 误差及加 COT 减基线差值预留版式。各区域每一档内采用相同有效参考样本，并报告样本数、日期数；空白表示结果待填，不代表零误差或 COT 无效。",
"fig:forecast": "22个验证目标日期、15,275条共同站点—初始化—目标时刻记录上的 GHI 预报评价。(a) 四种模型在相对历史末帧+15～240分钟的 RMSE。(b) 未来 COT 修正减同容量控制修正，阴影为逐时效目标日期块配对95%区间。(c,d) 历史 COT 减无 COT、未来 COT 修正减控制修正；行序相同，为牛首、王寨、米庄和三站等权。菱形为三个种子分别评分后的均值，空心圆为单种子，横线为5,000次配对日期块bootstrap的95%区间；负值改善，面板尺度不同。后两模型共享冻结历史 COT 基底，推理不含未来真实 AGRI 或 CPP。这是开发验证，不是独立测试。",
}

SUPPLEMENT_EN = r"""
\clearpage
\section*{Supplementary material}
\setcounter{figure}{0}
\renewcommand{\thefigure}{S\arabic{figure}}
\setcounter{table}{0}
\renewcommand{\thetable}{S\arabic{table}}

\subsection*{S1. Model sizes and comparison inputs}
The parameter counts in Table~\ref{tab:model_sizes} describe the implemented model classes, excluding optimizer states. They do not by themselves establish equivalent regional training or predictive performance. The Hunan auxiliary-image and retrieval-feature variants match one another but each contains 124 more trainable head parameters than the explicit-COT variant. The Hubei four-arm synchronous design uses identical tensor shapes and parameter counts with inactive auxiliary slots set to zero.

\begin{table}[!htbp]\centering\small
\caption{Implemented model dimensions and parameter counts. A retrieval model is frozen while fitting a GHI estimator. Head counts include the trainable station mapping; they do not include the separate frozen retriever.}\label{tab:model_sizes}
\begin{tabularx}{\linewidth}{p{29mm}Xrr}\toprule
Region and component & Input / feature definition & Readout width & Parameters\\\midrule
Hunan COT retriever & $16\times16$, 16 channels; widths 32--64--128 & --- & 470,913\\
Hubei COT retriever & $16\times16$, 16 channels; widths 16--32--64 & --- & 119,233\\
Hunan imagery head & $3\times3$ AGRI + geometry; $32\times3\times3$ features & 288 & 22,177\\
Hunan COT head & Imagery features + four COT feature channels & 324 & 24,489\\
Hunan auxiliary-image head & Additional learned image features & 324 & 24,613\\
Hunan retrieval-feature head & Hidden features of the frozen retriever & 324 & 24,613\\
Hubei synchronous head & Two auxiliary slots mapped $2\to4$; inactive slots zero & 324 & 24,489\\\bottomrule
\end{tabularx}
\end{table}

\begin{table}[!htbp]\centering\small
\caption{Inputs and fitted components in the four-arm forecasting comparison. All arms receive the same forecast AGRI and deterministic solar geometry. Historical observations end at initialization; forecast COT is derived from predicted AGRI.}\label{tab:forecast_inputs}
\begin{tabularx}{\linewidth}{p{29mm}XXX}\toprule
Comparison & Historical COT & Future COT & Fitted component\\\midrule
No COT & Neutral-zero slots & Absent & Base irradiance model\\
Historical COT & Eight observed-history retrieved fields & Absent & Otherwise matched base model\\
Historical COT + control & Same frozen historical-COT base & Neutral-zero slots in the correction & Gated signed residual head\\
Historical COT + forecast COT & Same frozen historical-COT base & Forecast-image COT and changes & Equal-capacity gated signed residual head\\\bottomrule
\end{tabularx}
\medskip
\begin{minipage}{\linewidth}\footnotesize
Base and correction stages each use 30 epochs, batch 1,536, AdamW with learning rate $10^{-3}$ and weight decay $10^{-4}$, and cosine decay to $10^{-5}$; seeds are 42, 43 and 44. Base and correction dates are disjoint. Correction outputs are zero-initialized, signed, and combined with a sigmoid gate. Both correction arms use the same future-input availability mask; unavailable input returns the frozen base prediction. This table concerns the forecasting experiment, not the Hubei synchronous auxiliary-slot comparison.
\end{minipage}
\end{table}
\FloatBarrier

\subsection*{S2. Conditional COT probe on development dates}
A separate exploratory probe uses 32,838 common records from 50 development target dates between 6 August 2025 and 12 July 2026. A deterministic ridge correction is evaluated with three target-date folds; each of three stations is assessed using three frozen baseline sources. These sources are not three newly trained probe seeds. Mean GHI RMSE is 118.799 W m$^{-2}$ for the existing frozen baseline, 124.850 for context-only correction, 124.307 after adding forecast COT to that context, and 99.569 when actual future imagery supplies the retrieved COT instead. Thus the forecast-COT probe remains worse than the existing baseline even though it slightly improves the context-only probe. The observed-future alternative is an offline information diagnostic. These development rows differ from the validation cohort in Fig.~\ref{fig:forecast}; their absolute RMSE differences are not a validation-to-oracle gain.
% FIGURE_PROBE
\FloatBarrier

\subsection*{S3. Same-event trajectories at all three stations}
Figure~\ref{fig:three_station_event} retains the three station trajectories for the event in Fig.~\ref{fig:events}. The common initialization and valid-row masks are preserved, with gaps retained where a station lacks a valid 15-minute GHI target. The station curves expand the spatial context of one event; they are not three independently selected events or evidence of an aggregate COT benefit.
% FIGURE_THREE_STATION
\FloatBarrier
"""

SUPPLEMENT_ZH = """
## 补充材料

### S1 模型规模与对照输入

表 S1 的参数量对应实际模型类，不含优化器状态，不据此认定两区域训练或预测性能等价。湖南额外图像特征与反演隐特征头彼此等容量，但各比显式 COT 头多124个可训练参数。湖北同期四臂保持相同张量形状与参数量，不活动的辅助槽置零。

**表 S1：实际模型维度与参数量。拟合 GHI 估计器时反演器冻结；头部计数含可训练站点映射，不含独立冻结反演器。**

| 区域与模块 | 输入／特征定义 | 读出维度 | 参数量 |
|---|---|---:|---:|
| 湖南 COT 反演器 | 16×16、16通道；宽度32–64–128 | — | 470,913 |
| 湖北 COT 反演器 | 16×16、16通道；宽度16–32–64 | — | 119,233 |
| 湖南图像头 | 3×3 AGRI＋几何；32×3×3特征 | 288 | 22,177 |
| 湖南 COT 头 | 图像特征＋4通道 COT 特征 | 324 | 24,489 |
| 湖南辅助图像头 | 额外可学习图像特征 | 324 | 24,613 |
| 湖南反演特征头 | 冻结反演器隐特征 | 324 | 24,613 |
| 湖北同期头 | 两辅助槽2→4编码；不活动槽置零 | 324 | 24,489 |

**表 S2：预报四臂输入与拟合组件。所有臂采用相同预测 AGRI 与确定性太阳几何。历史观测止于初始化，未来 COT 来自预测 AGRI。**

| 比较 | 历史 COT | 未来 COT | 拟合组件 |
|---|---|---|---|
| 无 COT | 中性零槽 | 不含 | 基底辐照度模型 |
| 历史 COT | 8帧真实历史反演场 | 不含 | 其他设置匹配的基底模型 |
| 历史 COT＋控制 | 同一冻结历史 COT 基底 | 修正中的未来槽为中性零 | 门控有符号残差头 |
| 历史 COT＋预测 COT | 同一冻结历史 COT 基底 | 预测图反演 COT 及其变化 | 等容量门控有符号残差头 |

基底与修正阶段各30轮、batch 1,536；AdamW学习率10⁻³、weight decay 10⁻⁴，余弦衰减至10⁻⁵；种子42、43、44。基底日期与修正日期分开。修正输出零初始化，可正可负，由 sigmoid 门控调节。两个修正臂使用同一未来输入可用性掩膜，输入不可用时回退冻结基底。本表仅指预报实验，不是湖北同期双辅助槽对照。

### S2 开发日期上的条件 COT 探针

独立探索性探针使用2025年8月6日至2026年7月12日之间50个开发目标日期、32,838条共同记录。确定性岭回归修正按目标日期三折评价，每个站点采用3份冻结基线来源；它们不是三个新训练的探针种子。平均 GHI RMSE 分别为：既有冻结基线118.799 W/m²，仅上下文修正124.850，加入预测 COT 后124.307，改用真实未来图像反演 COT 后99.569。因此预测 COT 虽略优于仅上下文探针，仍差于既有基线。真实未来输入仅作离线信息诊断。此开发样本不同于图8验证队列，不能将绝对 RMSE 差解释为验证到 oracle 的增益。

<!-- FIGURE_PROBE -->

### S3 同一事件的三站轨迹

图 S2 保留图7事件的三个站点轨迹，初始化和有效样本掩膜一致；缺少有效15分钟 GHI 标签时保留断线。三站轨迹扩展同一事件的空间背景，并不是三个独立选取的事件，也不能证明总体 COT 收益。

<!-- FIGURE_THREE_STATION -->
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-missing", action="store_true", help="Prepare source before all figures arrive; do not deliver.")
    args = parser.parse_args()
    SNAP.mkdir(exist_ok=True)
    for name in ("main.tex", "main_zh.md"):
        if not (SNAP / name).exists():
            shutil.copy2(BASE / name, SNAP / name)
    cfg = json.loads((ROOT / "FIGURE_ASSETS.json").read_text(encoding="utf-8"))
    integration = cfg.get("real_figure_integration")
    if integration:
        extra = json.loads((PAPER / integration).read_text(encoding="utf-8"))
        # Schema deliberately explicit: root can merge supplied captions and paths.
        if "captions_en" in extra:
            CAPTIONS_EN.update(extra["captions_en"])
        if "captions_zh" in extra:
            CAPTIONS_ZH.update(extra["captions_zh"])
    CAPTIONS_EN.update(cfg.get("captions_en", {}))
    CAPTIONS_ZH.update(cfg.get("captions_zh", {}))
    images = ROOT / "images"
    images.mkdir(exist_ok=True)
    assets, missing = {}, []
    for label, item in cfg["figures"].items():
        asset = PAPER / item["source"] if item.get("source") else None
        if asset is None or not asset.exists():
            missing.append(label)
            continue
        dest = images / (item["name"] + asset.suffix)
        shutil.copy2(asset, dest)
        png_source = PAPER / item.get("png_source", item["source"])
        if png_source.suffix.lower() == ".pdf":
            png_source = png_source.with_suffix(".png")
        if png_source.exists() and png_source.suffix.lower() == ".png":
            pngdest = images / (item["name"] + ".png")
            shutil.copy2(png_source, pngdest)
        else:
            pngdest = dest
        assets[label] = {"local": "images/" + dest.name, "markdown": pngdest.as_posix(), "source": str(asset), "sha256": sha(dest), "kind": item["kind"]}
    if missing and not args.allow_missing:
        raise FileNotFoundError("Required figures are not ready: " + ", ".join(missing))
    en = (SNAP / "main.tex").read_text(encoding="utf-8-sig")
    zh = (SNAP / "main_zh.md").read_text(encoding="utf-8-sig")
    en = rep(en, r"\usepackage[margin=25mm]{geometry}", r"\usepackage[a4paper,margin=16.5mm,headheight=14pt]{geometry}")
    en = rep(en, r"\usepackage{amsmath,amssymb,graphicx,booktabs,tabularx,placeins,url}", r"""\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb,graphicx,booktabs,tabularx,placeins,url,caption,xcolor}
\captionsetup{font=small,labelfont=bf,skip=7pt}
\definecolor{pendingcolor}{RGB}{135,77,19}
\setlength{\parskip}{3pt}
\setlength{\tabcolsep}{4.5pt}
\renewcommand{\arraystretch}{1.15}
\renewcommand{\topfraction}{0.92}
\renewcommand{\bottomfraction}{0.85}
\renewcommand{\textfraction}{0.08}
\renewcommand{\floatpagefraction}{0.70}
\setcounter{topnumber}{3}
\setcounter{bottomnumber}{2}""")
    en = rep(en, r"\newcommand{\pending}[1]{\textit{[To be completed: #1]}}", r"\newcommand{\pending}[1]{{\color{pendingcolor}\textit{[To be completed: #1]}}}")
    en = rep(en, "Figure-integrated review draft, 30 October 2026. Unresolved result fields remain explicit.", "Visual review edition, 30 October 2026. Empty result panels are labelled templates.")
    zh = rep(zh, "> 图表整合审阅稿，2026-10-30。尚未核验的结果保留待填标记。", "> 视觉整合审阅版，2026-10-30。空白结果面板均明确标为模板；示意图与实际数据图分开标注。")
    en = en.replace(r"\pending{Table 1:", r"\pending{Table 2:").replace("Complete Table 1 with", "Complete Table 2 with")
    en = rep(en, "Table~\\ref{tab:training} specifies the final architecture and settings for each evaluated configuration.", "Table~\\ref{tab:training} collects the architecture and training definitions. Implemented dimensions and parameter counts are provided in Table~\\ref{tab:model_sizes}; unavailable training details remain explicit.")
    zh = rep(zh, "最终结构与训练设置见表3。", "结构与训练定义汇总于表3；实际维度和参数量见表 S1，尚未核验的训练细节保留待填。")
    en = rep(en, r"\pending{widths, parameters, frame count, batch, updates}", r"Single frame; dimensions and parameters in Table~\ref{tab:model_sizes}; \pending{batch, updates}")
    zh = rep(zh, "| GHI估计器 | 3×3 AGRI＋几何，有／无COT | 宽度、参数量、帧数、batch、更新数 |", "| GHI估计器 | 3×3 AGRI＋几何，有／无COT | 单帧；维度和参数量见表 S1；【待填：batch、更新数】 |")
    en = rep(en, "This is a separately trained forecasting comparison, not a direct reuse of the synchronous head's weights.", "This is a separately trained forecasting comparison, not a direct reuse of the synchronous head's weights. Table~\\ref{tab:forecast_inputs} summarizes the four input definitions.")
    zh = rep(zh, "这是独立训练的预报对照，不是直接复用同期头权重。", "这是独立训练的预报对照，不是直接复用同期头权重；四臂输入定义见表 S2。")
    old_event = r"\pending{Describe a continuous cloud-passage or growth/decay event: cloud location, COT change, measured GHI change, and the corresponding difference between irradiance models. Include a contrasting event with little gain or an adverse response.}"
    new_event = r"""Figure~\ref{fig:events} shows a development-date source diagnostic at Niushou, initialized at 12:00 BJT on 6 August 2025. The date is the earliest eligible correction-development date with all three stations, and the initialization is the valid time nearest noon; the case was not selected for the largest model error. Observed and SimVP imagery are displayed with COT retrieved by the same model at +15, +120 and +240 min. The paired trajectories use the saved 15-minute GHI targets and station-pixel COT. The observed-future alternatives are offline diagnostics, and the plotted baseline already contains historical COT. These panels expose differences in image structure and retrieved thickness without treating either retrieved field as CPP truth or assigning a unique physical cause. The three-station trajectories are retained in Fig.~\ref{fig:three_station_event}.

\pending{Add synchronous paired GHI-estimation events, including a beneficial and a non-beneficial COT case, after the common-sample results are available. The forecast-source case above does not fill that evidence gap.}"""
    en = rep(en, old_event, new_event)
    zh = rep(zh, "【待填：连续云团经过或生消事件中，云的位置、COT变化、实测GHI变化与两种模型的差异；包含一个收益小或退化的对照事件。】", "图7展示牛首的开发日期来源诊断，初始化为2025年8月6日北京时间12:00。该日是三个站点均覆盖的最早合格修正开发日期，取最接近正午的有效初始化，不按最大模型误差选例。在+15、+120、+240分钟并排展示真实、SimVP图像及同一反演器得到的 COT；轨迹采用保存的15分钟 GHI 标签与站点像元 COT。真实未来输入仅作离线诊断，图中基线已含历史 COT。该图呈现图像结构与反演厚度差异，不将任一反演场视作 CPP 真值，也不指定唯一物理原因；三站轨迹见图 S2。\n\n【待填：共同样本结果齐备后，补充同期 GHI 估计的配对云事件，包括 COT 有益与无益个例。上述预报来源案例不替代这部分证据。】")
    # Known validation result is retained in the abstract and conclusion.
    en = rep(en, r"\pending{Forecasting results and the measured contribution of cloud-image prediction errors.}", "In the three-station forecasting comparison, historical COT reduced mean RMSE by 1.823\\% at the point estimate, with a paired date-block interval crossing zero; adding future predicted COT to a matched correction did not demonstrate additional benefit.")
    zh = rep(zh, "【待填：预报结果及云图预测误差的影响。】", "三站预报对照中，历史 COT 的平均 RMSE 点降幅为1.823%，但配对日期块区间跨零；在匹配修正中加入未来预测 COT 未显示额外收益。")
    en = rep(en, r"\pending{State the final forecasting gain under the specified configuration and horizon.}", "Over +15--240 min, the historical-COT forecast comparison yields a 1.823\\% point reduction with uncertainty spanning zero, while the future-COT correction shows no resolved additional benefit against its matched control.")
    zh = rep(zh, "【待填：明确配置与时效下的最终预报增益。】", "在+15～240分钟，历史 COT 预报对照的 RMSE 点降幅为1.823%，不确定区间跨零；未来 COT 修正相对匹配控制未显示可辨别的额外收益。")
    labels = ["fig:regions", "fig:architecture", "fig:framework", "fig:retrieval", "fig:gain", "fig:thickness", "fig:events", "fig:forecast", "fig:replacement"]
    # Remove old markdown image links before attaching one exact image per caption.
    zh = re.sub(r"!\[[^\]]*\]\([^\n]*\)\s*\n", "", zh)
    for index, label in enumerate(labels, 1):
        if label not in assets:
            continue
        caption = CAPTIONS_EN[label]
        if caption.startswith("CAPTION_PENDING"):
            if not args.allow_missing:
                raise ValueError("Missing final caption for " + label)
            continue
        en = replace_figure(en, label, fig_tex(assets[label]["local"], caption, label, "0.68" if label == "fig:events" else "0.62"))
        pattern = re.compile(r"\*\*图" + str(index) + r"[：:][^\n]+")
        zcaption = CAPTIONS_ZH[label]
        replacement = "![图" + str(index) + "](" + assets[label]["markdown"] + ")\n\n**图" + str(index) + "：" + zcaption + "**"
        zh, count = pattern.subn(lambda m: replacement, zh)
        if count != 1:
            raise ValueError(f"Chinese caption {index}: {count} matches")
    # Place the Chinese architecture diagram at the same point as in the English text.
    architecture = re.search(r"!\[图2\][\s\S]*?\*\*图2：[^^\n]*\*\*", zh)
    if architecture:
        block = architecture.group(0)
        zh = zh.replace(block, "", 1)
        zh = rep(zh, "模型优化缩放后的物理GHI均方误差：", block + "\n\n模型优化缩放后的物理GHI均方误差：")
    # Keep each figure near its own scientific subsection without forcing every figure.
    en = en.replace(r"\subsection{", r"\FloatBarrier" + "\n" + r"\subsection{")
    supp_en, supp_zh = SUPPLEMENT_EN, SUPPLEMENT_ZH
    for token, label, number in [("PROBE", "fig:probe", "S1"), ("THREE_STATION", "fig:three_station_event", "S2")]:
        if label not in assets or CAPTIONS_EN[label].startswith("CAPTION_PENDING"):
            if not args.allow_missing:
                raise ValueError("Missing supplementary figure/caption " + label)
            continue
        supp_en = supp_en.replace("% FIGURE_" + token, fig_tex(assets[label]["local"], CAPTIONS_EN[label], label, "0.70"))
        supp_zh = supp_zh.replace("<!-- FIGURE_" + token + " -->", f"![图{number}]({assets[label]['markdown']})\n\n**图{number}：{CAPTIONS_ZH[label]}**")
    en = rep(en, r"\end{document}", supp_en + "\n" + r"\end{document}")
    zh = zh.rstrip() + "\n\n" + supp_zh.strip() + "\n"
    (ROOT / "main.tex").write_text(en, encoding="utf-8")
    (ROOT / "main_zh.md").write_text(zh, encoding="utf-8")
    report = {"base_source_sha256": {n: sha(SNAP / n) for n in ("main.tex", "main_zh.md")}, "figure_assets": assets, "missing_assets": missing, "main_figures": 9, "supplement_figures": 2, "main_tables": 6, "supplement_tables": 2, "quantitative_values_changed": False, "new_experimental_results_claimed": False, "layout_only_assets_are_labelled": True, "primary_manuscript_edited": False}
    (ROOT / "INTEGRATION_MANIFEST.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(ROOT), "figures_copied": len(assets), "missing": missing}, ensure_ascii=False))


if __name__ == "__main__":
    main()
