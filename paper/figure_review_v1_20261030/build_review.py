"""Assemble the first figure/table review edition from verified local artifacts."""
from pathlib import Path
import csv, json, shutil, hashlib, re

P=Path(__file__).resolve().parent
PAPER=P.parent
F=PAPER/'figures/revision_v1_20261030'
E=PAPER/'revision_records/20261030_v1/evidence'

def csvrows(name):
    with (E/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def esc(x):
    return str(x).replace('&',r'\&').replace('%',r'\%').replace('_',r'\_')
def table(headers,rows,cols=None):
    if cols is None:cols='X'+'r'*(len(headers)-1)
    return r'\begin{tabularx}{\linewidth}{'+cols+r'}\toprule'+'\n'+' & '.join(headers)+r'\\\midrule'+'\n'+'\n'.join(' & '.join(map(str,r))+r'\\' for r in rows)+r'\bottomrule\end{tabularx}'+'\n'
def img(filename,width=177):
    return r'\begin{center}\includegraphics[width='+str(width)+r'mm]{images/'+filename+r'}\end{center}'+'\n'
def para(text):return text+'\n\n'
def section(num,title,sub):
    return r'\clearpage\noindent{\color{accent}\sffamily\bfseries\fontsize{23}{28}\selectfont '+num+r'}\quad{\sffamily\bfseries\Large '+title+r'}\par\vspace{2mm}{\color{muted}'+sub+r'}\par\vspace{4mm}'+'\n'
def note(text):return r'{\small\color{muted}'+re.sub(r'(?<!\\)%',r'\\%',text)+r'\par}'+'\n'

def main():
    image_sources={
      'architecture.pdf':F/'architecture/fig_core_architecture.pdf',
      'workflow.pdf':F/'workflow/fig_workflow.pdf',
      'forecast.pdf':F/'evidence/forecast_ghi.pdf',
      'event.pdf':F/'real_case/fig_cloud_source_mechanism.pdf',
      'stations.pdf':F/'real_case/fig_three_station_source_response.pdf',
      'probe.pdf':F/'probe/cot_conditional_information.pdf',
      'replacement.pdf':F/'source_replacement/source_replacement.pdf'}
    (P/'images').mkdir(exist_ok=True)
    for dst,src in image_sources.items():
        if not src.exists():raise FileNotFoundError(src)
        shutil.copy2(src,P/'images'/dst)
    source=json.loads((F/'source_replacement/summary.json').read_text())
    frows=csvrows('hubei_forecast_station_arm_metrics.csv')
    fcompare=csvrows('hubei_forecast_paired_contrasts.csv')
    hrows=csvrows('hunan_singleframe_station_arm_metrics.csv')
    hcompare=csvrows('hunan_singleframe_paired_contrasts.csv')
    names={'Niushou':'牛首','Wangzhai':'王寨','Mizhuang':'米庄','sili':'四里','zhujia':'竺家','Equal-station mean':'站点等权'}
    body=r'''\documentclass[UTF8,a4paper,10pt,fontset=none]{ctexart}
\usepackage[left=16.5mm,right=16.5mm,top=17mm,bottom=17mm]{geometry}
% Compatibility with the locally installed pre-2025 LaTeX format.
\providecommand{\IfDocumentMetadataT}[1]{}
\usepackage{amsmath,amssymb,graphicx,booktabs,tabularx,array,xcolor,fancyhdr,hyperref}
\setmainfont{Times New Roman}
\setsansfont{Arial}
\setCJKmainfont[BoldFont=Microsoft YaHei]{SimSun}
\setCJKsansfont{Microsoft YaHei}
\definecolor{accent}{HTML}{176B65}
\definecolor{ink}{HTML}{172C35}
\definecolor{muted}{HTML}{4D5B64}
\hypersetup{colorlinks=true,urlcolor=accent,linkcolor=accent}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\renewcommand{\arraystretch}{1.32}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\small\sffamily COT / GHI · 图表精修评审版}
\fancyhead[R]{\small\sffamily 30 OCT 2026}
\fancyfoot[L]{\small\color{muted}真实数据 · 来源明确 · 逐图评阅}
\fancyfoot[R]{\small\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\setlength{\headheight}{14pt}
\setlength{\emergencystretch}{2em}
\begin{document}\color{ink}
{\color{accent}\sffamily\bfseries\fontsize{27}{32}\selectfont COT论文\quad 图表精修第一版}\par
{\sffamily\large 从原文设计逻辑，到可复核的模型与结果表达}\par
\vspace{3mm}
本版先交付可以逐页评阅的结构图、流程图、真实云事件及结果图表。图形来自实际代码和保存预测，没有添加模拟结果。\textbf{它是图表评审版，不是已经完成独立验证的投稿稿。} 湖北同期单帧主结果、同一来源的COT质量及厚度分层仍需补齐；这些位置不借用旧分数填充。

\textbf{表1\quad 从原论文借鉴什么，怎样转化为本研究的图表。}\par
'''
    body+=table(['原文与图表','作者为何需要它','本版采用的表达'],[
      ['Chen et al., Applied Energy (2023), Fig.4 / Table3','解释多通道如何编码、融合；表中保持站点与方法的明确对应','结构展开到算子和张量；精确数值与比较问题分表'],
      ['Papachristopoulou et al., AMT (2024), Fig.2 / 11 / 15','把物理关系、误差、预报时效及云场变化分开回答','绝对误差与配对区间并列；COT输入质量不替代GHI收益'],
      ['Zhang et al., GRL (2026), Fig.1 / 2 / 3','连接网络、逐时效表现与空间演变','网络和递推分开；少量可读关键帧连接连续轨迹'],
      ['用户提供的Baguan-solar结构图','两阶段清楚分工；多源输入与关键融合展开','区分数据流程与网络结构；只画本项目实际运算']
    ],'p{49mm}YY')
    body+=para(r'原论文截图、页码和逐项分析保留在配套阅读册。这里借鉴论证和布局，不复制其数值，也不将跨论文RMSE当作同一测试集排名。')
    body+=r'\vspace{3mm}\textbf{表2\quad 本版图表使用的样本范围。}\par'+'\n'
    body+=table(['内容','有效行 / 日期','用途与边界'],[
      ['湖北四臂预报','15,275 / 22','三站、三seed；验证集；历史COT与新增未来COT分别比较'],
      ['AGRI × COT来源替换','8,111 / 12','三站、三seed；已有共同可替换掩膜；开发期固定头诊断'],
      ['条件信息探针','32,838 / 50','开发日期折外拟合；三份冻结基线来源，不是三次新探针训练'],
      ['真实云事件','2025-08-06','日历规则选例；牛首13个有效GHI区间；解释现象而非总体收益'],
      ['湖南单帧补充','4,440 / 51','两站、三seed；固定权重重评分，沿用原太阳几何处理']
    ],'p{43mm}p{30mm}Y')
    body+=note('行数不是独立天气样本数。不同表的日期、模型和输入范围不同，不能串成一条“精度逐步改善”的曲线。所有含未来真实图像的结果均明确标为离线诊断。')

    body+=section('01','核心网络结构','建议放在方法节：回答“COT在网络中到底怎样起作用”。')
    body+=img('architecture.pdf')
    body+=note('按实际代码展开反演、逐像元编码、空间位置保留、融合及GHI读出。湖南与湖北的宽度和辅助槽分别标明，不能混用。结构已核对；最终湖北同期主结果仍待与权重完整绑定。参数与空间解释见方法附表。')
    parameter_table=r'\textbf{表3\quad 代码实测的规模差异。}\par'+'\n'
    w=json.loads((F/'architecture/CODE_WITNESS.json').read_text())['models']
    parameter_table+=table(['实现','结构区别','参数数'],[
      ['湖南COT反演器',r'$w=32$；$32\to64\to128$',f"{w['hunan_r']['parameters']:,}"],
      ['湖北COT反演器',r'$w=16$；$16\to32\to64$',f"{w['hubei_r']['parameters']:,}"],
      ['湖南AGRI基线 / 加COT',r'读出输入$288/324$；COT为$1\to4$带偏置',f"{w['hunan_A']['parameters']:,} / {w['hunan_AC']['parameters']:,}"],
      ['湖南辅助 / 反演特征',r'$32\to4$；两者相同容量',f"{w['hunan_AP']['parameters']:,}"],
      ['湖北四臂单帧设计',r'COT与云概率双槽$2\to4$，不含偏置',f"{w['hubei_fourarm']['parameters']:,}"]
    ],'p{46mm}Yr')
    parameter_table+=note('参数计数只描述对应网络；不能把两个模型参数数相近等同于整个系统训练资源匹配。')

    body+=section('02','从数据到GHI的完整路径','建议放在方法节开头，与上一页结构图各司其职。')
    body+=img('workflow.pdf')
    body+=note('同期估计与可部署预报分开；预报只能从初始化之前的观测出发。CPP只作监督和参考评价。区域图、反演上下文、站点邻域尺度不同；GHI对应以目标时刻结束的15分钟平均。未来真实AGRI仅用于另列的离线诊断。')
    body+=r'\clearpage\noindent{\sffamily\bfseries\Large 方法附表：实现差异与共同控制}\par\vspace{4mm}'+'\n'
    body+=parameter_table
    body+=para(r'\textbf{时间与输入边界。} 同期估计可用目标时刻真实AGRI；预报先从已知历史预测未来AGRI，再经冻结反演器得到未来COT。GHI标签是以目标时刻结束的15分钟平均；像元时刻配准不等于瞬时像元与地面平均实现严格辐射闭合。')
    body+=para(r'\textbf{不能省略的对照差异。} 湖南的AGRI辅助特征与冻结反演器特征控制彼此等容量，但比显式COT臂多124个可训练参数。湖北四臂使用相同辅助槽和参数形状，通过中性零槽定义对照。两种设计都需要保留各自身份，不能合并成“全部模型严格等容量”。')
    body+=para(r'\textbf{空间结构怎样进入网络。} 单帧头的$1\times1$卷积逐像元编码，不在这一步混合相邻位置。随后拼接并保留通道、行、列顺序展平，才由全连接读出联合九个位置。因此它保留空间位置，却没有显式建模运动或历史轨迹；不能仅因输入整张小图就写成已学到云输运。')
    body+=para(r'\textbf{反演上下文与GHI邻域。} R在$16\times16$范围获取上下文，GHI读取同站点$3\times3$局地特征。加入COT的收益可能同时受云产品监督和更大上下文影响。要主张光学厚度本身的独特贡献，还应有相同视场与训练资源的表示对照。')

    body+=section('03','历史信息和未来信息分别有没有用','建议放预报应用节：先给真实GHI，再解释COT。')
    body+=img('forecast.pdf')
    body+=r'\textbf{表4\quad 湖北三站逐站GHI RMSE（W m$^{-2}$）。}\par'+'\n'
    rows=[]
    for st in ['Niushou','Wangzhai','Mizhuang','Equal-station mean']:
        rr={r['arm']:r for r in frows if r['station']==st}
        rows.append([names[st]]+[f"{float(rr[a]['rmse_wm2']):.3f}" for a in ['G0','G1','G2','G3']])
    body+=table(['站点','无COT','历史COT','历史+控制修正','历史+预测COT'],rows,'Yrrrr')
    body+=note('三seed先分别计算RMSE，再按站点和seed等权；不是把三seed预测集成后评分。后两臂具有相同参数量，并共用已训练的历史COT基底。')
    body+=para(r'\textbf{关键判断。} 历史COT点差为$-1.925$ W m$^{-2}$，日期块95\%区间$[-4.734,+0.670]$；新增未来COT相对同容量控制为$+0.026$，区间$[-0.003,+0.059]$。两者均没有给出可靠的正向增益。')

    body+=section('04','真实云图怎样连接COT与辐照度','机制案例；不能将个例改善替代总体预报收益。')
    body+=img('event.pdf',174)
    body+=note('2025年8月6日牛首，初始化12:00 BJT。真实与预测AGRI各紧邻其反演COT；统一C03/C02/C01、反射率0–1平方根显示，保留16×16原像元，标出站点与3×3输入。下方COT和15分钟GHI共享时间轴；缺测断线。灰线已含历史COT，紫线用真实未来图COT，仅作离线诊断。三份冻结基线来源的曲线取展示均值；R-COT超过100仍保留。本例按日历有效性选择，非按收益挑选。')

    body+=section('05','区分底图误差与COT映射的响应','同一固定模型的输入替换；独立于上一页个例和正式验证主表。')
    body+=img('replacement.pdf')
    body+=r'\textbf{表5\quad 固定头的MSE敏感性分解（W$^2$ m$^{-4}$）。}\par'+'\n'
    termnames={'D_AGRI':r'$E_{FO}-E_{OO}$：图像路径','D_COT':r'$E_{OF}-E_{OO}$：COT路径','interaction':r'$E_{FF}-E_{FO}-E_{OF}+E_{OO}$：交互','total':r'$E_{FF}-E_{OO}$：总变化'}
    rows=[]
    for k,v in source['MSE_decomposition'].items():rows.append([termnames[k],f"{v['point']:+.1f}",f"[{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}]"])
    body+=table(['项','点估计','日期块95\%区间'],rows,'Yrr')
    body+=note('O为真实图来源、F为预测图来源。前两项与交互之和等于总变化，区间本身不能相加。这里是固定映射对输入替换的响应，不是独立物理误差预算。')
    body+=para(r'\textbf{可成立的解释。} 换成真实AGRI明显降低误差，但在同源真实输入下额外COT仍未改善这个固定头。因此底图信息损失是重要线索，不能据此宣称唯一原因，也不能把映射适应问题排除。')

    body+=section('06','真实图COT有信息，预测源保留了多少','条件探针使用开发日期折外拟合，不能与正式验证绝对RMSE混排。')
    body+=img('probe.pdf')
    body+=para(r'\textbf{先看参照。} 已有历史COT预报基线为118.799 W m$^{-2}$；上下文探针为124.850，加入预测COT为124.307，两者都比已有基线差。只有使用真实未来图COT的离线探针达到99.569。')
    body+=para(r'\textbf{再看来源替换。} 对已经拟合的探针更换COT来源，可检验其响应是否随来源变化。图中的虚线列依赖未来观测。较好的离线分数说明这种来源在当前诊断中可被利用，却不证明已经有可部署的改善方案。')
    body+=note('32,838条共同记录、50个开发目标日期；三个站点与三份冻结基线来源等权。探针为确定性岭回归，不能把这三份来源写成三次独立探针种子。误差线为5,000次目标日期块配对bootstrap的95%区间。底部纵轴是GHI误差差，不是COT物理精度。')

    body+=section('07','区域差异与同期证据的适用范围','两张材料各说明一个问题：同日三站差异，以及旧处理下的固定权重结果。')
    body+=img('stations.pdf',170)
    body+=note('同一事件三站分别有13、14、11个有效GHI区间；各排共用尺度，缺测不连线。真实未来COT仍仅作离线诊断，不把三站当作独立新天气事件。')
    body+=r'\vspace{3mm}\textbf{表6\quad 湖南单帧固定权重重评分（补充诊断；W m$^{-2}$）。}\par'+'\n'
    rows=[]
    for st in ['sili','zhujia','Equal-station mean']:
        rr={r['arm']:r for r in hrows if r['station']==st}
        rows.append([names[st]]+[f"{float(rr[a]['rmse_wm2']):.3f}" for a in ['A','AC','AP','AL']])
    body+=table(['站点','AGRI','AGRI+COT','辅助特征','反演特征'],rows,'Yrrrr')
    body+=para(r'\textbf{如何读。} 两站总体108.181$\to$103.995，点降幅3.870\%；反演隐特征总体103.456，说明这组结果不能证明显式COT是唯一有效表示。')
    body+=note('此表保留原权重和原太阳几何处理；剔除已确认无效站点日后，在4,440行、51日期上统一重评分。模型曾按剔除前验证集选模，不是重训、独立测试或修正太阳几何后结果，不能填充湖北同期主结果缺口。')

    body+=section('08','进入正文前，还要补齐什么','保留科学问题和图表位置；不以美化代替缺失证据。')
    body+=r'\textbf{表7\quad 图表与论证的对应关系。}\par'+'\n'
    body+=table(['问题','本版已完成','仍需完成'],[
      ['方法如何复现','数据流程、实际网络层次、空间尺度、冻结与监督边界','将最终同期主结果的权重、配置与图中的区域实现一一绑定'],
      ['COT能否改善同期GHI','湖南固定权重补充表；可复核的代码与方法图','湖北同一修正来源的同期主比较、容量/表示控制'],
      ['哪些云况受益','真实云事件与三站响应展示','同一最终来源的CPP质量、厚度分层及另一互补事件'],
      ['未来COT是否有效','湖北四臂、逐时效、日期块区间','这些是验证证据；独立确认须另按冻结方案进行'],
      ['为什么未来收益有限','固定头替换与折外探针分开展示','不能给唯一归因或将不同队列的差值画成收益衰减量']
    ],'p{33mm}YY')
    body+=para(r'\textbf{建议阅读顺序。} 主文先用完整路径和网络结构解释方法；同期主结果和厚度分层补齐后，连接真实云事件；最后用预报表与逐时效图检验应用边界。固定头来源替换可放预报节或讨论；条件探针和三站完整曲线适合补充材料。')
    body+=para(r'\textbf{统一视觉规则。} 观测用黑色；已有基线用灰色；历史COT用青绿色；预测COT用橙色；未来真实来源用紫色并标离线。差值始终为候选减参照，负值改善。空间图保留真实像元，轨迹有缺测就断线，不为平滑和锐化改变数据。')
    body+=r'\vspace{3mm}\textbf{原文与工具来源}\par'+'\n'
    body+=r'''{\small
Chen et al. (2023), \textit{Applied Energy}, 352, 121979. \href{https://doi.org/10.1016/j.apenergy.2023.121979}{DOI}\par
Papachristopoulou et al. (2024), \textit{AMT}, 17, 1851--1877. \href{https://amt.copernicus.org/articles/17/1851/2024/}{全文}\par
Zhang et al. (2026), \textit{GRL}, 53, e2025GL119553. \href{https://doi.org/10.1029/2025GL119553}{DOI}\par
Baguan-solar, 用户结构截图参考。\href{https://arxiv.org/abs/2603.14845}{arXiv:2603.14845}\par
科学可视化技能用于设计与审图；数值由保存预测重算。Kassis et al. (2026), \textit{Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents}. \href{https://arxiv.org/abs/2609.00065}{arXiv:2609.00065}。技能不是科学结论的证据。\par}
\end{document}
'''
    (P/'main.tex').write_text(body,encoding='utf-8')
    (P/'BUILD_MANIFEST.json').write_text(json.dumps({'scope':'first figure/table review edition; not final manuscript','figures':{name:{'source':str(src.relative_to(PAPER)),'sha256':hashlib.sha256(src.read_bytes()).hexdigest()} for name,src in image_sources.items()},'numeric_sources':{str(p.relative_to(PAPER)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [E/'hubei_forecast_station_arm_metrics.csv',E/'hunan_singleframe_station_arm_metrics.csv',F/'source_replacement/summary.json']},'tex_sha256':hashlib.sha256((P/'main.tex').read_bytes()).hexdigest()},indent=2),encoding='utf-8')
    print('Built first figure/table review edition source')

if __name__=='__main__':main()
