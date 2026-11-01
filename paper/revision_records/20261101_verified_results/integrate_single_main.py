"""One-time integration into the existing main manuscript, not a new edition."""
from pathlib import Path
import re,json,shutil,hashlib
P=Path(__file__).resolve().parents[2];R=P.parent;D=Path(__file__).resolve().parent
V=P/'visual_preview_v3_20261030'
e=(P/'main.tex').read_text(encoding='utf-8');z=(P/'main_zh.md').read_text(encoding='utf-8')
v=(V/'main.tex').read_text(encoding='utf-8');vz=(V/'main_zh.md').read_text(encoding='utf-8')
for n in ['main.tex','main_zh.md']:
 q=D/'before_unification'/n;q.parent.mkdir(exist_ok=True)
 if not q.exists():shutil.copy2(P/n,q)
A=P/'figures/integrated_20261101';A.mkdir(exist_ok=True)
for q in (V/'images').iterdir():
 if q.is_file():shutil.copy2(q,A/q.name)
def portion(s,a,b):return s[s.index(a):s.index(b,s.index(a))]
def fig(s,label):
 i=s.index(r'\label{'+label+'}');a=s.rfind(r'\begin{figure}',0,i);b=s.index(r'\end{figure}',i)+len(r'\end{figure}')
 return s[a:b]
def table(s,label):
 i=s.index(r'\label{'+label+'}');a=s.rfind(r'\begin{table}',0,i);b=s.index(r'\end{table}',i)+len(r'\end{table}')
 return s[a:b]
def zfig(s,path):
 i=s.index('![',s.index(path)-100) if False else s.index(path)
 a=s.rfind('![',0,i);b=s.index('**',i);b=s.index('**',b+2)+2
 return s[a:b]
def setsec(s,start,end,new):return s[:s.index(start)]+new+'\n\n'+s[s.index(end,s.index(start)):]
def paths(s):
 s=s.replace('{images/','{figures/integrated_20261101/')
 s=s.replace('F:/CODE_Classify/辐照度预报/solar-cot-ifs/paper/visual_preview_v3_20261030/images/','F:/CODE_Classify/辐照度预报/solar-cot-ifs/paper/figures/integrated_20261101/')
 return s

# Retain the latest 30-reference prose, but recover the visual layout and methods.
preamble=v[:v.index(r'\begin{document}')]
e=preamble+e[e.index(r'\begin{document}'):]
e=e.replace(r'\section{Study area and data}',table(v,'tab:related')+'\n\n'+r'\section{Study area and data}',1)
related=portion(vz,'**表1：','## 2 研究区域与数据') if '**表1：' in vz else ''
z=z.replace('## 2 研究区域与数据',related+'## 2 研究区域与数据',1)
# Station coordinates are the actual registration values, not the empty map.
e=e.replace(table(e,'tab:data'),table(v,'tab:data'))
za=z.index('**表1：',z.index('## 2 研究区域与数据'))
zb=z.index('### 2.2',za)
va=vz.index('**表2：');vb=vz.index('### 2.2',va)
z=z[:za]+vz[va:vb]+z[zb:]
for label in ['fig:regions','fig:retrieval']:
 # Current manuscript has the macro invocation, visual draft has an image.
 m=re.search(r'\\figslot\{[^\n]+\}\{[^\n]+\}\{'+re.escape(label)+r'\}',e)
 assert m,label;e=e[:m.start()]+fig(v,label)+e[m.end():]
z=z.replace('**图1：研究区域与AGRI、站点GHI及CPP参考覆盖。**','**图1：研究区域与观测覆盖。空面板为明确标注的数据覆盖模板，并非实测地图或可用率。**')
# Original generic Chinese map/retrieval captions replaced below by source captions.
for number,name in [('1','fig01_regions'),('3','fig04_retrieval')]:
 key='**图'+number+'：';a=z.index(key);b=z.index('\n',a)
 z=z[:a]+zfig(vz,'images/'+name)+z[b:]
e=e.replace('feature widths 32, 64, and 128,','feature widths 32, 64, and 128 in Hunan and 16, 32, and 64 in Hubei,')
z=z.replace('特征宽度为32、64、128','湖南特征宽度为32、64、128，湖北为16、32、64')
a=e.index('The encoded local features');b=e.index('The models minimize',a)
pa=portion(v,'In the single-frame Hunan implementation','\n\n\\begin{figure}')
e=e[:a]+pa+'\n\n'+fig(v,'fig:architecture')+'\n\n'+e[b:]
a=z.index('编码后的局地特征');b=z.index('模型优化缩放后的',a)
za=vz.index('湖南单帧',vz.index('### 3.2')) if '湖南单帧' in vz else -1
assert za>=0
zb=vz.index('模型优化缩放后的',za)
z=z[:a]+vz[za:zb]+z[b:]
# Keep the latest observed-trained forecast replay and additionally document the
# separate Hubei forecast-trained residual experiment whose figures are reused.
fourmethod=portion(v,'The forecast-trained comparison separates','\n\\begin{figure}')
pos=e.index(r'\section{Experimental design}')
start=e.index(r'\figslot{Observed AGRI')
e=e[:start]+r'\noindent The separate Hubei forecast-trained comparison is described in Supplementary material S5. Its input flow is shown in Fig.~\ref{fig:framework}; it is distinct from replaying the Hunan three-frame models.'+'\n\n'+fig(v,'fig:framework')+'\n\n'+e[pos:]
zstart=z.index('**图2：COT辅助反演与预报流程。**');zend=z.index('## 4 实验设计',zstart)
z=z[:zstart]+'湖北另行训练的预报对照见补充材料S5，其输入流见图3，与湖南三帧模型直接用于预测图的评价不同。\n\n'+zfig(vz,'images/fig03_workflow.png')+'\n\n'+z[zend:]

# Primary synchronous estimates await results on the independently verified
# geometry; the reproducible original-geometry estimates remain supplementary.
old_est=portion(e,r'\subsection{Effect of COT on irradiance estimation}',r'\subsection{Dependence on optical thickness')
old_est_z=portion(z,'### 5.2 COT对辐照度反演的影响','### 5.3 光学厚度及云况依赖')
old_bins=portion(e,r'\subsection{Dependence on optical thickness',r'\subsection{Cloud events')
old_bins_z=portion(z,'### 5.3 光学厚度及云况依赖','### 5.4 云事件')
new_est=r'''\subsection{Effect of COT on irradiance estimation}
\pending{Insert the matched Hubei and Hunan single-image comparisons whose retrieval and irradiance inputs use independently verified solar geometry, with station-wise errors and paired uncertainty.}
The regional comparisons assess the contribution of COT at the same target intervals, separately from its forecasting application. Original-geometry representation comparisons are retained in Supplementary material S4. They quantify behaviour under that preprocessing and are not substituted for the primary estimates.
'''+fig(v,'fig:gain')
new_est_z='''### 5.2 COT对辐照度反演的影响

【待填：采用独立核验太阳几何的湖北与湖南单帧配对结果，包括逐站误差和配对区间。】

区域比较在相同目标区间评价COT的增量，与预报应用分别报告。原几何条件下的表示对照保留在补充材料S4，量化该预处理下的模型行为，不替代主结果。

'''+zfig(vz,'images/fig05_gain.png')
e=setsec(e,r'\subsection{Effect of COT on irradiance estimation}',r'\subsection{Dependence on optical thickness',new_est)
z=setsec(z,'### 5.2 COT对辐照度反演的影响','### 5.3 光学厚度及云况依赖',new_est_z)
new_bins=r'''\subsection{Dependence on optical thickness and cloud conditions}
\pending{Insert cloud-thickness-stratified errors on the same verified-geometry model lineage as the primary estimation comparison. Report common records, date blocks and paired uncertainty for each stratum.}
Original-geometry Hubei results show condition-dependent responses, including adverse near-zero-COT behaviour at one station (Supplementary material S4). They motivate the stratified comparison but do not establish its outcome for a different preprocessing configuration.
'''+fig(v,'fig:thickness')
new_bins_z='''### 5.3 光学厚度及云况依赖

【待填：与同期主比较采用相同已核验几何模型来源的厚度分层结果，逐档给出共同记录、日期块与配对区间。】

原几何湖北结果呈现云况依赖，并有一个站点在近零COT下退化（补充材料S4）。这些结果说明需要分层评价，但不能确定另一预处理配置下的结果。

'''+zfig(vz,'images/fig06_strata.png')
e=setsec(e,r'\subsection{Dependence on optical thickness',r'\subsection{Cloud events',new_bins)
z=setsec(z,'### 5.3 光学厚度及云况依赖','### 5.4 云事件',new_bins_z)
# Recover actual cloud fields and aligned station trajectories, never synthesize.
ev=portion(v,r'\subsection{Cloud events and irradiance response}',r'\subsection{Forecasting gain')
zv=portion(vz,'### 5.4 云事件与辐照度响应','### 5.5 预报增益')
e=setsec(e,r'\subsection{Cloud events and irradiance response}',r'\subsection{Effect of COT on forecasting',ev)
z=setsec(z,'### 5.4 云事件与辐照度响应','### 5.5 COT对预报',zv)
# Latest lead figure retains its own identity, separate from the old four-arm plot.
a=e.index(r'\subsection{Effect of COT on forecasting');b=e.index(r'\section{Discussion}',a)
section=e[a:b].replace('fig:forecast','fig:forecast_latest')
section+=r'\noindent A separate forecast-trained Hubei comparison does not show a detectable extra benefit from future COT beyond its matched historical-COT correction (Supplementary material S5). This differs in region, model training and comparison definition from the Hunan replay and is retained as a boundary on generalization.'+'\n\n'
e=e[:a]+section+e[b:]
a=z.index('### 5.5 COT对预报');b=z.index('## 6 讨论',a)
z=z[:b]+'湖北另行训练的预报对照未显示未来COT在匹配历史COT修正之上的可辨别增量（补充材料S5）。其区域、模型训练及对照定义均与湖南直接回放不同，作为推广范围的限制保留。\n\n'+z[b:]
# Avoid unverified synchronous headline numbers; keep the completed forecasting result.
ap=portion(e,r'\begin{abstract}',r'\end{abstract}')
ap=re.sub(r'On the regional validation sets,.*?With independently trained', 'Original-geometry estimation comparisons and their cloud-condition dependence are reported separately from the verified-geometry forecasting application. With independently trained',ap,flags=re.S)
ap=ap.replace('On the regional validation sets,','On the regional validation sets,')
e=setsec(e,r'\begin{abstract}',r'\end{abstract}',ap)
a=z.index('在区域验证集上');b=z.index('采用湖南独立训练',a)
z=z[:a]+'原几何条件下的同期对照及云况依赖与已核验几何的预报应用分别报告。'+z[b:]
e=e.replace('The Hunan retrieval-feature contrast suggests that cloud-product supervision contributes a useful representation, while the Hubei thickness analysis identifies a substantial correction of positive bias in the sampled thick-cloud conditions. These findings support a practical value of COT without demonstrating that a scalar optical-thickness field contains uniquely useful information beyond all alternative cloud-supervised features.', 'The original-geometry comparisons in Supplementary material S4 suggest a contribution from cloud-product supervision and motivate evaluating thickness-dependent bias. Their absolute gains do not establish the corresponding effects for a model using different solar geometry. '+r'\pending{Integrate the primary verified-geometry estimation and representation-control results here.}')
z=z.replace('湖南反演特征对照提示云产品监督可形成有用表示；湖北厚度分析则显示，在已采样厚云条件下，COT明显纠正了正偏差。这些结果支持COT的实际价值，但不说明标量光学厚度场包含超出所有其他云监督特征的独有信息。','补充材料S4的原几何对照提示云产品监督可能有用，并促使我们评价厚度依赖偏差；其绝对增益不能确定另一太阳几何模型下的对应作用。【待填：结合已核验几何的同期主结果与表示对照完成解释。】')
e=re.sub(r'On the evaluated single-image validation sets,.*?With Hunan DyDiff predictions',r'\\pending{State the primary regional estimation gains and cloud-condition dependence after completing their verified-geometry comparison.} With Hunan DyDiff predictions',e,flags=re.S)
z=re.sub('在所评价的单帧验证集上，.*?湖南DyDiff预测云图','【待填：完成已核验几何的同期比较后，填写分区域增益及云况依赖。】湖南DyDiff预测云图',z,flags=re.S)

# Preserve all genuine material from the 18-page visual review in supplements.
supp=portion(v,r'\section*{Supplementary material}',r'\end{document}')
suppz=vz[vz.index('## 补充材料'):]
e=e.replace(r'\end{document}',supp+'\n'+r'\end{document}')
z+='\n\n'+suppz

# S4 is deliberately a labelled historical-preprocessing comparison, including
# both positive and adverse effects. No source image is silently promoted.
legacy=r'''\clearpage
\subsection*{S4. Estimation comparisons under the original solar-geometry preprocessing}
These comparisons use stored models fitted with the original solar-geometry preprocessing. The Hubei CPP supervision is correctly matched in UTC, but both its retrieval inputs and GHI geometry retain that original preprocessing. The Hunan single-image comparison also retains its original geometry and fixed selected weights, with a common station-day quality exclusion. These validation estimates are reproducible configuration-specific results, not the primary verified-geometry estimation scores. The thickness-stratified Hubei results have the same provenance as its overall comparison.
'''+old_est.replace(r'\subsection{Effect of COT on irradiance estimation}','').replace('fig:gain','fig:legacy_gain').replace('tab:controls','tab:legacy_controls')+old_bins.replace(r'\subsection{Dependence on optical thickness and cloud conditions}','').replace('fig:thickness','fig:legacy_thickness')
legacy=legacy.replace('Adding COT improves single-image GHI estimation in both regional validation sets','Under the original geometry, adding COT lowers single-image GHI errors in both regional validation sets').replace('These independently trained regional comparisons indicate a repeatable benefit for the evaluated configurations, without requiring equal absolute errors or equal percentage gains between regions.','The comparison is conditional on these preprocessing configurations; it does not isolate the effect of correcting solar geometry.')
legacyz='''\n\n### S4 原太阳几何预处理下的同期比较

这些比较使用原太阳几何预处理下拟合并保存的模型。湖北CPP监督已正确匹配UTC，但反演输入与GHI几何仍采用原预处理。湖南单帧比较也保留原几何及已选定权重，统一排除无效站点日。它们是可复现的特定配置验证结果，不是主文已核验几何的同期评分；湖北厚度分层与其总体比较同源。

'''+old_est_z.replace('### 5.2 COT对辐照度反演的影响','')+old_bins_z.replace('### 5.3 光学厚度及云况依赖','')
legacyz=legacyz.replace('两个区域验证集的单帧GHI反演均因加入COT而改善','原几何条件下，两个区域验证集的单帧GHI误差均因加入COT而降低').replace('两区域独立训练的比较说明，这些配置下的收益可在区域间重复，而绝对误差和相对降幅不必相同。','该比较以这些预处理配置为条件，不能分离太阳几何修正的作用。')
# Old main captions are regenerated with supplementary numbering below.
oldfore=portion(v,r'\subsection{Forecasting gain and propagation of image errors}',r'\section{Discussion}')
oldfore=oldfore.replace(r'\subsection{Forecasting gain and propagation of image errors}',r'\subsection*{S5. Hubei forecast-trained comparisons and controlled input substitutions}')
oldfore=fourmethod+'\n\n'+oldfore
oldfore=oldfore.replace(fourmethod+'\n\n'+r'\subsection*{S5.',r'\subsection*{S5.',1) if False else oldfore
heading=r'\subsection*{S5. Hubei forecast-trained comparisons and controlled input substitutions}'
oldfore=oldfore.replace(heading,'');oldfore=heading+'\n'+fourmethod+'\n'+oldfore[len(fourmethod):]
oldforez=portion(vz,'### 5.5 预报增益与底图误差传递','## 6 讨论').replace('### 5.5 预报增益与底图误差传递','### S5 湖北预报专训及固定输入替换')
zm=portion(vz,'预报专训比较将历史COT', '\n\n') if '预报专训比较将历史COT' in vz else ''
# Fully specified forecast training is retained in supplementary Table S2.
oldforez=oldforez.replace('### S5 湖北预报专训及固定输入替换','### S5 湖北预报专训及固定输入替换\n\n四臂输入及训练预算见补充表S2，与湖南同期训练三帧模型的直接回放分别报告。',1)
e=e.replace(r'\end{document}',legacy+'\n\n'+oldfore+'\n\n'+r'\end{document}')
z+='\n'+legacyz+'\n'+oldforez
e=paths(e);z=paths(z)
# Repair references referring to old figure numbers by using labels in LaTeX.
# Chinese numbering is assigned from final LaTeX float order by image basename.
def renumber(s):
 section='';mainn=0;suppn=0;out={}
 for m in re.finditer(r'\\section\*\{Supplementary material\}|\\begin\{figure\}.*?\\end\{figure\}',s,re.S):
  block=m.group()
  if 'section*' in block:section='S';continue
  lab=re.search(r'\\label\{([^}]+)',block);img=re.search(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)',block)
  if section:suppn+=1;num='S'+str(suppn)
  else:mainn+=1;num=str(mainn)
  if lab:out[lab.group(1)]=(num,Path(img.group(1)).name if img else '')
 return out
fm=renumber(e)
for lab,(num,name) in fm.items():
 # Associate each Markdown caption with its actual adjacent image.
 if not name:continue
 png=name.rsplit('.',1)[0]+'.png'
 for match in list(re.finditer(r'!\[[^\]]*\]\([^\n]*'+re.escape(png)+r'\)',z))[::-1]:
  following=z[match.end():]
  cap=re.match(r'\s*\*\*图[^：]*：',following)
  if cap:
   a=match.end()+cap.start();b=match.end()+cap.end()
   lead=following[:cap.end()];lead=re.sub(r'图[^：]*：','图'+num+'：',lead)
   z=z[:match.end()]+lead+z[b:]
  else:
   # Latest-script captions precede their images.
   a=z.rfind('**图',0,match.start());b=z.find('：',a)
   if a>=0 and z[a:match.start()].count('**')<=2:z=z[:a]+'**图'+num+z[b:]
  z=z[:match.start()]+re.sub(r'!\[[^\]]*\]',f'![图{num}]',z[match.start():match.end()])+z[match.end():]
# Explicit main prose cross-references after adding the architecture figure.
z=z.replace('见图7。','见图8。') if fm.get('fig:forecast_latest',('',''))[0]=='8' else z
e=e.replace('The Hunan auxiliary-image and retrieval-feature variants match one another but each contains 124 more trainable head parameters than the explicit-COT variant.','The original single-image Hunan auxiliary-image and retrieval-feature variants match one another but each contains 124 more trainable head parameters than the explicit-COT variant. These counts do not describe the separate three-frame forecasting heads.')
z=z.replace('湖南额外图像特征与反演隐特征头彼此等容量，但各比显式 COT 头多124个可训练参数。','原单帧湖南额外图像特征与反演隐特征头彼此等容量，但各比显式 COT 头多124个可训练参数。这些计数不代表另行评价的三帧预报头。')
# Keep all bibliographic sources from the latest main, never the preview's old 8.
assert e.count(r'\bibitem{')==30
assert e.count(r'\begin{document}')==1 and e.count(r'\end{document}')==1
assert e.count(r'\label{fig:forecast}')==1 and e.count(r'\label{fig:forecast_latest}')==1
assert '10.54' not in portion(e,r'\begin{abstract}',r'\end{abstract}')
assert '10.54' not in portion(e,r'\section{Results}',r'\section{Discussion}')
for n,s in [('main.tex',e),('main_zh.md',z)]: (P/n).write_text(s,encoding='utf-8')
(D/'UNIFIED_FIGURE_MAP.json').write_text(json.dumps(fm,ensure_ascii=False,indent=2),encoding='utf-8')
print('Integrated existing main; figure map:',fm)
