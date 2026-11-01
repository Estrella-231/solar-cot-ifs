from pathlib import Path
import re,json,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
P=Path(__file__).parent;ROOT=P.parents[1];M=ROOT/'paper';F=M/'figures/hubei_sync_corrected_20261102'
r=json.loads((P/'INDEPENDENT_FP64_RESULTS.json').read_text());s=json.loads((P/'COT_QUALITY_AND_GHI_STRATA.json').read_text());pack=json.loads((P/'recovered/pack/COMPLETE.json').read_text())
bk=M/'revision_records/20261102_hubei_sync';bk.mkdir(exist_ok=True)
for n in ['main.tex','main_zh.md']:
 if not (bk/(n+'.before')).exists():shutil.copy2(M/n,bk/(n+'.before'))
e=(M/'main.tex').read_text(encoding='utf-8');z=(M/'main_zh.md').read_text(encoding='utf-8')
def extract_fig(txt,label):
 i=txt.index(r'\label{'+label+'}');a=txt.rfind(r'\begin{figure}',0,i);b=txt.index(r'\end{figure}',i)+len(r'\end{figure}');return txt[a:b]
def replace_sec(txt,a,b,new):i=txt.index(a);j=txt.index(b,i);return txt[:i]+new+'\n\n'+txt[j:]
def img(label,name,caption):return r'\begin{figure}[!htbp]\centering'+'\n'+rf'\includegraphics[width=\linewidth]{{figures/hubei_sync_corrected_20261102/{name}.pdf}}'+'\n'+rf'\caption{{{caption}}}\label{{{label}}}'+'\n'+r'\end{figure}'
def zfig(txt,name):
 m=re.search(r'!\[[^\]]*\]\([^\n]*'+re.escape(name)+r'\)',txt);assert m,name
 # Integrated source captions follow images; latest result captions precede images.
 a=m.start();b=txt.index('**',m.end());b=txt.index('**',b+2)+2;return txt[a:b]

# Deterministic architecture: actual 3D/2D kernels, dimensions and readout.
fig,ax=plt.subplots(figsize=(7.1,3.7),layout='constrained');ax.set(xlim=(0,12),ylim=(0,6));ax.axis('off')
def box(x,y,w,h,t,c):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',edgecolor='#47627a',facecolor=c,lw=.9));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=7)
def arrow(x,y,u,v):ax.add_patch(FancyArrowPatch((x,y),(u,v),arrowstyle='-|>',mutation_scale=9,color='#47627a',lw=.9))
box(.1,3.8,2.2,1.5,'Observed AGRI + geometry\nT−30, T−15, T\n16 × 16 retrieval context','#e8f1f8')
box(2.8,4.05,2.1,1.0,'Frozen COT retriever R\nEach frame independently','#e6f2ea');arrow(2.4,4.55,2.7,4.55)
box(5.3,3.8,2.6,1.5,'Aligned station 3 × 3 fields\n13 AGRI + 2 geometry\n1 log-COT + 1 neutral slot\n[3, 17, 3, 3]','#e8f1f8');arrow(5,4.55,5.2,4.55);arrow(1.2,3.7,5.2,4.0)
box(8.5,3.8,3.1,1.5,'Temporal Conv3D: 17 → 32\nKernel 3 × 3 × 3\nSiLU; temporal dimension → 1','#eeeaf6');arrow(8,4.55,8.4,4.55)
box(8.5,1.7,3.1,1.15,'Spatial Conv2D: 32 → 32\nKernel 3 × 3; SiLU\n32 × 3 × 3','#eeeaf6');arrow(10.05,3.7,10.05,2.95)
box(5.3,1.7,2.6,1.15,'Flatten 288 → 64 → 1\nSiLU then softplus\nClear-sky index','#fff1e0');arrow(8.4,2.27,8,2.27)
box(2.1,1.7,2.6,1.15,'× interval clear-sky GHI\n15-min mean GHI\nTarget interval ends at T','#fff1e0');arrow(5.2,2.27,4.8,2.27)
ax.text(.1,.65,'Paired comparison: same observed frames, geometry, weights at initialization and training budget.\nThe no-COT arm sets only the standardized COT input to zero. R remains frozen.',fontsize=7,va='center')
for ext in ['pdf','png']:fig.savefig(F/f'three_frame_method.{ext}',dpi=240)
plt.close(fig)

# The single-frame and historical forecast-trained diagrams remain in supplements.
old_arch=extract_fig(e,'fig:architecture');old_work=extract_fig(e,'fig:framework')
za=zfig(z,'fig02_architecture.png');zw=zfig(z,'fig03_workflow.png')
e=e.replace(old_arch,img('fig:architecture','three_frame_method','Three-frame COT-assisted GHI estimation. Observed AGRI at the target time and 15 and 30 minutes earlier is processed by a frozen retriever. Registered station fields are fused by the implemented temporal and spatial convolutions. The paired models have identical trainable capacity; the standardized COT slot is set to zero in the no-COT model. The extra neutral slot is inactive in both models.'))
z=z.replace(za,'![图2](figures/hubei_sync_corrected_20261102/three_frame_method.png)\n\n**图2：真实三帧COT辅助GHI估计。冻结反演器分别处理目标时刻及前15、30分钟的观测AGRI，对齐的站点场经实际时空卷积融合。配对模型的可训练容量相同，无COT模型将标准化COT槽置零；额外中性槽在两组均不活动。**')
e=e.replace(old_work,'');z=z.replace(zw,'')
e=e.replace(r'Its input flow is shown in Fig.~\ref{fig:framework}; it is distinct from replaying the Hunan three-frame models.',r'Its historical input flow is retained in Fig.~\ref{fig:framework} and differs from the current fixed observed-trained three-frame application.')
z=z.replace('其输入流见图3，与湖南三帧模型直接用于预测图的评价不同。','历史输入流保留在补充图中，与当前真实图训练三帧模型的固定应用不同。')
e=e.replace('Models trained specifically on forecast imagery are evaluated separately.','Previously completed forecast-trained models are retained only as historical controls in Supplementary material S5; the current forecasting route keeps the observed-trained models fixed.')
z=z.replace('专门使用预测云图训练的模型另行评价。','此前已完成的预测图训练模型仅作为补充材料S5的历史对照；当前预报路线固定真实图训练的模型。')
a=e.index('In the single-frame Hunan implementation');b=e.index(r'\begin{figure}',a)
enmethod=r'''The primary estimator uses the ordered three-frame sequence at $\tau-30$, $\tau-15$ and $\tau$ minutes. At each time, the 13 AGRI channels, two geometric channels and standardized log-COT are registered on the same station-centred $3\times3$ grid. A neutral extra slot gives the implemented tensor shape $[3,17,3,3]$. A temporal convolution maps 17 channels to 32 with a $3\times3\times3$ kernel; a $3\times3$ spatial convolution retains 32 channels, followed by ordered flattening and a $288\to64\to1$ readout. SiLU activations precede the final softplus. Both paired models have 42,529 trainable parameters and receive identical initialization; only the standardized COT slot is set to zero for the baseline. The frame order and local spatial positions are retained, without imposing a cloud-motion trajectory. Original single-frame designs are described in Supplementary material S4.

'''
e=e[:a]+enmethod+e[b:]
a=z.index('湖南单帧实现中');b=z.index('![图2]',a)
z=z[:a]+'''主估计器使用目标前30分钟、前15分钟及目标时刻的有序三帧。每帧的13个AGRI、2个几何通道与标准化对数COT在同一站点中心3×3网格配准；额外中性槽使实际张量为[3,17,3,3]。时间卷积以3×3×3核将17通道映射至32通道，随后3×3空间卷积保留32通道，有序展平后经288→64→1读出。中间采用SiLU，最终为softplus。两组均有42,529个可训练参数，初始权重相同；基线仅将标准化COT槽置零。帧次序与局地位置被保留，但并未强加云运动轨迹。原单帧设计见补充材料S4。

'''+z[b:]
e=e.replace('H_A(X_{s,\\tau},g_{s,\\tau})','H_A(X_{s,\\tau-30:\\tau},g_{s,\\tau-30:\\tau})').replace('H_{AC}(X_{s,\\tau},g_{s,\\tau},Z_{s,\\tau})','H_{AC}(X_{s,\\tau-30:\\tau},g_{s,\\tau-30:\\tau},Z_{s,\\tau-30:\\tau})')
z=z.replace('H_A(X_{s,\\tau},g_{s,\\tau})','H_A(X_{s,\\tau-30:\\tau},g_{s,\\tau-30:\\tau})').replace('H_{AC}(X_{s,\\tau},g_{s,\\tau},Z_{s,\\tau})','H_{AC}(X_{s,\\tau-30:\\tau},g_{s,\\tau-30:\\tau},Z_{s,\\tau-30:\\tau})')
e=e.replace('The learning target and reconstruction are','The learning target and reconstruction are')
e=e.replace('Single-frame COT-assisted GHI estimation','Original single-frame COT-assisted GHI estimation')
e=e.replace('main single-frame comparison and the temporal extension','three-frame comparisons and original single-frame controls')
z=z.replace('单帧主比较与时间扩展分别报告','三帧比较与原单帧对照分别报告')
e=e.replace(r'\pending{widths, parameters, frame count, batch, updates}',r'Three frames; 42,529 parameters; Hubei: 30 epochs, batch 512, AdamW $10^{-3}$, weight decay $10^{-4}$; cosine decay to $10^{-5}$; clip 5; seeds 42--44; select minimum validation RMSE')
z=z.replace('宽度、参数量、帧数、batch、更新数','三帧；42,529参数；湖北30轮、batch512，AdamW学习率10⁻³、weight decay10⁻⁴，余弦衰减至10⁻⁵、clip5；种子42～44，以验证RMSE最小选模')

# Fill retrieval quality using exactly the frozen corrected Hubei retriever.
q=s['quality']
enq=rf'''\subsection{{Evaluation of retrieved COT}}
On {q['pixels']:,} valid $16\times16$ CPP-reference pixels from {q['frames']} validation frames and {q['dates']} target dates, the Hubei retriever has COT RMSE {q['rmse']:.3f}, MAE {q['mae']:.3f}, and bias $+{q['bias']:.3f}$ (Fig.~\ref{{fig:retrieval}}). These dimensionless measures quantify agreement with the CPP retrieval reference, rather than independent cloud truth. GHI evaluation includes all valid station labels, including dates without CPP; the retrieval-reference and station-GHI sample counts therefore differ.
'''+img('fig:retrieval','cot_quality','Hubei corrected-geometry COT retrieval: valid-pixel density against CPP (left) and mean bias within reference thickness classes (right). Scores use all 67,187 valid pixels across 264 frames and 16 target dates. CPP is the retrieval reference. This evaluates the same frozen R used to generate the GHI COT inputs.')
zq=f'''### 5.1 反演COT的评价

湖北反演器在{q['frames']}帧、{q['dates']}个验证目标日期的{q['pixels']:,}个16×16有效CPP参考像元上，COT RMSE为{q['rmse']:.3f}、MAE为{q['mae']:.3f}、偏差为+{q['bias']:.3f}（图3）。这些无量纲指标衡量与CPP检索参考的一致性，并非独立云真值。GHI评价包含所有有效站点标签及无CPP日期，因此参考检索与站点GHI的样本数不同。

![图3](figures/hubei_sync_corrected_20261102/cot_quality.png)

**图3：湖北修正几何COT反演：相对CPP的有效像元密度（左）和参考厚度分层平均偏差（右）。共67,187个有效像元、264帧、16个目标日期；CPP为检索参考，反演器与GHI的COT输入来源相同且固定。**'''
e=replace_sec(e,r'\subsection{Evaluation of retrieved COT}',r'\subsection{Effect of COT on irradiance estimation}',enq)
z=replace_sec(z,'### 5.1 反演COT的评价','### 5.2 COT对辐照度反演的影响',zq)
enres=rf'''\subsection{{Effect of COT on irradiance estimation}}
Adding COT to the observed three-frame Hubei inputs reduces station- and seed-averaged GHI RMSE from {r['A_rmse']:.3f} to {r['AC_rmse']:.3f} W\,m$^{{-2}}$, a {r['gain_pct']:.2f}\% reduction (Fig.~\ref{{fig:gain}}). The paired difference is ${r['delta']:.3f}$ W\,m$^{{-2}}$ (95\% target-date-block interval: $[{r['delta_ci95'][0]:.3f},{r['delta_ci95'][1]:.3f}]$). All nine station--random-seed pairs improve. The comparison contains 1,905 common station--time records across 20 validation dates; no forecast imagery enters these estimates.

Niushou improves from 69.968 to 64.451 W\,m$^{{-2}}$ (7.88\%), Wangzhai from 67.452 to 59.215 W\,m$^{{-2}}$ (12.21\%), and Mizhuang from 85.797 to 82.222 W\,m$^{{-2}}$ (4.17\%). Mizhuang's date-block interval includes zero, so improvement is not resolved equally at every station. The trainable head capacity, initial weights, example order, optimizer and number of updates are matched within each pair. Thus, additional trainable GHI capacity alone does not explain this paired difference, while the comparison does not isolate COT from every alternative cloud-supervised representation.

The Hubei three-frame data contain 20,035 training records. Target labels are retained from the fixed candidate pool when all three observed frames are available in the same partition. For complementary Hunan evidence, the fixed three-frame models evaluated on the 63,480 forecast-associated validation records give 90.318 W\,m$^{{-2}}$ without COT and 89.155 W\,m$^{{-2}}$ with COT (1.29\%); its paired interval includes zero ($[-3.345,0.958]$ W\,m$^{{-2}}$). These overlapping records use the forecast-comparison weighting and are not 63,480 independent station observations. Original single-frame representation controls remain in Supplementary material S4; their gains are not used as a geometry-only ablation.
'''+img('fig:gain','ghi_gain','Observed three-frame Hubei GHI estimation. Left: station-specific and equally averaged RMSE, first computed per random seed. Right: paired COT-minus-baseline RMSE differences with 95% target-date-block intervals from 5,000 draws; negative values favour COT. Counts are 663, 607 and 635 validation station--time records at Niushou, Wangzhai and Mizhuang, respectively.')
enz=f'''### 5.2 COT对辐照度反演的影响

湖北真实三帧输入加入COT后，站点与种子等权GHI RMSE由{r['A_rmse']:.3f}降至{r['AC_rmse']:.3f} W/m²，降低{r['gain_pct']:.2f}%（图4）。配对差为{r['delta']:.3f} W/m²，95%目标日期块区间为[{r['delta_ci95'][0]:.3f},{r['delta_ci95'][1]:.3f}]。9个站点—随机种子配对均改善。评价覆盖20个验证日期的1,905条共同站点—时次记录，不输入预测云图。

牛首由69.968降至64.451 W/m²（7.88%），王寨由67.452降至59.215 W/m²（12.21%），米庄由85.797降至82.222 W/m²（4.17%）。米庄日期块区间包含零，因此各站改善的确定性并不相同。每个配对的GHI可训练容量、初始权重、样本顺序、优化器和更新数相同，增加GHI可训练容量本身不足以解释配对差异，但该比较没有将COT与所有其他云监督表示区分开。

湖北三帧训练共20,035条记录；目标标签来自固定候选池，要求三张真实图均可用且属于同一划分。作为湖南补充，固定三帧模型在63,480条预报关联验证记录上使用真实图时，无COT和有COT RMSE为90.318与89.155 W/m²（降低1.29%），配对区间[−3.345,0.958] W/m²包含零。这些重叠记录沿用预报比较的权重，并非63,480个独立站点观测。原单帧表示对照仍保留在补充材料S4，其增益不能当作仅改变太阳几何的消融。

![图4](figures/hubei_sync_corrected_20261102/ghi_gain.png)

**图4：湖北真实三帧同期GHI估计。左：逐站和等权平均RMSE，先分别计算各随机种子。右：加COT减基线的RMSE差及5,000次目标日期块重采样95%区间；负值有利COT。牛首、王寨、米庄分别663、607、635条验证站点—时次记录。**'''
enres=enres.replace('95%','95\\%').replace('5,000 draws','5,000 draws')
e=replace_sec(e,r'\subsection{Effect of COT on irradiance estimation}',r'\subsection{Dependence on optical thickness',enres)
z=replace_sec(z,'### 5.2 COT对辐照度反演的影响','### 5.3 光学厚度及云况依赖',enz)
sb=s['strata'];gains=', '.join(f"{v['gain_pct']:.2f}\\%" for v in sb)
ent=rf'''\subsection{{Dependence on optical thickness and cloud conditions}}
The reference-covered Hubei subsets show RMSE reductions of {gains} for near-zero, thin, moderate and thick COT, respectively (Fig.~\ref{{fig:thickness}}). These subsets contain 226, 61, 185 and 140 station--time records across 9, 4, 6 and 6 target dates. Thick-cloud RMSE decreases from 51.521 to 33.224 W\,m$^{{-2}}$; its relative reduction has a 95\% interval of $[29.92,44.43]$\%. Thick-cloud mean bias decreases from $+22.408$ to $+0.635$ W\,m$^{{-2}}$, consistent with correcting excessive irradiance under strong optical attenuation.

The thin- and moderate-cloud reduction intervals include zero. Near-zero RMSE decreases, but its positive bias increases from $+2.921$ to $+9.830$ W\,m$^{{-2}}$; moderate-cloud bias changes from $-8.975$ to $-19.541$ W\,m$^{{-2}}$. COT therefore improves error magnitude without consistently improving signed bias in every condition. The 612 reference-covered records are a subset of the 1,905 overall GHI records; missing CPP is not interpreted as clear sky. Limited date coverage restricts generalization of the large thick-cloud reduction.
'''+img('fig:thickness','ghi_thickness','Hubei GHI errors (left) and paired relative reductions (right) conditional on valid station-centre CPP reference COT. Intervals use 5,000 paired target-date draws, retaining fixed station--seed groups. Labels give common station--time records and target dates; records without CPP remain in the overall GHI evaluation.')
zt='''### 5.3 光学厚度及云况依赖

湖北参考覆盖子组中，近零、薄云、中等厚度和厚云的RMSE降幅分别为4.20%、6.09%、6.87%、35.51%（图5），各含226、61、185、140条站点—时次记录，覆盖9、4、6、6个目标日期。厚云RMSE由51.521降至33.224 W/m²，降幅95%区间[29.92,44.43]%；平均偏差由+22.408降至+0.635 W/m²，与纠正强光学衰减条件下过高辐照度相符。

薄云与中等厚度降幅区间包含零。近零RMSE降低，但正偏差由+2.921增至+9.830 W/m²；中等厚度偏差由−8.975变为−19.541 W/m²。因此，误差幅度改善并不意味着所有云况下的有符号偏差均改善。612条参考覆盖记录只是1,905条总体GHI记录的子集，无CPP不被解释为晴空。有限日期覆盖限制了厚云大幅改善的可推广性。

![图5](figures/hubei_sync_corrected_20261102/ghi_thickness.png)

**图5：有效站点中心CPP参考COT分层的湖北GHI误差（左）和配对相对降幅（右）。区间采用5,000次配对目标日期重采样，保持固定站点—种子组；标注为共同站点—时次和目标日期数，无CPP记录仍参与总体GHI评价。**'''
e=replace_sec(e,r'\subsection{Dependence on optical thickness',r'\subsection{Cloud events',ent)
z=replace_sec(z,'### 5.3 光学厚度及云况依赖','### 5.4 云事件',zt)
# Scientific conclusions, not a repair narrative.
e=e.replace('With independently trained Hunan DyDiff imagery and three-frame irradiance models,',r'In Hubei, COT reduces observed three-frame GHI RMSE from 74.41 to 68.63 W\,m$^{-2}$ (7.76\%). The largest reference-covered reduction occurs for thick clouds, with limited date support. With independently trained Hunan DyDiff imagery and three-frame irradiance models,',1)
z=z.replace('采用湖南独立训练的DyDiff云图及三帧辐照度模型，','湖北真实三帧GHI RMSE由74.41降至68.63 W/m²（7.76%）；参考覆盖子组中厚云降幅最大，但日期有限。采用湖南独立训练的DyDiff云图及三帧辐照度模型，',1)
pending=r'\pending{Integrate the primary verified-geometry estimation and representation-control results here.}'
e=e.replace(pending,r'The Hubei three-frame result provides a matched-head-capacity demonstration of a COT-related GHI improvement. The larger thick-cloud correction is associated with lower positive irradiance bias; the smaller Hunan point reduction is not resolved from zero. Cloud-supervised representation is therefore useful under some evaluated conditions, rather than uniformly beneficial across regions and cloud regimes.')
z=z.replace('【待填：结合已核验几何的同期主结果与表示对照完成解释。】','湖北三帧结果在匹配GHI容量条件下显示COT相关改善；厚云的较大修正伴随正偏差降低。湖南较小的点降幅未能与零区分，因此云监督表示在部分已评价条件下有用，并非跨区域、跨云况均匀受益。')
e=e.replace(r'\pending{State the primary regional estimation gains and cloud-condition dependence after completing their verified-geometry comparison.}',r'In Hubei, COT reduces observed three-frame GHI RMSE by 7.76\%, with point improvements at all three stations. The reference-covered thick-cloud subset shows a 35.51\% reduction and substantially lower positive bias, while thin- and moderate-cloud intervals include zero. The complementary Hunan observed-input comparison gives a smaller 1.29\% point reduction with an interval crossing zero.')
z=z.replace('【待填：完成已核验几何的同期比较后，填写分区域增益及云况依赖。】','湖北真实三帧GHI RMSE降低7.76%，三站点估计均改善。参考覆盖厚云子组降低35.51%，正偏差明显减小；薄云与中等厚度区间包含零。湖南真实图补充比较的点降幅为1.29%，区间跨零。')
# Preserve the two old diagrams rather than deleting previous illustrations.
arch=old_arch.replace('fig:architecture','fig:single_frame_original')
work=old_work
i=e.index(r'\subsection*{S4.');j=e.index(r'\subsection*{S5.',i)
e=e[:j]+r'\FloatBarrier'+'\n'+arch+'\n'+e[j:]
i=e.index(r'\subsection*{S5.');e=e[:i]+e[i:].replace('\nThe forecast-trained comparison separates','\n'+work+'\nThe forecast-trained comparison separates',1)
z=z.replace('### S5 湖北预报专训及固定输入替换',za+'\n\n### S5 湖北预报专训及固定输入替换\n\n'+zw,1)
# Update model table with the actually trained main three-frame head.
e=e.replace(r'Hunan COT retriever &',r'Both regions, three-frame GHI & Temporal $17\to32$, spatial $32\to32$; $3\times3$ fields & 288 & 42,529\\'+'\n'+r'Hunan COT retriever &',1)
z=z.replace('| 湖南 COT 反演器 |','| 两区域三帧GHI头 | 时间17→32、空间32→32；3×3场 | 288 | 42,529 |\n| 湖南 COT 反演器 |',1)
# Record scope in data methods, without server paths or historical error details.
e=e.replace(r'\pending{Complete Table 1 with the fixed Hubei partition and the final evaluated cohorts.}',r'The Hubei three-frame comparison retains 20,035 training records and 1,905 validation records on 20 target dates from the fixed candidate labels, requiring all context frames to be present in the same partition. Target labels and clear-sky quantities are identical between paired models.')
z=z.replace('【待填：表2中固定的湖北划分和最终评价样本。】','湖北三帧比较从固定候选标签保留20,035条训练记录及20个目标日期的1,905条验证记录，要求上下文图均可用且属于同一划分；两组目标标签与晴空量完全相同。')
# Neutralize misleading old-table generic single-frame names.
e=e.replace('Hubei synchronous head &','Hubei original single-frame head &')
z=z.replace('| 湖北同期头 |','| 湖北原单帧头 |')
# Reassign all caption numbers from final LaTeX float order, excluding macros.
body=e[e.index(r'\begin{document}'):];n=0;sn=0;sup=False;mapping={}
for m in re.finditer(r'\\section\*\{Supplementary material\}|\\begin\{figure\}.*?\\end\{figure\}',body,re.S):
 b=m.group()
 if b.startswith(r'\section*'):sup=True;continue
 if sup:sn+=1;num='S'+str(sn)
 else:n+=1;num=str(n)
 lab=re.search(r'\\label\{([^}]+)',b).group(1);name=Path(re.search(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)',b).group(1)).stem+'.png';mapping[lab]=(num,name)
for lab,(num,name) in mapping.items():
 m=re.search(r'!\[[^\]]*\]\([^\n]*'+re.escape(name)+r'\)',z);assert m,(lab,name)
 if 'verified_results_20261101' in m.group():a=z.rfind('**图',0,m.start())
 else:a=z.index('**图',m.end())
 b=z.index('：',a);z=z[:a]+'**图'+num+z[b:]
 m=re.search(r'!\[[^\]]*\]\([^\n]*'+re.escape(name)+r'\)',z);z=z[:m.start()]+re.sub(r'!\[[^\]]*\]',f'![图{num}]',m.group())+z[m.end():]
z=z.replace('图7展示牛首','图6展示牛首').replace('图 S2 保留图7事件','图 S2 保留图6事件').replace('分时效差异见图8','分时效差异见图7')
# New figures have all been linked without touching the old visual source copies.
for name,txt in [('main.tex',e),('main_zh.md',z)]: (M/name).write_text(txt,encoding='utf-8')
(bk/'FIGURE_MAP.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2),encoding='utf-8')
print('Updated only existing bilingual main; figures',n,sn)
