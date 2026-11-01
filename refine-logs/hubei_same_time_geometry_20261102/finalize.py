from pathlib import Path
import json,hashlib,re
import pymupdf
P=Path(__file__).parent;ROOT=P.parents[1];M=ROOT/'paper'
e=(M/'main.tex').read_text(encoding='utf-8').replace(r'W\,m{-2}$',r'W\,m$^{-2}$');(M/'main.tex').write_text(e,encoding='utf-8')
r=json.loads((P/'INDEPENDENT_FP64_RESULTS.json').read_text());pack=json.loads((P/'recovered/pack/COMPLETE.json').read_text())
(P/'EXPERIMENT_TRACKER.md').write_text('''# 状态：已完成并独立复算

2026-11-02。搜索当前本地/服务器实验及关联会话，未发现可直接复用的正确太阳几何、真实三帧湖北同期配对结果。既有单帧真实图训练头保存的是预测图验证，最新刚完成的1.702%是湖南预报，均不当作本项同期结果。

- CPU准备217108.tc6000：已完成，复用修正缓存，11,033帧。
- 首次GPU217109.tc6000：进入训练前cuDNN符号冲突，保留run.log，无模型训练进度损失。
- 重试217110.tc6000：node21两张A100 GPU0/1，已完成；实时C状态时walltime00:02:44，后PBS清除历史号。完成状态另由9份COMPLETE、18组best/last权重与逐样本预测证明。
- 用户要求加卡时已6/9配对完成，剩余不足约1分钟；保持现有训练收尾，未迁移或丢弃进度。
- 真实三帧、正确几何、冻结湖北R；20,035 train、1,905 validation，三站各3seed；每臂30epoch、global batch512，同初始化、样本顺序和更新数。
- RMSE 74.405617→68.629188 W/m²，下降7.763432%；9/9配对改善。日期块95%差区间[-9.650116,-2.312954]，两日块敏感性区间[-10.418694,-1.697954]。米庄单站区间跨零。
- 独立评分INDEPENDENT_FP64_RESULTS.json完全吻合保存预测与主评分。R参考质量和GHI分层见COT_QUALITY_AND_GHI_STRATA.json。
- 权重与预测已回收至recovered/training，代码与参数在本目录；未读test/buffer、未训练预测图GHI头。
- 唯一主稿paper/main.tex及main_zh.md已同步填入结果和真实数据图；原单帧及历史预报训练图保留在补充材料。
''',encoding='utf-8')
(P/'SOURCE_SEARCH.md').write_text('''# 已有结果查找与新任务范围

查阅本地/服务器当前实验树、10月21日太阳几何修复根、10月20日四臂根、10月22日training_input_source根，以及关联会话最近记录。

| 资产 | 实际内容 | 本轮处理 |
|---|---|---|
| hubei_cloud_probability_20261020 | 单帧同期10.54%，正确CPP UTC但错误几何 | 保留历史，不复用为修正后主成绩 |
| hubei_solar_geometry_repair_20261021/GHI_runs_attempt12 | 正确几何的未来预测AGRI头 | 不是同期，不混用 |
| hubei_training_input_source_20261022_v1 | REAL/REAL_COT单帧真实图训练，但保存的验证输入为预测图 | 不冒充真实三帧同期评分 |
| fulltrain_forecast_cot_20261101/observed_trained_heads_evaluation_v1 | 11月2日完成的湖南预测云图接真实图旧头，1.702% | 与湖北同期区分 |

按用户当前真实三帧路线补做，使用既有修正R与已恢复观测，不重训上游。候选标签池25,899行，三帧上下文可用且划分相同后21,940行：20,035 train、1,905 val。3,959行仅因上下文不可用/边界而退出；不按天气、COT值或误差筛选，两臂共用同一行轴。旧10.54%的2,195条单帧验证与当前1,905条三帧验证不同，不能将降幅变化归因于几何修复一个因素。
''',encoding='utf-8')
idx=M/'VERSION_INDEX.md';s=idx.read_text(encoding='utf-8')
s=s.replace('更新：2026-11-01。','更新：2026-11-02。湖北修正几何真实三帧同期实验已完成并填入：RMSE74.406→68.629 W/m²（7.76%）；COT质量、分层和三帧方法图均已更新。',1)
s=s.replace('当前21页，正文8图、补充6图；其中4个正文图为明确标注的结果/覆盖留位模板','当前23页，正文7图、补充8图；仅数据覆盖图仍为明确标注的留位模板')
s=s.replace('主文相应同期数字继续留位。','主文已用11月2日正确来源三帧同期结果填入。')
s=s.replace('最终同期比较、COT质量汇总及数据/训练设置未齐部分仍明确留位。本次不启动训练，不读取test。','湖北同期比较、同一R的COT质量及GHI分层已完成。数据覆盖、部分训练设置与同期连续事件等未齐部分仍明确留位。详细结果见refine-logs/hubei_same_time_geometry_20261102/RESULTS.md；test未读取。')
idx.write_text(s,encoding='utf-8')
for name in ['PAPER_PLAN.md','FIGURE_PLAN.md']:
 f=M/name;s=f.read_text(encoding='utf-8');s='## 2026-11-02 当前主稿进展\n\n遵循真实三帧训练、冻结用于预报的路线。湖北修正几何同期RMSE74.406→68.629（7.76%），20日期1,905条共同验证、三站三seed；正确来源COT质量与分层已回填，中英文同步。当前正文7图、补充8图，唯一入口见VERSION_INDEX.md；下方为历史设计/阶段记录，旧“同期待填”不代表当前状态。\n\n'+s;f.write_text(s,encoding='utf-8')
manifest={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in P.glob('*.py')}
manifest.update({str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (P/'recovered/training').rglob('*') if f.is_file() and f.suffix in ('.pt','.npz','.json')})
(P/'REPRODUCTION_FILES_SHA256.json').write_text(json.dumps(manifest,indent=2))
z=(M/'main_zh.md').read_text(encoding='utf-8');body=e[e.index(r'\begin{document}'):]
labels=re.findall(r'\\label\{([^}]+)',body);assert len(labels)==len(set(labels));assert e.count(r'\begin{equation}')==11 and len(re.findall(r'\\tag\{\d+\}',z))==11
assert e.count(r'\bibitem{')==30
assert len(re.findall(r'\\includegraphics',body))==15
caps=re.findall(r'\*\*图(S?\d+)：',z);assert len(caps)==len(set(caps))==15,caps
assert r'\tau-30:\tau' in e and r'\tau-30:\tau' in z
print('FINAL_CONTENT_CHECK_PASS',len(manifest),'reproduction files')
