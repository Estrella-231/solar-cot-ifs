from pathlib import Path
import json,re,hashlib
import pymupdf
P=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent
(P/'VERSION_INDEX.md').write_text('''# 唯一主稿入口

更新：2026-11-01。18页图文预览与最新结果已整合回原有主稿，不另建正文分支。

- 英文：`paper/main.tex`
- 中文：`paper/main_zh.md`
- 对应PDF：`paper/main.pdf`（当前21页，正文8图、补充6图；其中4个正文图为明确标注的结果/覆盖留位模板）

今后只维护这两个源文件及其编译输出。`draft_v1_20261030/`、`visual_preview_v3_20261030/`、`figure_review_v1_20261030/`及revision/build目录仅作历史审阅、图件来源或备份，不再独立写作；不得按不同目录里的main.pdf修改日期选择主稿。

## 已完成的整合

保留最新30条参考文献和11个公式，恢复18页版的方法结构图、预报输入图、真实AGRI/COT/GHI事件、模型参数表、条件探针、三站轨迹及湖北四臂/输入替换对照。加入湖南独立DyDiff完整验证和逐时效图、固定COT替换诊断。

湖北10.54%的来源冲突已经解决：它修正了CPP的UTC匹配，但保留错误太阳几何。现存服务器文件、25,899条train/val输入几何及保存评分来源均已核对，详见`revision_records/20261101_verified_results/AUDIT_1054_GEOMETRY.md`。10.54%和39.57%厚云结果仅在补充S4作为原几何配置结果报告，主文相应同期数字继续留位。湖南原单帧3.870%重评分亦保留S4，与湖南修正R三帧预报1.737%分开。

最终同期比较、COT质量汇总及数据/训练设置未齐部分仍明确留位。本次不启动训练，不读取test。

## 复核记录

`revision_records/20261101_verified_results/`保存来源审计、整合前备份、单次整合脚本、图件映射及渲染检查。脚本是修订过程档案，不是日常编译命令，后续正文修改后不要直接重跑覆盖。
''',encoding='utf-8')
a=P/'PAPER_PLAN.md';s=a.read_text(encoding='utf-8')
s=s.replace('已将核实的区域同期增益、湖北厚度分层、湖南完整预报及固定COT替换诊断同步填入中英文正文，并生成图4、5、7。剩余内容见 revision_records/20261101_verified_results/STATUS.md。原规划中的占位描述用于追踪尚需整合的内容。','18页图文预览及最新结果已并入唯一主稿main.tex/main_zh.md，见VERSION_INDEX.md。湖北10.54%已确认采用错误太阳几何，连同39.57%厚云结果移补充S4；主文对应同期结果继续留位。湖南完整预报与固定COT替换保留主文。详细审计见revision_records/20261101_verified_results/AUDIT_1054_GEOMETRY.md。下文为科学问题与证据组织规划，历史已填描述不代表来源已合规。')
a.write_text(s,encoding='utf-8')
f=P/'FIGURE_PLAN.md';s=f.read_text(encoding='utf-8')
s=s.replace('更新：2026-10-30。','更新：2026-11-01。当前唯一主稿恢复18页版图文并整合最新结果：正文8图、补充6图，具体编号见revision_records/20261101_verified_results/UNIFIED_FIGURE_MAP.json。原几何同期图移补充S4，主文对应图仍明确留位；不把旧配置分数当作几何修复后结果。以下7图方案为设计目标，实际版式以唯一主稿为准。\n\n原设计：2026-10-30。',1)
f.write_text(s,encoding='utf-8')
f=D/'STATUS.md';s=f.read_text(encoding='utf-8')
f.write_text('## 最新状态：已统一主稿，10.54%几何来源已明确\n\n以VERSION_INDEX.md与AUDIT_1054_GEOMETRY.md为准；下方是分阶段历史。14页回填稿已被21页整合稿替代，原几何10.54%/39.57%/3.87%已撤出摘要。\n\n'+s,encoding='utf-8')
for q in [P/'visual_preview_v3_20261030/README.md',P/'revision_records/20261030_v1/README.md']:
 q.write_text('> 已于2026-11-01整合到上级唯一主稿 main.tex / main_zh.md。本目录为历史只读审阅资料，不再继续写正文。\n\n'+q.read_text(encoding='utf-8'),encoding='utf-8')
e=(P/'main.tex').read_text(encoding='utf-8');z=(P/'main_zh.md').read_text(encoding='utf-8')
body=e[e.index(r'\begin{document}'):]
labels=re.findall(r'\\label\{([^}]+)',body);assert len(labels)==len(set(labels))
cited=set(x.strip() for s in re.findall(r'\\cite\{([^}]+)',e) for x in s.split(','));bib=set(re.findall(r'\\bibitem\{([^}]+)',e));assert cited<=bib
imgs=re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)',body)
assert all((P/x).is_file() for x in imgs)
assert len(imgs)==14 and len(bib)==30
assert len(re.findall(r'\\begin\{equation\}',e))==11
assert len(re.findall(r'\\tag\{\d+\}',z))==11
abstract=e[e.index(r'\begin{abstract}'):e.index(r'\end{abstract}')]
assert '10.54' not in abstract and '3.87' not in abstract
assert e.count('10.54')==1 and z.count('10.54')==1
log=(P/'main.log').read_text(errors='replace')
assert 'undefined' not in log and 'Overfull' not in log
pages=len(pymupdf.open(P/'main.pdf'))
check={'unique_main':'paper/main.tex + paper/main_zh.md','pages':pages,'main_figures':8,'supplement_figures':6,'explicit_empty_templates':4,'references':len(bib),'equations':11,'all_image_paths_exist':True,'duplicate_labels':False,'undefined_citations_or_refs':False,'overfull_boxes':False,'underfull_boxes':log.count('Underfull'),'visual_check':'All-page contact; final 177mm source figures and event pages inspected','cross_review':'Two material wording issues fixed; bilingual figure numbering repaired','source_audit':'AUDIT_1054_GEOMETRY.md and .json','sha256':{n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['main.tex','main_zh.md','main.pdf']},'test_payloads_read':0}
(D/'UNIFIED_MAIN_CHECK.json').write_text(json.dumps(check,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(check,ensure_ascii=False,indent=2))
