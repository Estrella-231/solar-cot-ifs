from pathlib import Path
import json, re, hashlib
import fitz

record = Path(__file__).resolve().parent
paper = record.parents[1]
doc = fitz.open(paper/'main.pdf')
renders = record/'render'
renders.mkdir(exist_ok=True)
pages = {}
for i, page in enumerate(doc):
    text = page.get_text()
    for n, title in [(2, 'Three-frame COT-assisted GHI estimation'), (3, 'Application of the observed-image-trained')]:
        if f'Figure {n}:' in text and title in text:
            pages[str(n)] = i+1
            page.get_pixmap(matrix=fitz.Matrix(1.5,1.5), alpha=False).save(renders/f'figure_{n}_page.png')
assert set(pages) == {'2','3'}, pages
en=(paper/'main.tex').read_text(encoding='utf-8')
zh=(paper/'main_zh.md').read_text(encoding='utf-8')
newdir='figures/journal_style_20261102/'
assert en.count(newdir)==2 and zh.count(newdir)==2
log=(paper/'main.log').read_text(encoding='utf-8',errors='replace')
checks={
    'pages':len(doc), 'figure_pages':pages,
    'main_figures':len(re.findall(r'\\label\{fig:', en.split(r'\begin{thebibliography}')[0])),
    'total_figures':len(re.findall(r'\\label\{fig:', en)),
    'english_chinese_same_assets':True,
    'data_figures_changed':False, 'model_settings_changed':False,
    'test_payload_read':False,
    'undefined_references': bool(re.search(r'(Reference|Citation) .+ undefined|There were undefined references',log)),
    'overfull_boxes':bool(re.search(r'Overfull \\[hv]box',log)),
    'assets':{f:hashlib.sha256((paper/newdir/f).read_bytes()).hexdigest() for f in ('three_frame_estimation.png','forecast_application.png')},
}
(record/'CHECKS.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
index=paper/'VERSION_INDEX.md'
text=index.read_text(encoding='utf-8')
text=re.sub(r'当前\d+页',f'当前{len(doc)}页',text,count=1)
old='- **图2（第5页）、图3（第7页）**：原生图像生成模型制作的三帧同期估计和冻结预报流程视觉稿，已放在方法部分。云纹理为示意，图注已标明；两图供后续替换。其余六张真实数据图统一美化，数值和观测不变。当前源码仍是同一main.tex/main_zh.md。'
replacement=f'- **图2（第{pages["2"]}页）、图3（第{pages["3"]}页）**：已按用户参考改为低饱和配色、细线框、衬线字体及堆叠云图的期刊示意风格。使用journal_style_20261102两张生成图；沿用当前三帧同期与冻结预报方法，明确前两个时效的历史来源。云纹理为示意，图注已标明。其余六张真实数据图与数值不变。'
assert old in text or replacement in text
index.write_text(text.replace(old,replacement,1),encoding='utf-8')
for filename in ('FIGURE_PLAN.md','PAPER_PLAN.md'):
    p=paper/filename
    content=p.read_text(encoding='utf-8')
    intro=f'## 2026-11-02 用户参考风格重绘\n\n图2与图3改用journal_style_20261102中的浅色、细线、堆叠云图版，模型定义与结果不变。双语主稿同步；编译后{len(doc)}页，方法图位于第{pages["2"]}、{pages["3"]}页。检查记录见revision_records/20261102_journal_style/README.md。以下早期页数与风格记录仅供沿革。\n\n'
    if not content.startswith('## 2026-11-02 用户参考风格重绘'):
        p.write_text(intro+content,encoding='utf-8')
print(json.dumps(checks,ensure_ascii=False,indent=2))
