from pathlib import Path
import json,re,csv,subprocess,hashlib
from pypdf import PdfReader,PdfWriter
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent.parent
pages=json.loads((ROOT/'page_manifest.json').read_text(encoding='utf-8'))
reader=PdfReader(ROOT/'main.pdf')
assert len(reader.pages)==len(pages),(len(reader.pages),len(pages))
missing=[]
for i,(pdfpage,entry) in enumerate(zip(reader.pages,pages)):
    text=pdfpage.extract_text()
    if entry['id'] not in text:missing.append([i+1,entry['id']])
assert not missing,missing
log=(ROOT/'main.log').read_text(encoding='utf-8',errors='replace')
assert 'Missing character' not in log
assert 'Overfull' not in log
assert '! LaTeX Error' not in log

writer=PdfWriter();writer.append(reader)
seen=set()
for i,p in enumerate(pages):
    if p['group'] not in seen:
        seen.add(p['group']);writer.add_outline_item(p['title'],i)
writer.add_metadata({'/Title':'COT研究全部结果与复现索引','/Subject':'2026-10-19；逐页一个图表；含当前结果、历史探索、模型及代码目录'})
with (ROOT/'COT_全部结果与复现索引.pdf').open('wb') as f:writer.write(f)

# A short reading route, extracted from exactly the same compiled pages; no new result values.
short_ids=['P001']
pick=[('E01','figure'),('E02','table'),('E03','figure'),('E04','figure'),('E05','table'),('E06','table'),('E08','table'),('E10','table'),('E12','table'),('E15','table'),('E16','figure'),('E23','figure'),('E24','table'),('E25','table'),('E26','table'),('E27','table'),('E28','table'),('E29','figure'),('E30','table'),('E31','table'),('E32','table'),('E33','table')]
for group,kind in pick:
    found=[p for p in pages if p['group']==group and p['kind']==kind and p['scope']!='工程复现附录']
    # First E12 table is a geometry diagnostic; the next is the actual GHI result.
    if group=='E12':found=found[1:]
    if found:short_ids.append(found[0]['id'])
short_ids += [p['id'] for p in pages if p['group']=='最新辅助监督' and p['kind']=='table' and '站点中心' in p['title']]
short_ids += [p['id'] for p in pages if p['group']=='湖北最终三臂']
short_ids=list(dict.fromkeys(short_ids));index={p['id']:i for i,p in enumerate(pages)}
short=PdfWriter()
for sid in short_ids:
    short.add_page(reader.pages[index[sid]]);short.add_outline_item(pages[index[sid]]['title'],len(short.pages)-1)
short.add_metadata({'/Title':'COT研究重点结果速览','/Subject':'从完整LaTeX结果册中提取；原页码保留'})
with (ROOT/'COT_重点结果速览.pdf').open('wb') as f:short.write(f)

review=ROOT/'review';review.mkdir(exist_ok=True)
subprocess.run(['pdftoppm','-scale-to','430','-png',str(ROOT/'main.pdf'),str(review/'thumb')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
thumbs=sorted(review.glob('thumb-*.png'))
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
for k in range(0,len(thumbs),30):
    sheet=Image.new('RGB',(430*5,326*6),'#dce2e8');draw=ImageDraw.Draw(sheet)
    for j,path in enumerate(thumbs[k:k+30]):
        im=Image.open(path).convert('RGB');im.thumbnail((420,295))
        x=(j%5)*430+5;y=(j//5)*326+25
        sheet.paste(im,(x,y));draw.text((x,y-20),f'PDF page {k+j+1}',font=font,fill='black')
    sheet.save(review/f'contact_{k//30+1}.png')

selected={p['source'] for p in pages if p['kind']=='figure'}
inventory=[]
for q in REPO.rglob('*.png'):
    if ROOT in q.parents or any(x in q.parts for x in ('node_modules','python_deps','.git')):continue
    rel=q.relative_to(REPO).as_posix()
    state='本册已采用' if rel in selected else ('作废CPP目录；不引用' if (q.parent/'INVALID_CPP_JOIN.md').exists() else '其他图/重复快照/工程材料；保留原目录')
    inventory.append(dict(path=str(q),status=state,bytes=q.stat().st_size))
with (ROOT/'ALL_FIGURE_INDEX.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(inventory[0]));w.writeheader();w.writerows(inventory)

report=dict(compiled_pages=len(reader.pages),planned_pages=len(pages),one_figure_or_table_per_page=True,
            page_id_parity=True,missing_glyph_warnings=0,overfull_warnings=0,
            short_reading_pages=len(short_ids),short_original_page_ids=short_ids,
            png_inventory_count=len(inventory),contact_sheets=(len(thumbs)+29)//30,
            note='结构检查通过；contact和代表页另做人工视觉检查；不重新运行模型或读取test')
(ROOT/'LAYOUT_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
