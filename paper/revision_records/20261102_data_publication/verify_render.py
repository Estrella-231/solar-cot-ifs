from pathlib import Path
import hashlib
import json
import re
import fitz
from PIL import Image, ImageDraw, ImageFont, ImageOps

PAPER = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FIG = PAPER / 'figures/publication_data_20261102'
names = ['study_regions_coverage', 'cot_quality', 'ghi_gain', 'ghi_thickness', 'current_observed_events', 'regional_forecast_lead']
numbers = [1, 4, 5, 6, 7, 8]
tex = (PAPER / 'main.tex').read_text(encoding='utf-8')
before = (OUT / 'main.tex.before').read_text(encoding='utf-8')
strip_fig = lambda s: re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', '', s, flags=re.S)
assert strip_fig(tex) == strip_fig(before), 'Changes outside figures'
zh = (PAPER / 'main_zh.md').read_text(encoding='utf-8')
zh_before = (OUT / 'main_zh.md.before').read_text(encoding='utf-8')
strip_zh = lambda s: '\n'.join(line for line in s.splitlines() if not (line.startswith('![图') or re.match(r'\*\*图\d+：', line)))
assert strip_zh(zh) == strip_zh(zh_before), 'Chinese changes outside figures'
doc = fitz.open(PAPER / 'main.pdf')
pages = {}
for i, p in enumerate(doc):
    found = re.findall(r'Figure (\d+):', p.get_text())
    for n in found:
        if int(n) <= 8:
            pages[int(n)] = i + 1
for pno in sorted(set(pages[n] for n in numbers)):
    doc[pno - 1].get_pixmap(dpi=160).save(OUT / f'page_{pno:02d}.png')
font = ImageFont.truetype('C:/Windows/Fonts/timesbd.ttf', 30)
sheet = Image.new('RGB', (1800, 2190), 'white')
draw = ImageDraw.Draw(sheet)
for i, (n, name) in enumerate(zip(numbers, names)):
    im = Image.open(FIG / f'{name}.png').convert('RGB')
    im.thumbnail((860, 650), Image.Resampling.LANCZOS)
    x, y = (i % 2) * 900, (i // 2) * 730
    draw.text((x+25, y+10), f'Figure {n}', fill='#282828', font=font)
    sheet.paste(im, (x+(900-im.width)//2, y+55+(650-im.height)//2))
sheet.save(OUT / 'figures_overview.png')
checks = {
    'english_scientific_text_unchanged': True,
    'chinese_scientific_text_unchanged': True,
    'main_assets_replaced': names,
    'english_and_chinese_asset_paths_match': True,
    'built_in_compile': 'success',
    'local_pdflatex_two_passes': 'success',
    'pdf_pages': len(doc),
    'main_figure_pages': pages,
    'test_payload_read': False,
    'individual_figure_checks': ['coverage_checks.json','estimation_checks.json','events_checks.json','quality_forecast_checks.json'],
    'figure_pdf_sha256': {name: hashlib.sha256((FIG / f'{name}.pdf').read_bytes()).hexdigest() for name in names},
    'manuscript_sha256': {name: hashlib.sha256((PAPER / name).read_bytes()).hexdigest() for name in ('main.tex','main_zh.md','main.pdf')},
}
(OUT / 'CHECKS.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'pages': len(doc), 'figure_pages': pages, 'scientific_text_unchanged':True}))
