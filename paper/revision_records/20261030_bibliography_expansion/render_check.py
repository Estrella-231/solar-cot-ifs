from pathlib import Path
import subprocess
import json
import fitz
from PIL import Image, ImageOps, ImageDraw

D = Path(__file__).resolve().parent
P = D.parents[1]
logs=[]
for _ in range(2):
    r=subprocess.run(['C:/texlive/2025/bin/windows/pdflatex.exe','-interaction=nonstopmode','-halt-on-error','main.tex'],cwd=P,capture_output=True,text=True,encoding='utf8',errors='replace')
    logs.append(r.stdout)
    if r.returncode: raise RuntimeError(r.stdout[-3000:])
(D/'compile.log').write_text('\n'.join(logs),encoding='utf-8')
out=D/'render'; out.mkdir(exist_ok=True)
doc=fitz.open(P/'main.pdf')
pages=[]
for i,page in enumerate(doc):
    pix=page.get_pixmap(matrix=fitz.Matrix(1.25,1.25),alpha=False)
    path=out/f'page_{i+1:02}.png'; pix.save(path)
    im=Image.open(path).convert('RGB'); im.thumbnail((400,570))
    card=Image.new('RGB',(420,605),'#dddddd')
    card.paste(im,((420-im.width)//2,24))
    ImageDraw.Draw(card).text((10,5),f'Page {i+1}',fill='black')
    pages.append(card)
for start in range(0,len(pages),6):
    sheet=Image.new('RGB',(1260,1210),'white')
    for j,im in enumerate(pages[start:start+6]):sheet.paste(im,((j%3)*420,(j//3)*605))
    sheet.save(out/f'contact_{start+1:02}_{min(start+6,len(pages)):02}.png')
checks={'page_count':len(doc),'warnings':[l for l in logs[-1].splitlines() if any(k in l for k in ('Warning','Overfull','Underfull'))],
        'missing_citation_markers': any('[?]' in p.get_text() for p in doc),
        'reference_numbers_present': all(f'[{i}]' in '\n'.join(p.get_text() for p in doc[-4:]) for i in range(1,31))}
(D/'LAYOUT_CHECK.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
print(json.dumps(checks,indent=2))
