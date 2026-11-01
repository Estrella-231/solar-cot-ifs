"""Render/crop cited AMT original pages, without redrawing or enhancing them."""
from pathlib import Path
import hashlib
import json
import subprocess
from PIL import Image

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[2]
SOURCE = PAPER / 'references/narrative_review_20261015/papachristopoulou_2024_cloud_aerosol.pdf'
TOOL = Path('C:/Users/29728/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe')
DPI = 150
ITEMS = [
    ('fig1',5,1855,(270,145,735,510)),
    ('fig2',7,1857,(250,145,780,430)),
    ('fig11',17,1867,(270,145,765,415)),
    ('fig15',19,1869,(655,760,510,330)),
    ('table4',17,1867,(270,680,740,350)),
]
for page in sorted({v[1] for v in ITEMS}):
    subprocess.run([str(TOOL),'-png','-singlefile','-r',str(DPI),'-f',str(page),'-l',str(page),str(SOURCE),str(HERE/f'page_{page}')],check=True)
records=[]
for label,page,published,bounds in ITEMS:
    x,y,w,h=bounds
    subprocess.run([str(TOOL),'-png','-singlefile','-r',str(DPI),'-f',str(page),'-l',str(page),
                    '-x',str(x),'-y',str(y),'-W',str(w),'-H',str(h),str(SOURCE),str(HERE/f'{label}_original')],check=True)
    output=HERE/f'{label}_original.png'
    with Image.open(output) as image: size=list(image.size)
    records.append(dict(id=label,pdf_page_one_based=page,published_page=published,
                        full_page=f'page_{page}.png',crop_file=output.name,
                        crop_xywh_pixels_at_150dpi=list(bounds),crop_origin='top left of rendered PDF page',
                        crop_xywh_pdf_points=[v*72/DPI for v in bounds],output_size=size,
                        sha256=hashlib.sha256(output.read_bytes()).hexdigest()))
manifest=dict(
    source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    source_title='Effects of clouds and aerosols on downwelling surface solar irradiance nowcasting and short-term forecasting',
    authors='Papachristopoulou et al.',year=2024,venue='Atmospheric Measurement Techniques',volume=17,pages='1851–1877',
    doi='10.5194/amt-17-1851-2024',official_url='https://amt.copernicus.org/articles/17/1851/2024/',
    copyright='© Author(s) 2024',license='Creative Commons Attribution 4.0',license_url='https://creativecommons.org/licenses/by/4.0/',
    operations='Original PDF page rasterization at 150 dpi and rectangular cropping only; no redraw, contrast change, annotation, or data modification',
    crop_notice='Figure/table captions remain on the supplied full-page screenshot; each figure crop retains axes, labels, legends and colorbars as applicable.',
    renderer=str(TOOL),dpi=DPI,items=records)
(HERE/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('Rendered',len(ITEMS),'original figure/table crops')
