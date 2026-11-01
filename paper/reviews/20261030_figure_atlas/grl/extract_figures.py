"""Render original journal pages and crop figures for private critical reading.
No numerical recreation, recoloring, rescaling of data or figure editing.
"""
from pathlib import Path
import hashlib,json,subprocess
from PIL import Image
from pypdf import PdfReader

HERE=Path(__file__).resolve().parent
SOURCE=Path('C:/Users/29728/xwechat_files/wxid_s79spdfmjaq912_2146/msg/file/2026-10/202603_一作+通讯_GRL_Short‑Term Forecasting of Cloud Physical Properties Based on Fourier Neural Operator Method.pdf')
POPPLER=Path('C:/Users/29728/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe')
DPI=220
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

# Bounds are normalized fractions of a rendered original page: left/top/right/bottom.
# Captions are included in the four principal crops, never redrawn.
CROPS=[
 ('figure1_method',4,(.045,.105,.952,.470),'Figure 1: architecture and autoregression; original caption included'),
 ('figure2_hourly_metrics',7,(.045,.102,.952,.570),'Figure 2: all 24 panels with scales and original caption'),
 ('figure3_spatial_forecasts',8,(.267,.102,.956,.814),'Figure 3: all paired spatial panels, colorbars and original caption'),
 ('table1_modis_reference',7,(.045,.750,.477,.933),'Table 1: all rows, header and original title'),
 ('figure2_cot_row_detail',7,(.110,.423,.895,.525),'Figure 2 panels s-x only; COT row detail, axes and density colorbar retained'),
 ('figure3_cot_rows_detail',8,(.272,.611,.955,.763),'Figure 3 COT true/forecast rows only; maps and common colorbar retained'),
]
reader=PdfReader(SOURCE)
records=[]
for page in [4,7,8]:
    dest=HERE/f'page{page:02d}_original'
    subprocess.run([str(POPPLER),'-f',str(page),'-l',str(page),'-png','-singlefile','-r',str(DPI),str(SOURCE),str(dest)],check=True)
    image_path=dest.with_suffix('.png')
    with Image.open(image_path) as im:
        w,h=im.size
        for name,pg,bounds,note in CROPS:
            if pg!=page:continue
            box=tuple(round(f*v) for f,v in zip(bounds,(w,h,w,h)))
            target=HERE/f'{name}.png'
            im.crop(box).save(target,dpi=(DPI,DPI))
            records.append(dict(name=name,page_one_based=page,original_page_file=image_path.name,
                original_page_size_px=[w,h],original_page_sha256=sha(image_path),
                crop_bounds_px_ltrb=list(box),crop_bounds_normalized_ltrb=list(bounds),
                crop_file=target.name,crop_sha256=sha(target),note=note))
manifest=dict(title='Short-Term Forecasting of Cloud Physical Properties Based on Fourier Neural Operator Method',
    journal='Geophysical Research Letters',year=2026,volume=53,article='e2025GL119553',doi='10.1029/2025GL119553',
    authors_first='Feng Zhang',source_path=str(SOURCE),source_bytes=SOURCE.stat().st_size,source_sha256=sha(SOURCE),
    pdf_page_count=len(reader.pages),pdf_metadata={str(k):str(v) for k,v in reader.metadata.items()},
    page_numbering='PDF page numbers equal printed article page numbers (1-11)',
    render_dpi=DPI,renderer=str(POPPLER),crops=records,
    use='Original source excerpts for private critical reading; not original artwork and not inserted into our manuscript.',
    transformations=['PDF rasterization at 220 dpi','rectangular cropping only; no content alteration'],
    source_license_notice='PDF page 1 states CC Attribution-NonCommercial-NoDerivs; cite original article and do not reuse as own figure.',
    extraction_script_sha256=sha(Path(__file__)))
(HERE/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(pages=len(reader.pages),source_bytes=SOURCE.stat().st_size,crops=len(records)),ensure_ascii=False))
