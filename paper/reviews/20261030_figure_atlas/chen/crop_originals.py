"""Crop unchanged source figures from 180-dpi Poppler page screenshots."""
from pathlib import Path
from PIL import Image
import hashlib, json

HERE=Path(__file__).resolve().parent
PDF=HERE.parents[2]/'references/narrative_review_20261015/chen_2023_spectral_irradiance.pdf'
# Coordinates are pixels, top-left origin, PIL exclusive right/bottom.
ITEMS=[
    ('fig1_data_workflow',3,[80,125,1415,680],'Figure 1'),
    ('fig2_multispectral_inputs',3,[70,715,1415,1090],'Figure 2'),
    ('fig4_model_architecture',6,[80,125,1415,690],'Figure 4'),
    ('fig6_cloud_condition_timeseries',8,[75,770,1420,1455],'Figure 6'),
    ('fig11_cloud_enhancement',12,[85,125,1420,830],'Figure 11'),
    ('table3_station_controls',7,[80,130,1410,530],'Table 3'),
]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
record={'paper':'Chen et al. (2023), Applied Energy 352, 121979',
        'doi':'10.1016/j.apenergy.2023.121979','source_pdf':str(PDF),
        'source_pdf_sha256':sha(PDF),'render':'Poppler pdftoppm, 180 dpi, PNG',
        'transform':'Rectangular crop only; no resampling, recoloring, redrawing or caption replacement',
        'bounds_convention':'[left,top,right,bottom] source pixels, origin upper left; right/bottom excluded','items':[]}
for stem,page,bounds,label in ITEMS:
    full=HERE/f'chen_page-{page:02}.png';out=HERE/f'{stem}.png'
    with Image.open(full) as im:
        im.crop(bounds).save(out)
        size=list(im.size)
    record['items'].append(dict(label=label,pdf_page_one_based=page,printed_page=page,
                               page_image=full.name,page_image_size=size,page_sha256=sha(full),
                               crop_image=out.name,bounds=bounds,crop_sha256=sha(out)))
(HERE/'CROP_MANIFEST.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
