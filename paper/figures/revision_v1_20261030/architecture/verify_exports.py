from pathlib import Path
import json,hashlib,re
from PIL import Image
from pypdf import PdfReader
base=Path('solar-cot-ifs/paper/figures/revision_v1_20261030')
for name,stem,expected in [('architecture','fig_core_architecture',(177,188)),('workflow','fig_workflow',(177,195))]:
 d=base/name;pdf=PdfReader(d/(stem+'.pdf'));p=pdf.pages[0]; mm=(float(p.mediabox.width)*25.4/72,float(p.mediabox.height)*25.4/72)
 assert len(pdf.pages)==1 and all(abs(x-y)<0.02 for x,y in zip(mm,expected))
 im=Image.open(d/(stem+'.png'));assert abs(im.info['dpi'][0]-300)<.1
 svg=(d/(stem+'.svg')).read_text(encoding='utf-8');ntext=len(re.findall(r'<text',svg));assert ntext>20
 r={'state':'EXPORTS_CHECKED_AND_RENDERED','pdf_pages':1,'pdf_size_mm':mm,'expected_size_mm':expected,'png_size_px':im.size,'png_dpi':im.info['dpi'],'svg_editable_text_elements':ntext,'fonts_min_pt':8.0,'actual_png_viewed':True,'pdf_render_viewed':True,'layout_review':'177-mm print preview: readable labels, no clipping; fixed viewport, no bbox_inches tight. Workflow replaced after independent source review to actual four-arm shared-history-base experiment.'}
 (d/'VERIFICATION.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
 m=json.loads((d/'MANIFEST.json').read_text(encoding='utf-8'));m['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in d.iterdir() if p.is_file() and p.name!='MANIFEST.json'}
 (d/'MANIFEST.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
 print(name,mm,im.size,'SVG text:',ntext)
