"""Read-only PPT evidence inventory and embedded-image contact sheets."""
import hashlib,json,posixpath,re,zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image,ImageDraw,ImageFont
import io

OUT=Path(__file__).resolve().parent
SOURCE=Path('C:/Users/29728/Desktop/2026.10.16.pptx')
NS={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}

def main():
    folder=OUT/'ppt_embedded';folder.mkdir(exist_ok=True)
    items=[]
    with zipfile.ZipFile(SOURCE) as z:
        slides=sorted([p for p in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+.xml',p)],key=lambda p:int(re.search(r'(\d+)\.xml',p).group(1)))
        for no,path in enumerate(slides,1):
            root=ET.fromstring(z.read(path))
            texts=[t.text or '' for t in root.findall('.//a:t',NS)]
            relpath=posixpath.join(posixpath.dirname(path),'_rels',posixpath.basename(path)+'.rels')
            rels={r.attrib['Id']:r.attrib['Target'] for r in ET.fromstring(z.read(relpath))} if relpath in z.namelist() else {}
            images=[]
            for n,blip in enumerate(root.findall('.//a:blip',NS)):
                rid=blip.attrib.get('{'+NS['r']+'}embed')
                if rid not in rels:continue
                p=posixpath.normpath(posixpath.join(posixpath.dirname(path),rels[rid]))
                data=z.read(p);dest=folder/f'slide{no}_image{n+1}{Path(p).suffix}';dest.write_bytes(data)
                try:
                    with Image.open(io.BytesIO(data)) as im:dim=im.size
                except Exception:dim=None
                images.append(dict(path=str(dest),archive=p,dimensions=dim,sha256=hashlib.sha256(data).hexdigest()))
            notes=f'ppt/notesSlides/notesSlide{no}.xml'
            note_text=' | '.join(t.text or '' for t in ET.fromstring(z.read(notes)).findall('.//a:t',NS)) if notes in z.namelist() else ''
            items.append(dict(slide=no,text=texts,notes=note_text,images=images))
            thumbs=[]
            for entry in images:
                if not entry['dimensions']:continue
                im=Image.open(entry['path']).convert('RGB');im.thumbnail((1080,650))
                canvas=Image.new('RGB',(1120,710),'white');canvas.paste(im,((1120-im.width)//2,35))
                ImageDraw.Draw(canvas).text((12,10),Path(entry['path']).name,fill='black')
                thumbs.append(canvas)
            if thumbs:
                columns=min(2,len(thumbs));rows=(len(thumbs)+columns-1)//columns
                sheet=Image.new('RGB',(1120*columns,710*rows),'#dddddd')
                for n,im in enumerate(thumbs):sheet.paste(im,((n%columns)*1120,(n//columns)*710))
                sheet.save(OUT/f'slide{no}_evidence.png')
    result=dict(source=str(SOURCE),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),slides=items)
    (OUT/'ppt_inventory.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    for r in items:print(json.dumps(dict(slide=r['slide'],text=r['text'],notes=r['notes'],images=[Path(i['path']).name for i in r['images']]),ensure_ascii=False))

if __name__=='__main__':main()
