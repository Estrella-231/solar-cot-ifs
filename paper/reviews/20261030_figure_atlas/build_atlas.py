"""Read-only paper excerpts -> a self-contained local reading atlas."""
from pathlib import Path
import base64, html, re, json, hashlib
import markdown

HERE=Path(__file__).resolve().parent
SECTIONS=[('plan','PLAN.md','Plan'),('chen','chen/ANALYSIS.md','Applied Energy'),
          ('amt','amt/ANALYSIS.md','AMT'),('grl','grl/ANALYSIS.md','GRL'),
          ('architecture','architecture/ANALYSIS.md','Architecture')]

def resolve_url(url,root):
    if re.match(r'https?://',url): return url
    p=Path(url) if re.match(r'^[A-Za-z]:',url) else root/url
    return p.resolve().as_uri()

def convert(path):
    body=markdown.markdown(path.read_text(encoding='utf-8'),extensions=['tables','fenced_code'])
    def img(m):
        src=m.group(1)
        p=Path(src) if re.match(r'^[A-Za-z]:',src) else path.parent/src
        if not p.is_file(): raise FileNotFoundError(p)
        mime='image/png' if p.suffix.lower()=='.png' else 'image/jpeg'
        return 'src="data:'+mime+';base64,'+base64.b64encode(p.read_bytes()).decode()+'"'
    body=re.sub(r'src="([^"]+)"',img,body)
    body=re.sub(r'href="([^"]+)"',lambda m:'href="'+html.escape(resolve_url(m.group(1),path.parent),quote=True)+'"',body)
    return body

def main():
    parts=[]; inputs={}
    for ident,name,label in SECTIONS:
        p=HERE/name
        parts.append('<section id="'+ident+'">'+convert(p)+'</section>')
        inputs[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    nav=''.join('<a href="#'+i+'">'+l+'</a>' for i,_,l in SECTIONS)
    out='''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>COT paper | Figure and table reading atlas</title>
<style>
:root{color-scheme:light;--ink:#1f2933;--accent:#116765;--rule:#d9e1e5}
*{box-sizing:border-box}body{margin:0;background:#f1f3f3;color:var(--ink);font:16px/1.8 "Microsoft YaHei","Segoe UI",sans-serif}
nav{position:sticky;top:0;background:#fff;border-bottom:1px solid var(--rule);z-index:2;padding:12px 4%;display:flex;gap:24px;flex-wrap:wrap}
nav a{font-weight:600;color:var(--accent);text-decoration:none}
main{max-width:1180px;margin:32px auto;padding:0 24px}section{background:white;padding:34px 48px;margin:0 0 32px;border:1px solid var(--rule);scroll-margin-top:70px}
h1{font-size:28px;line-height:1.4;margin:0 0 24px}h2{font-size:23px;line-height:1.5;border-top:2px solid var(--rule);padding-top:32px;margin-top:48px}
h3{font-size:19px}a{color:var(--accent)}p{margin:16px 0}strong{color:#111b22}
img{display:block;max-width:100%;height:auto;margin:24px auto;border:1px solid #e4e8ea;padding:8px;background:white}
table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.6;margin:24px 0}th,td{text-align:left;vertical-align:top;padding:10px 12px;border-bottom:1px solid var(--rule)}th{background:#edf3f3;border-top:1.5px solid #9caeb4}code{font-size:.9em;overflow-wrap:anywhere}li{margin:7px 0}
@media(max-width:700px){main{padding:0 8px}section{padding:22px 18px}h1{font-size:23px}table{display:block;overflow:auto}nav{gap:16px}}
@media print{body{background:white;font-size:10pt}nav{display:none}main{max-width:none;margin:0;padding:0}section{border:0;padding:0;break-before:page}h2{break-after:avoid}img{max-height:180mm;object-fit:contain;break-inside:avoid}a{color:inherit}table{font-size:8pt}}
</style><nav>'''+nav+'</nav><main>'+''.join(parts)+'</main></html>'
    output_path=HERE/'READING_ATLAS.html'
    output_path.write_bytes(out.encode('utf-8'))
    (HERE/'ATLAS_BUILD_MANIFEST.json').write_text(json.dumps({'inputs':inputs,'output_bytes':output_path.stat().st_size,
         'output_sha256':hashlib.sha256(output_path.read_bytes()).hexdigest(),
         'images':'Embedded exact locally rendered original-paper crops; no recoloring or fabricated graphics',
         'scope':'Reading and planning only; no manuscript result edits'},indent=2),encoding='utf-8')
    print('Built READING_ATLAS.html')

if __name__=='__main__': main()
