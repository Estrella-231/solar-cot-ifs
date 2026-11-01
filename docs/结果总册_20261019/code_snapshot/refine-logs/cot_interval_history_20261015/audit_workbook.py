"""Read only the requested train/val source cells plus workbook headers."""
import argparse,csv,importlib.util,json,zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--requests',type=Path,required=True);ap.add_argument('--extractor',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    spec=importlib.util.spec_from_file_location('wb_allowed_cells',a.extractor);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    rows=list(csv.DictReader(a.requests.open()));result={}
    for st,name in [('sili','四里光伏.xlsx'),('zhujia','竺家光伏.xlsx')]:
        path=Path('/home/Data_Pool_3/chenyi/GHI/data/hunan/raw/stations')/name;needed=defaultdict(dict);expected={}
        for r in rows:
            if r['station']!=st:continue
            t=datetime.fromisoformat(r['time_bjt'])
            # Audit daylight zeros only; do not request midnight VAL2400 from a different date.
            if not 6<=t.hour<=19:continue
            needed[t.date()][t.strftime('%H%M')]=t;expected[t]=float(r['expected'])
        vals,meta=mod.extract_requested(path,needed);diffs=[abs(v-expected[t]) for t,v in vals.items() if v is not None]
        headers={};extra=[]
        with zipfile.ZipFile(path) as z:
            ss=mod.shared_strings(z);wb=ET.fromstring(z.read('xl/workbook.xml'))
            sheet_names=[s.attrib['name'] for s in wb.findall('.//{'+mod.MAIN+'}sheet')]
            for sheet_name in sheet_names:
                with z.open(mod.sheet_path(z,sheet_name)) as stream:
                    for _,e in ET.iterparse(stream,events=('end',)):
                        if e.tag=='{'+mod.MAIN+'}row':
                            h=[mod.cell_payload(c,ss) for c in e.findall('{'+mod.MAIN+'}c')];headers[sheet_name]=h;break
            # Metadata columns are only decoded on the already permitted daily rows.
            with z.open(mod.sheet_path(z,'结果集')) as stream:
                date_col=None;colnames={}
                for _,e in ET.iterparse(stream,events=('end',)):
                    if e.tag!='{'+mod.MAIN+'}row':continue
                    cells={mod.col_num(c.attrib['r']):c for c in e.findall('{'+mod.MAIN+'}c')}
                    if date_col is None:
                        colnames={i:mod.cell_payload(c,ss) for i,c in cells.items()};date_col=next(i for i,n in colnames.items() if n=='DATA_DATE');e.clear();continue
                    dc=cells.get(date_col);raw=mod.cell_payload(dc,ss) if dc is not None else None
                    if raw and mod.excel_date(raw) in needed:
                        extra.append({str(colnames[i]):mod.cell_payload(c,ss) for i,c in cells.items() if colnames.get(i) and not str(colnames[i]).startswith('VAL')})
                    e.clear()
        result[st]=dict(source=str(path),metadata=meta,sheets=sheet_names,non_value_headers={s:[x for x in h if x and not x.startswith('VAL')] for s,h in headers.items()},
            requested_nonmissing_cells=len(diffs),requested_missing_cells=sum(v is None for v in vals.values()),zero_cells=sum(v==0 for v in vals.values()),max_abs_cached_source_difference=max(diffs,default=None),
            selected_day_metadata=extra,values={t.isoformat(sep=' '):v for t,v in vals.items()})
    a.output.write_text(json.dumps(dict(state='COMPLETE_ALLOWED_SOURCE_WORKBOOK_RECHECK',stations=result,test_payloads_read=0),ensure_ascii=False,indent=2))
    print(json.dumps({st:{k:v for k,v in r.items() if k not in ('values','selected_day_metadata')} for st,r in result.items()},ensure_ascii=False))


if __name__=='__main__':main()
