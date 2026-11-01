"""Read requested train/val zero cells and non-measurement workbook metadata."""
import argparse,csv,json,zipfile,collections,importlib.util
from pathlib import Path
from datetime import datetime,timedelta
from xml.etree import ElementTree as ET

def main():
    p=argparse.ArgumentParser();p.add_argument('--requests',type=Path,required=True);p.add_argument('--extractor',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    sp=importlib.util.spec_from_file_location('reader',a.extractor);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
    wanted=collections.defaultdict(lambda:collections.defaultdict(dict))
    for r in csv.DictReader(a.requests.open()):
        assert set(r['split'].split('|'))<={'train','val'}
        t=datetime.fromisoformat(r['time_bjt']);day=t.date();hh=t.strftime('%H%M')
        if hh=='0000':day-=timedelta(days=1);hh='2400'
        wanted[r['station']][day][hh]=r
    report={}
    for st in ('sili','zhujia'):
        path=Path('/home/Data_Pool_3/chenyi/GHI/data/hunan/raw/stations')/('四里光伏.xlsx' if st=='sili' else '竺家光伏.xlsx')
        counts=collections.Counter();examples=[];metadata=[];daycounts=collections.Counter()
        with zipfile.ZipFile(path) as z:
            wb=ET.fromstring(z.read('xl/workbook.xml'));sheets=[sh.attrib.get('name') for sh in wb.findall(f'.//{{{m.MAIN}}}sheet')];shared=m.shared_strings(z);xml=m.sheet_path(z,'结果集');headers=None
            for event,e in ET.iterparse(z.open(xml),events=('end',)):
                if e.tag!=f'{{{m.MAIN}}}row':continue
                cells={m.col_num(c.attrib['r']):c for c in e.findall(f'{{{m.MAIN}}}c') if 'r' in c.attrib}
                if headers is None:
                    headers={i:m.cell_payload(c,shared) for i,c in cells.items()};rev={v:i for i,v in headers.items()};datecol=rev['DATA_DATE'];e.clear();continue
                dc=cells.get(datecol);dt=m.cell_payload(dc,shared) if dc is not None else None
                if not dt:e.clear();continue
                day=m.excel_date(dt)
                if day not in wanted[st]:e.clear();continue
                nonval={str(headers[i]):m.cell_payload(c,shared) for i,c in cells.items() if i in headers and not str(headers[i]).startswith('VAL')}
                if len(metadata)<12:metadata.append(nonval)
                for hh,req in wanted[st][day].items():
                    c=cells.get(rev['VAL'+hh]);val=m.cell_payload(c,shared) if c is not None else None;formula=c.find(f'{{{m.MAIN}}}f') if c is not None else None
                    kind='missing_cell' if c is None else ('formula' if formula is not None else (('explicit_numeric_zero' if c.attrib.get('t','n') in ('n','') else 'explicit_text_zero') if val is not None and float(val)==0 else 'other'))
                    counts[kind]+=1;daycounts[str(day)]+=1
                    assert val is not None and float(val)==0,(st,req['time_bjt'],val)
                    if len(examples)<20:examples.append(dict(time_bjt=req['time_bjt'],cell=c.attrib,raw_payload=val,formula=formula.text if formula is not None else None,xml=ET.tostring(c,encoding='unicode')))
                e.clear()
        report[st]=dict(path=str(path),sheets=sheets,non_VAL_headers={str(i):h for i,h in headers.items() if not str(h).startswith('VAL')},counts=dict(counts),zero_days=len(daycounts),selected_day_metadata=metadata,examples=examples)
    a.output.write_text(json.dumps(dict(state='COMPLETE_ORIGINAL_ZERO_CELL_AUDIT',stations=report,test_payloads_read=0),ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({st:r['counts'] for st,r in report.items()}))
if __name__=='__main__':main()
