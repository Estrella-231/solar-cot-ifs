"""Wait on the real extraction process, retry only failures, freeze final index."""
import csv,json,os,sys,time,hashlib,tempfile
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));import extract_cpp_region as E
FIRST=Path('/tmp/hubei_cot_regional_input_20261023_attempt1');OUT=Path('/tmp/hubei_cot_regional_input_20261023_ready');PID=2318421

def live():
 try:return b'extract_cpp_region.py' in Path(f'/proc/{PID}/cmdline').read_bytes()
 except FileNotFoundError:return False

def main():
 while live():time.sleep(30)
 if not (FIRST/'REPORT.json').exists():raise RuntimeError('Original process terminal without REPORT; do not restart source job')
 old=json.loads((FIRST/'REPORT.json').read_text());records=json.loads((FIRST/'frames.json').read_text());OUT.mkdir(exist_ok=False);(OUT/'payload/train').mkdir(parents=True);(OUT/'payload/val').mkdir()
 E.init(P/'grid_static256.npz',OUT);failures=[];attempts=[]
 for item in old['failures']:
  row=item['row'];p=Path(row['CPP_path']);success=False
  for attempt in range(1,4):
   try:
    before=p.stat();h=hashlib.sha256()
    with tempfile.TemporaryDirectory(prefix='cot_cpp_retry_',dir='/tmp') as tmp:
     local=Path(tmp)/p.name
     with p.open('rb') as f,local.open('wb') as o:
      for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk);o.write(chunk)
     after=p.stat();assert before.st_size==after.st_size==local.stat().st_size and before.st_mtime_ns==after.st_mtime_ns
     r=E.work({**row,'CPP_path':str(local)})
    r.update(CPP_path=str(p),source_bytes=before.st_size,source_mtime_ns=before.st_mtime_ns,source_full_sha256=h.hexdigest(),repaired_from_attempt1=True)
    records.append(r);attempts.append(dict(key=row['time_bjt'],attempt=attempt,state='PASS_LOCAL_STAGED_RETRY'));success=True;print('REPAIRED',row['time_bjt'],flush=True);break
   except Exception as ex:
    attempts.append(dict(key=row['time_bjt'],attempt=attempt,error=repr(ex)));print('RETRY_FAIL',row['time_bjt'],repr(ex),flush=True)
    if attempt<3:time.sleep(attempt*3)
  if not success:failures.append(item)
  (OUT/'RETRY_PROGRESS.json').write_text(json.dumps(dict(repaired=len(attempts)-sum('error' in a for a in attempts),attempts=attempts,remaining_failures=len(failures)),indent=2))
 selected=list(csv.DictReader((P/'selected_cpp_agri.csv').open()));expected={r['time_bjt']:r for r in selected};records.sort(key=lambda r:r['time_bjt']);assert len(set(r['time_bjt'] for r in records))==len(records)
 oldref=np.load(P/'old_CPP_reference.npz');oi={str(k):i for i,k in enumerate(oldref['time_bjt'])};comp=[];maxdiff=0.
 for r in records:
  key=r['time_bjt'];assert expected[key]['split']==r['split'];path=Path(r['output_path']);assert E.sha(path)==r['output_sha256']
  dest=OUT/'payload'/r['split']/(key+'.npz')
  if dest!=path:dest.symlink_to(path)
  with np.load(path) as z:
   assert z['cot'].shape==z['clp'].shape==z['mask'].shape==(256,256)
   if key in oi:
    j=oi[key];assert str(oldref['split'][j])==r['split'];c=z['cot'][103:119,118:134];m=z['mask'][103:119,118:134]&oldref['mask'][j,0]
    if m.any():
     diff=float(abs(c[m]-oldref['cot'][j,0][m]).max());maxdiff=max(maxdiff,diff);comp.append(dict(time_bjt=key,pixels=int(m.sum()),max_abs_error=diff))
  r['ready_path']=str(dest)
 summary={}
 for s in ['train','val']:
  rr=[r for r in records if r['split']==s];summary[s]=dict(frames=len(rr),frames_with_valid=sum(r['valid_pixels']>0 for r in rr),valid_pixels=sum(r['valid_pixels'] for r in rr),dates=len(set(r['time_bjt'][:8] for r in rr)),by_month={m:dict(frames=sum(r['time_bjt'].startswith(m) for r in rr),valid_pixels=sum(r['valid_pixels'] for r in rr if r['time_bjt'].startswith(m))) for m in sorted(set(r['time_bjt'][:6] for r in rr))})
 (OUT/'frames.json').write_text(json.dumps(records));(OUT/'REFERENCE_COMPARISON.json').write_text(json.dumps(dict(frames=len(comp),pixels=sum(r['pixels'] for r in comp),max_abs_error=maxdiff,tolerance=1e-4,records=comp),indent=2))
 passed=not failures and len(records)==len(expected) and bool(comp) and maxdiff<1e-4
 report=dict(state='PASS_REGIONAL_CPP_SOURCE_AND_REFERENCE' if passed else 'GATED_REMAINING_SOURCE_OR_REFERENCE_ISSUE',frames=len(records),expected=len(expected),summary=summary,remaining_failures=failures,retry_attempts=attempts,reference_max_abs_error=maxdiff,reference_frames=len(comp),test_payloads_read=0,buffer_payloads_read=0,GHI_labels_read=0,auto_start_training=False)
 (OUT/'REPORT.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['summary','retry_attempts']}),flush=True)
if __name__=='__main__':main()
