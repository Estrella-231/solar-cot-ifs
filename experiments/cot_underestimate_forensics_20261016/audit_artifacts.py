"""Check immutable-source joins and export a small deliverable hash manifest."""
import argparse,json,hashlib,csv,py_compile
from pathlib import Path
import numpy as np
from PIL import Image

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--directory',type=Path,required=True);a=p.parse_args();d=a.directory;r=d/'results';b=d/'payload';sel=json.loads((r/'selected_cases.json').read_text(encoding='utf-8'));receipt=json.loads((b/'extraction_receipt.json').read_text());summary=json.loads((r/'frozen_model_summary.json').read_text());fd=json.loads((b/'finite_difference_check.json').read_text());orig=a.repo/'experiments/r_geometry_downstream_ghi_20261016/results/paired_three_arm_v1';z=np.load(orig/'analysis/paired_validation_scoring.npz');rows=list(csv.DictReader((orig/'validation_rows.csv').open(encoding='utf-8',newline='')))
 assert sha(orig/'analysis/paired_validation_scoring.npz')==sel['source_predictions_sha256'];assert sha(r/'selected_cases.json')==receipt['source_cases_sha256'];assert sha(b/'real_cloud_cases_and_gradients.npz')==receipt['cloud_payload_sha256'];assert summary['all_checkpoints_frozen_and_unchanged'] and fd['state']=='PASS_INDEPENDENT_CENTRAL_DIFFERENCE'
 for c in sel['cases']:
  i=c['validation_index'];assert c['station']==rows[i]['station'] and c['utc']==rows[i]['target_time_utc'];assert c['ghi']==z['truth'][i];assert abs(c['A_mean']-z['prediction'][:,0,i].mean())<1e-10;assert abs(c['AC_mean']-z['prediction'][:,2,i].mean())<1e-10
 cpp=np.load(b/'cpp_reference_cases.npz');assert np.isnan(cpp['reference_COT'][~cpp['valid']]).all() and np.isfinite(cpp['reference_COT'][cpp['valid']]).all()
 for s in d.glob('*.py'):py_compile.compile(str(s),doraise=True)
 files=[];images=[]
 for f in sorted(d.rglob('*')):
  if not f.is_file() or '__pycache__' in f.parts or f.name in ('frozen_probe_allval.npz','artifact_manifest.json'):continue
  rel=f.relative_to(d).as_posix();files.append(dict(path=rel,bytes=f.stat().st_size,sha256=sha(f)))
  if f.suffix=='.png':
   with Image.open(f) as im:im.verify()
   with Image.open(f) as im:images.append(dict(path=rel,pixels=list(im.size)))
 out=dict(state='PASS_IMMUTABLE_SOURCES_AND_DIAGNOSTIC_ARTIFACTS',validation_rows=len(rows),selected_cases=len(sel['cases']),pairs=len(sel['pairs']),frozen_checkpoints=len(receipt['models']),real_source_frames=len(receipt['frames']),PNG_files_verified=images,files=files,test_payloads_read=0,training_run=False,qsub_used=False)
 (d/'artifact_manifest.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('files','PNG_files_verified')}))
if __name__=='__main__':main()
