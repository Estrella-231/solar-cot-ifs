"""Re-audit every low-coverage frame against independently calibrated HDF masks."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser()
for n in ('base','grid','region-root','audit-script','output-dir'):
    p.add_argument('--'+n, type=Path, required=True)
p.add_argument('--report',type=Path,required=True)
a=p.parse_args()
report=json.loads(a.report.read_text())
bad=[f for f in report['frames'] if f['valid_fraction'] < .99]
a.output_dir.mkdir(parents=True,exist_ok=True)
results=[]
for f in bad:
    stamp=f['utc']; tag=stamp.replace('-','').replace(':','').replace('T','')
    output=a.output_dir/(tag+'.json')
    subprocess.run([sys.executable,str(a.audit_script),'--base',str(a.base),'--grid',str(a.grid),
        '--region',str(a.region_root/(tag+'.npz')),'--time',stamp,'--output',str(output)],check=True)
    results.append(json.loads(output.read_text()))
summary={'frames':len(results),'all_masks_match':all(x['mask_same'] for x in results),
         'observed_fractions':{x['utc']:x['stored_observed_fraction'] for x in results},
         'test_used':False,'audits':results}
(a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'frames':len(results),'all_masks_match':summary['all_masks_match'],
                  'fractions':summary['observed_fractions']}),flush=True)
if not summary['all_masks_match']:
    raise SystemExit('source observation mask mismatch')
