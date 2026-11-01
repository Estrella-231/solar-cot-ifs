import csv, json, os, hashlib
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

BASE=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_solar_geometry_repair_20261021')
RPACK=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_cloud_probability_20261020/R_pack_correctutc_attempt2')
ROOT=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/hubei_independent_trainval_stage_20261017/data')
rows=list(csv.DictReader((RPACK/'rows.csv').open(encoding='utf-8-sig')))
missing=[]; bad_shape=[]; nonfinite=[]; compared=0; mismatches=[]; maxdiff=0.0; examples=[]
AG=[0,1,2,3,4,5,8,9,10,11,12,13,14]
x_norm=np.load(RPACK/'x_norm.npy',mmap_mode='r',allow_pickle=False)
norm=json.loads((RPACK/'norm.json').read_text())
mu=np.asarray(norm['mean'],dtype=np.float64); sd=np.asarray(norm['std'],dtype=np.float64)
for i,r in enumerate(rows):
    bjt=datetime.strptime(r['time_bjt'],'%Y%m%d%H%M'); utc=bjt-timedelta(hours=8); u=utc.strftime('%Y%m%d%H%M')
    p=ROOT/f'{u}.npy'
    if not p.is_file():
        missing.append({'bjt':r['time_bjt'],'utc':u,'path':str(p)}); continue
    a=np.load(p,mmap_mode='r',allow_pickle=False)
    if a.shape != (20,256,256):
        bad_shape.append({'bjt':r['time_bjt'],'shape':list(a.shape),'path':str(p)}); continue
    crop=np.asarray(a[AG,103:119,118:134],dtype=np.float32)
    if not np.isfinite(crop).all(): nonfinite.append({'bjt':r['time_bjt']}); continue
    old=np.asarray(x_norm[i,:13],dtype=np.float64)*sd[:13,None,None]+mu[:13,None,None]
    d=np.abs(crop.astype(np.float64)-old.astype(np.float64)); md=float(d.max()); maxdiff=max(maxdiff,md); compared+=1
    if md != 0.0:
        entry={'bjt':r['time_bjt'],'utc':u,'max_abs':md,'n_different':int(np.count_nonzero(d)),'per_channel_max':d.max(axis=(1,2)).tolist()}
        mismatches.append(entry)
        if len(examples)<16: examples.append(entry)
report={'state':'STAGE_R_COVERAGE_AUDIT','r_rows':len(rows),'unique_bjt':len({r['time_bjt'] for r in rows}),'root':str(ROOT),'exists':len(rows)-len(missing),'missing_count':len(missing),'missing_examples':missing[:12],'bad_shape':bad_shape[:12],'bad_shape_count':len(bad_shape),'nonfinite':nonfinite[:12],'nonfinite_count':len(nonfinite),'compared_to_trainval_r_patch':compared,'mismatch_count':len(mismatches),'max_abs_diff':maxdiff,'mismatch_examples':examples,'all_mismatches':mismatches,'test_payloads_read':0,'buffer_payloads_read':0}
out=BASE/'STAGE_R_COVERAGE_AUDIT_attempt6.json'; out.write_text(json.dumps(report,indent=2)); print(json.dumps({k:v for k,v in report.items() if k!='all_mismatches'},indent=2)); print('REPORT',out)
