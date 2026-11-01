from pathlib import Path
import subprocess,json
import numpy as np,pandas as pd
P=Path(__file__).parent
code=r'''
import numpy as np,csv,json
from pathlib import Path
p=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/refine-logs/hubei_same_time_geometry_20261102')
rs=list(csv.DictReader((p/'pack/rows.csv').open()))
keys=list(csv.DictReader((p/'frames/keys.csv').open()));lut={r['time_bjt']:i for i,r in enumerate(keys)}
xx=np.load(p/'pack/x.npy',mmap_mode='r');raw=np.load(p/'frames/raw.npy',mmap_mode='r');cot=np.load(p/'frames/cot_log1p.npy',mmap_mode='r')
ad=[];cd=[];ii=[]
for i,r in enumerate(rs):
 if r['split']=='val' and r['station']=='Niushou' and r['time_bjt'][:8] in ('20250926','20250927') and 1030<=int(r['time_bjt'][-4:])<=1430:
  k=lut[r['time_bjt']];assert keys[k]['split']=='val'
  ad.append(float(abs(xx[i,-1,:15]-raw[k,:15,7:10,5:8]).max()))
  cd.append(float(abs(xx[i,-1,15]-cot[k,7:10,5:8]).max()));ii.append(i)
assert len(ii)==31 and max(ad)==0 and max(cd)==0
print(json.dumps({'selected_scored_rows':len(ii),'pack_AGRI_vs_frame_max_abs':max(ad),'pack_COT_log1p_vs_frame_max_abs':max(cd),'test_payloads_read':0,'state':'EXACT_CURRENT_PACK_MATCH'}))
'''
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','zjnu-hpc','/public/home/slfu/miniconda3/envs/swc/bin/python','-'],input=code.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
assert r.returncode==0,r.stderr.decode(errors='replace')
out=json.loads(r.stdout)
im=np.load(P/'validation_target_images.npz');lut={t:i for i,t in enumerate(im['times_bjt'])}
v=P.parents[2]/'refine-logs/hubei_same_time_geometry_20261102/r_validation'
keys=pd.read_csv(v/'val_keys.csv',dtype={'time_bjt':str});pred=np.load(v/'val_prediction_cot.npy');mask=np.load(v/'val_valid_mask.npy')
dd=[]
for j,t in enumerate(keys.time_bjt):
 if t in lut and t[:8] in ('20250926','20250927') and 1030<=int(t[-4:])<=1430:
  d=np.abs(np.expm1(im['cot_log1p'][lut[t]])-pred[j,0]);dd.extend(d[mask[j,0]].tolist())
out['separate_R_validation_output_check']={'valid_pixel_pairs':len(dd),'max_abs':max(dd),'mean_abs':float(np.mean(dd)),'note':'R validation used float16 autocast; current GHI bank uses FP32 forward. Event fields are exactly the current GHI pack fields, not substituted R validation predictions.'}
(P/'SELECTED_ALIGNMENT_CHECK.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
