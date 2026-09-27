import json,sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from torch.nn import functional as F
ROOT=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907'); EXP=ROOT/'experiments/regional_station_cot_20260925'
BANK=EXP/'regional_ghi_pilot_bank_val512_20260927_v1/val'; CACHE=EXP/'agri_cpp_pair_bank_v1'; EXT=EXP/'r_pair_audit_20260927_v1/extracted_arrays/r256_full_exploratory_seed42_batch512_node22'
OUT=EXP/'sili_transport_cloud_object_val512_20260927_v1';OUT.mkdir(exist_ok=True)
seq=np.load(BANK/'sequence_indices.npy');hist=np.load(BANK/'cot_local_history.npy',mmap_mode='r');simvp=np.load(BANK/'cot_local_forecast.npy',mmap_mode='r')
truth=np.load(EXT/'reference_log1p_cot.npy',mmap_mode='r');valid_ref=np.load(EXT/'valid_mask.npy',mmap_mode='r')
summary=json.loads((EXP/'sili_transport_variants_val512_20260927_v1/summary.json').read_text())
motion={int(x['sequence']):x for x in json.loads((EXP/'sili_transport_variants_val512_20260927_v1/motion_vectors.json').read_text())}
lookup={}
for sh in json.loads((CACHE/'index.json').read_text())['shards']:
 d=CACHE/sh['directory']
 for r in json.loads((d/'records.json').read_text()):
  if r['split']=='val' and r['included_for_R']:lookup[r['utc']]=(d,int(r['array_row']))
sc=json.loads((ROOT/'configs/s_frozen_hunan_seed42.json').read_text());rows=[r for r in __import__('csv').DictReader(open(sc['manifest'])) if r['split']=='val']
rr,cc=163,148;sl=(slice(rr-32,rr+32),slice(cc-32,cc+32));torch.set_num_threads(2)
def warp(field,shift,lead):
 h,w=field.shape;yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij');sy=yy-lead*float(shift[0]);sx=xx-lead*float(shift[1])
 grid=torch.stack((2*sx/(w-1)-1,2*sy/(h-1)-1),-1).float()[None]
 source=torch.tensor(np.asarray(field).copy(),dtype=torch.float32)
 moved=F.grid_sample(torch.expm1(source[None,None]),grid,mode='bilinear',padding_mode='zeros',align_corners=True)[0,0].numpy()
 inside=((sy>=0)&(sy<=h-1)&(sx>=0)&(sx<=w-1)).numpy()
 return moved,inside
names=['persistence','global_last1','global_med4','global_med7','local64_med7','local32_med7','local16_med7','simvp']
acc={n:np.zeros((16,8),np.float64) for n in names}
# columns: TP, FP, FN, true frames, predicted >=10 frames with thick reference, thick-lead count, centroid distance sum, centroid n
for j,k in enumerate(seq):
 row=rows[int(k)];stamps=[Path(x).stem for x in row['data_relpaths'].split('|')][8:]
 mv=motion[int(k)];vec={'global_last1':mv['last1'],'global_med4':mv['med4'],'global_med7':mv['med7'],'local64_med7':mv['local64'],'local32_med7':mv['local32'],'local16_med7':mv['local16']}
 base=hist[j,0,-1,0];pred={n:np.repeat(np.expm1(base)[None],16,axis=0) for n in names if n!='simvp'};supports={n:np.ones((16,64,64),bool) for n in vec}
 for n,v in vec.items():
  for ld in range(16):pred[n][ld],supports[n][ld]=warp(base,v,ld+1)
 pred['simvp']=np.expm1(simvp[j,0,:,0])
 ref=np.full((16,64,64),np.nan,np.float32);mask=np.zeros((16,64,64),bool)
 for ld,t in enumerate(stamps):
  if t in lookup:
   d,ix=lookup[t];ref[ld]=np.expm1(truth[ix,0][sl]);mask[ld]=valid_ref[ix,0][sl]
 common=mask.copy()
 for s in supports.values():common &= s
 for ld in range(16):
  ok=common[ld];truth30=(ref[ld]>=30)&ok;has=truth30.any()
  if not ok.any():continue
  for n in names:
   pm30=(pred[n][ld]>=30)&ok;tp=np.logical_and(pm30,truth30).sum();fp=pm30.sum()-tp;fn=truth30.sum()-tp
   ac=acc[n][ld];ac[0]+=tp;ac[1]+=fp;ac[2]+=fn;ac[3]+=int(has)
   if has:
    ac[5]+=1;ac[4]+=int((pred[n][ld][truth30]>=10).any())
    if pm30.any():
     cy,cx=np.where(truth30);py,px=np.where(pm30);ac[6]+=np.hypot(cy.mean()-py.mean(),cx.mean()-px.mean());ac[7]+=1
  if j%64==0 and ld==15: print('SEQUENCE',j,flush=True)
records=[]
for n in names:
 for ld,a in enumerate(acc[n]):
  tp,fp,fn,tf,hit10,nth,cd,cn=a
  records.append({'method':n,'lead_min':15*(ld+1),'thick_pixel_precision_cot30':tp/(tp+fp) if tp+fp else None,'thick_pixel_recall_cot30':tp/(tp+fn) if tp+fn else None,'thick_pixel_iou_cot30':tp/(tp+fp+fn) if tp+fp+fn else None,'frames_with_reference_thick_cloud':int(nth),'frames_with_pred_cot_ge10_inside_true_thick':int(hit10),'centroid_error_pixels_when_pred_and_ref_thick_exist':cd/cn if cn else None,'centroid_frames':int(cn)})
out={'cohort_sequences':len(seq),'test_used':False,'thresholds':{'cpp_reference_thick_cot':30,'prediction_masks_cot':30,'recovery_at_true_thick_regions_cot':10},'scoring':'All methods scored on exact intersection of CPP valid pixels and all translation in-domain supports. CPP remains retrieval reference; sequences/leads overlap.','records':records}
(OUT/'metrics.json').write_text(json.dumps(out,indent=2))
print('COMPLETE',len(seq),OUT,flush=True)
