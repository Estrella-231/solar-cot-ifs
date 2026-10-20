"""Read only two already-used validation events; audit canonical UTC/BJT binding."""
from pathlib import Path
import json,csv,hashlib
from datetime import datetime,timedelta
import numpy as np
root=Path('/home/Data_Pool/zjnu/hubei_cot_independent_20261016');bank=root/'forecast_bank/strict24_same_day_v3_best_epoch10_fulltrainval_20261016';out=root/'source_events_audit_20261020_v2';out.mkdir(exist_ok=False)
keys=list(csv.DictReader((bank/'val/keys.csv').open()));pred=np.load(bank/'val/pred_agri_patch.npy',mmap_mode='r');truth=np.load(bank/'val/true_agri_patch.npy',mmap_mode='r');geom=np.load(bank/'val/future_geometry_patch.npy',mmap_mode='r')
idx=root/'manifests/strict24frame_same_day_20261016_v2/index.csv';ir=list(csv.DictReader(idx.open()));lookup={r['data_paths']:r for r in ir if r['split']=='val'}
align=[]
for k in keys:
 row=lookup.get(k['data_paths']);assert row is not None,'source index binding'
 times=[Path(p).stem[:12] for p in row['aux_paths'].split('|')];dt=[datetime.strptime(t,'%Y%m%d%H%M') for t in times];stamp=datetime.strptime(k['init_time_bjt'],'%Y%m%d%H%M')
 align.append(dict(sample_index=k['sample_index'],bank_init_bjt=k['init_time_bjt'],aux_init_bjt=times[7],offset_minutes=int((dt[7]-stamp).total_seconds()/60),target_offset_minutes=[int((dt[j+8]-datetime.strptime(t,'%Y%m%d%H%M')).total_seconds()/60) for j,t in enumerate(k['target_times_bjt'].split('|'))]))
with (out/'all_validation_key_offsets.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(align[0]));w.writeheader();w.writerows(align)
ch=[0,1,2,3,4,5,8,9,10,11,12,13,14];events=[];source=Path('/home/Data_Pool_3/chenyi/Auxiliary_data/HuBei_AGRI_Preprocessed')
for target in ('202508230900','202511060900'):
 i=next(i for i,k in enumerate(keys) if k['target_times_bjt'].split('|')[-1]==target);k=keys[i];canonical=[];canonical_g=[];checks=[]
 for l,t in enumerate(k['target_times_bjt'].split('|')):
  utc=datetime.strptime(t,'%Y%m%d%H%M')-timedelta(hours=8);u=utc.strftime('%Y%m%d%H%M');p=source/'data'/u[:4]/u[:6]/u[:8]/(u+'.npy');
  if not p.exists():
   canonical.append(np.full((13,16,16),np.nan,np.float32));canonical_g.append(np.full((3,16,16),np.nan,np.float32));checks.append(dict(target=t,canonical_source=str(p),state='missing_source_no_substitution'));continue
  a=np.load(p,mmap_mode='r');v=np.asarray(a[:,103:119,118:134]);canonical.append(v[ch].copy())
  cossoz=np.cos(np.deg2rad(v[18]));cosraa=np.cos(np.deg2rad(v[17]-v[15]));canonical_g.append(np.stack((cossoz,cosraa,v[19])))
  same=source/'data'/t[:4]/t[:6]/t[:8]/(t+'.npy');other=np.load(same,mmap_mode='r');error=float(abs(truth[i,l]-other[ch,103:119,118:134]).max());assert error<1e-3
  patch=list((source/'data16_label'/t[:6]).glob(t+'_*.npz'));pc=None;gc=None
  if patch:
   with np.load(patch[0],allow_pickle=False) as z:
    pc=float(abs(z['patch'][ch]-v[ch]).max());gc=float(abs(np.stack((z['cosSOZ'],z['cosRAA'],z['day_mask']))-canonical_g[-1]).max());assert pc<1e-6
  checks.append(dict(target=t,canonical_source=str(p),misbound_source=str(same),saved_truth_vs_misbound_source_max_abs=error,canonical_source_vs_npz_max_abs=pc,canonical_geometry_vs_npz_max_abs=gc))
 ev=dict(sequence_index=i,init=k['init_time_bjt'],declared_targets=k['target_times_bjt'].split('|'),actual_source_init_bjt=align[i]['aux_init_bjt'],actual_targets_bjt=[(datetime.strptime(t,'%Y%m%d%H%M')+timedelta(hours=8)).strftime('%Y%m%d%H%M') for t in k['target_times_bjt'].split('|')],checks=checks);events.append(ev)
 np.savez_compressed(out/(k['init_time_bjt']+'.npz'),canonical_agri=np.stack(canonical),canonical_geometry=np.stack(canonical_g),saved_truth=np.asarray(truth[i]),saved_pred=np.asarray(pred[i]),saved_geometry=np.asarray(geom[i]))
result=dict(state='TIME_KEY_MISMATCH_CONFIRMED',validation_sequences=len(keys),all_offsets_minutes=sorted(set(r['offset_minutes'] for r in align)),all_target_offsets_minutes=sorted({x for r in align for x in r['target_offset_minutes']}),events=events,test_payloads_read=0,buffer_payloads_read=0)
(out/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='events'},indent=2))
