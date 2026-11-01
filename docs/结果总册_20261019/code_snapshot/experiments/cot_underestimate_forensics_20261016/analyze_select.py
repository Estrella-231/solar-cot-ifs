"""Read saved validation predictions, select prespecified failure/control cases."""
import csv,json,hashlib,argparse
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def csvout(p,rows):
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 run=a.repo/'experiments/r_geometry_downstream_ghi_20261016/results/paired_three_arm_v1';arc=run/'analysis/paired_validation_scoring.npz';z=np.load(arc);v=np.load(run/'validation_inputs.npz');rows=list(csv.DictReader((run/'validation_rows.csv').open(encoding='utf-8',newline='')))
 assert len(rows)==4263 and {r['split'] for r in rows}=={'val'} and np.array_equal(z['pack_index'],v['pack_index'])
 pred=z['prediction'];y=z['truth'];A=pred[:,0];C=pred[:,2];d=C-A;need=y[None,:]-A;ds=(C-y)**2-(A-y)**2
 station=np.asarray([r['station'] for r in rows]);day=np.asarray([(datetime.fromisoformat(r['target_time_utc'])+timedelta(hours=8)).date().isoformat() for r in rows]);clear=np.array([float(r['clear']) for r in rows]);kt=y/clear
 cot=np.expm1(v['cot_corrected_log1p'][:,:,0,1,1]);logcot=v['cot_corrected_log1p'][:,-1,0,1,1];zen=np.rad2deg(np.arccos(np.clip(v['geometry'][:,-1,0,1,1],-1,1)))
 bad=((need>20)&(d<-5)).all(0);good=((need>20)&(d>5)&(ds<0)).all(0)
 summaries={}
 for ai,name in [(1,'old_R'),(2,'corrected_R')]:
  c=pred[:,ai];dd=c-A;ss=(c-y)**2-(A-y)**2;rr={}
  for g,m in [('all',np.ones(A.shape,bool)),('A_under',need>0),('A_over',need<0)]:
   rr[g]=dict(seed_rows=int(m.sum()),mean_adjustment=float(dd[m].mean()),correct_direction_fraction=float((dd[m]*need[m]>0).mean()),squared_error_improved_fraction=float((ss[m]<0).mean()))
  summaries[name]=rr
 native={};source=a.repo/'refine-logs/cot_value_first_20261013/tmp/unique_target_labels_train_val.csv'
 with source.open(encoding='utf-8',newline='') as f:
  for r in csv.DictReader(f):
   if r['new_split']!='val':continue
   for t,s in zip(r['label_record_times_bjt'].split('|'),r['raw_ghi_5min_wm2'].split('|')):
    if t[:10]!=(datetime.fromisoformat(r['target_time_utc'])+timedelta(hours=8)).date().isoformat():continue
    try:x=float(s)
    except ValueError:continue
    if np.isfinite(x):
     k=(r['station'],t)
     if k in native:assert abs(native[k]-x)<1e-8
     native[k]=x
 label_err=[];near_scan=np.full(len(rows),np.nan);within=np.full(len(rows),np.nan)
 for i,r in enumerate(rows):
  t=datetime.fromisoformat(r['target_time_utc'])+timedelta(hours=8);vv=[native.get((r['station'],(t-timedelta(minutes=k)).isoformat(sep=' '))) for k in (10,5,0)]
  assert all(x is not None for x in vv);label_err.append(abs(float(np.float32(np.mean(vv)))-y[i]));assert label_err[-1]<.0002
  nxt=native.get((r['station'],(t+timedelta(minutes=5)).isoformat(sep=' ')))
  if nxt is not None:near_scan[i]=nxt-y[i]
  within[i]=vv[-1]-vv[0]
 records=[]
 for i,r in enumerate(rows):
  rec=dict(validation_index=i,pack_index=int(z['pack_index'][i]),station=r['station'],utc=r['target_time_utc'],bjt=(datetime.fromisoformat(r['target_time_utc'])+timedelta(hours=8)).isoformat(sep=' '),date_bjt=day[i],ghi=float(y[i]),clear=float(clear[i]),kt=float(kt[i]),A_mean=float(A[:,i].mean()),AC_mean=float(C[:,i].mean()),mean_delta=float(d[:,i].mean()),delta_SSE_mean=float(ds[:,i].mean()),under_drop_seed_count=int(((need[:,i]>0)&(d[:,i]<0)).sum()),severe_under_drop_3seed=int(bad[i]),helpful_under_up_3seed=int(good[i]),zenith_deg=float(zen[i]),cot_center_current=float(cot[i,-1]),cot_center_previous=float(cot[i,-2]),cot_center_earlier=float(cot[i,0]),native5_change_within_label=float(within[i]),near_scan5_minus_label15='' if not np.isfinite(near_scan[i]) else float(near_scan[i]),suspect_source_zero=int(r['suspect_daylight_long_zero']))
  for j,s in enumerate((42,43,44)):rec.update({f'A_{s}':float(A[j,i]),f'AC_{s}':float(C[j,i])})
  records.append(rec)
 csvout(a.output/'row_diagnostics.csv',records)
 strata=[]
 for name,m in [('all',np.ones(len(rows),bool)),('severe_under_drop',bad),('helpful_under_up',good),('center_COT_ge10',cot[:,-1]>=10),('high_kt_ge0.8',kt>=.8),('solar_altitude_lt20',zen>70)]:
  mm=np.broadcast_to(m,A.shape);nn=m&np.isfinite(near_scan)
  strata.append(dict(group=name,rows=int(m.sum()),station_dates=len(set(zip(station[m],day[m]))),mean_AC_minus_A=float(d[mm].mean()),mean_delta_SSE=float(ds[mm].mean()),near_scan_proxy_available=int(nn.sum()),near_scan5_lower_by_gt50_fraction=None if not nn.any() else float((near_scan[nn]<-50).mean()),within_interval_abs_change_gt100_fraction=float((np.abs(within[m])>100).mean())))
 cases=[];pairs=[];used=set()
 def add(i,role,pair=None):
  if i not in used:
   cases.append(dict(**records[i],role=role,pair=pair));used.add(i)
  return next(j for j,c in enumerate(cases) if c['validation_index']==i)
 for st in ('sili','zhujia'):
  pool=np.where(bad&(station==st)&(zen<70)&(clear>200))[0];pool=sorted(pool,key=lambda i:(-ds[:,i].mean(),rows[i]['target_time_utc']));dates=set();anchors=[]
  for i in pool:
   if day[i] in dates:continue
   anchors.append(i);dates.add(day[i])
   if len(anchors)==3:break
  for i in anchors:
   ii=add(i,'severe_under_drop');candidates=[j for j in np.where(good&(station==st)&(clear>200))[0] if day[j]!=day[i] and j not in used]
   def dist(j):
    mi=datetime.fromisoformat(rows[i]['target_time_utc']).month;mj=datetime.fromisoformat(rows[j]['target_time_utc']).month;md=min(abs(mi-mj),12-abs(mi-mj))
    return abs(float(zen[j]-zen[i]))/10+abs(float(logcot[j]-logcot[i]))/.5+md/3
   j=min(candidates,key=lambda j:(dist(j),rows[j]['target_time_utc']));jj=add(j,'matched_helpful_under_up',ii);pairs.append(dict(bad_case=ii,control_case=jj,matching_distance=float(dist(j)),zenith_difference_deg=float(zen[j]-zen[i]),log1p_COT_difference=float(logcot[j]-logcot[i]),same_station=True,different_BJT_dates=True))
 prior=a.repo/'experiments/monthly_R_station16_20261013/cot_shadow_projection_20261014';shadow=list(csv.DictReader((prior/'shadow_metrics.csv').open(encoding='utf-8',newline='')));lookup={(r['station'],r['target_time_utc']):i for i,r in enumerate(rows)};cth=[]
 oldgrid=json.loads((prior/'projection_receipt.json').read_text())['grid_static_sha256'];newgrid=json.loads((a.repo/'experiments/r_geometry_downstream_ghi_20261016/audit/paired_r_cot_bank_manifest.json').read_text())['grid_static_sha256'];assert oldgrid==newgrid
 for s in shadow:
  i=lookup[(s['station'],s['target_time_utc'])];assert float(np.float32(float(s['observed_ghi_wm2'])))==y[i];ci=add(i,'existing_CTH_anchor');cth.append(dict(case_index=ci,validation_index=i,station=s['station'],utc=s['target_time_utc'],current_GHI=float(y[i]),current_A_mean=float(A[:,i].mean()),current_AC_mean=float(C[:,i].mean()),current_delta_SSE_mean=float(ds[:,i].mean()),current_helpful_seed_count=int((ds[:,i]<0).sum()),shadow_nearest_km=float(s['shadow_nearest_distance_km_main']),shadow_nearest_inclusive_QC_km=float(s['nearest_shadow_distance_km_inclusive_QC']),coverage4km=s['coverage_4km'],CTH_station_m=float(s['CTH_at_station_m']),CTH_quality=int(s['CTH_station_quality_low_bits']),source_projection_sha256=sha(prior/'projection_receipt.json')))
 csvout(a.output/'existing_CTH_current_model_join.csv',cth)
 for ci,c in enumerate(cases):
  c['case_index']=ci;t=datetime.fromisoformat(c['bjt']);c['native_5min']=[dict(bjt=(t+timedelta(minutes=k)).isoformat(sep=' '),ghi=native.get((c['station'],(t+timedelta(minutes=k)).isoformat(sep=' ')))) for k in range(-60,61,5) if (t+timedelta(minutes=k)).date()==t.date()]
 dump(a.output/'selected_cases.json',dict(state='SELECTED_BEFORE_RAW_CLOUD_READ',cases=cases,pairs=pairs,source_predictions_sha256=sha(arc),source_validation_rows_sha256=sha(run/'validation_rows.csv'),test_payloads_read=0))
 dump(a.output/'summary.json',dict(state='PASS_SAVED_PREDICTIONS_DIAGNOSTIC',validation_rows=len(rows),seeds=[42,43,44],direction=summaries,severe_under_drop_rows=int(bad.sum()),helpful_under_up_rows=int(good.sum()),severe_station_dates=len(set(zip(station[bad],day[bad]))),strata=strata,label_mean_max_abs_vs_saved_FP32=float(max(label_err)),source_native_labels_sha256=sha(source),selected_case_count=len(cases),selected_pairs=len(pairs),main_scope='descriptive validation diagnostics; three seeds share rows; independent heads do not identify a causal effect',test_payloads_read=0))
 print(json.dumps(dict(severe_rows=int(bad.sum()),pairs=pairs,cases=[(c['station'],c['utc'],c['role']) for c in cases],strata=strata),indent=2))
if __name__=='__main__':main()
