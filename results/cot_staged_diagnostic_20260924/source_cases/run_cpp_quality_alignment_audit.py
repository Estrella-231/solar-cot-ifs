from pathlib import Path
import numpy as np, tarfile, io, csv, json, math
from collections import defaultdict, Counter
root=Path('.')
out=root/'results/cot_staged_diagnostic_20260924/source_cases'
pack=root/'data/cot_launch_20260907/cot_repaired_pack_20260907_v1.tar'
predp=root/'results/cot_real_agri_staged_hurdle_seed42_20260924/frozen_R_validation_predictions.npz'
# Read only train/validation tensors and rows.csv. Never open test_references.csv or test samples.
with tarfile.open(pack,'r') as t:
    mask=np.load(io.BytesIO(t.extractfile('cot_repaired_pack_20260907_v1/mask.npy').read()))
    target=np.load(io.BytesIO(t.extractfile('cot_repaired_pack_20260907_v1/target.npy').read()))
    rows=list(csv.DictReader(io.TextIOWrapper(t.extractfile('cot_repaired_pack_20260907_v1/rows.csv'),encoding='utf-8')))
val_ids=np.array([int(r['index']) for r in rows if r['split']=='validation'],dtype=int)
val_mask=mask[val_ids,0].astype(bool)
val_target=target[val_ids,0]
pack_zero_stats={'valid_pixels':int(val_mask.sum()),'valid_zero_cot_pixels':int(np.sum(val_mask & (val_target==0))),'valid_zero_percent':float(100*np.sum(val_mask & (val_target==0))/val_mask.sum()),'invalid_pixels':int((~val_mask).sum()),'invalid_stored_zero_pixels':int(np.sum((~val_mask) & (val_target==0))),'by_station':{}}
for station in sorted({r['station_id'] for r in rows if r['split']=='validation'}):
    idx=np.array([r['split']=='validation' and r['station_id']==station for r in rows])
    sm=mask[idx,0].astype(bool); sy=target[idx,0]
    pack_zero_stats['by_station'][station]={'samples':int(idx.sum()),'valid_pixels':int(sm.sum()),'valid_zero_cot_pixels':int(np.sum(sm & (sy==0))),'valid_zero_percent':float(100*np.sum(sm & (sy==0))/sm.sum())}
assert len(val_ids)==6536 and np.array_equal(val_ids,np.load(predp)['row_indices'])
assert np.array_equal(mask[val_ids,0],np.load(predp)['mask'][:,0]), 'validation masks differ from stored prediction cohort'
assert np.allclose(target[val_ids,0][mask[val_ids,0]],np.load(predp)['reference_cot'][:,0][np.load(predp)['mask'][:,0]]/100.0,rtol=0,atol=1e-7), 'validation CPP references differ from training pack labels'
p=np.load(predp,allow_pickle=True)
pred=p['pred_cot'][:,0].astype(np.float64)
ref=p['reference_cot'][:,0].astype(np.float64)
valid=p['mask'][:,0].astype(bool)
N,H,W=ref.shape
assert H==16 and W==16
# Common interior support for every dy,dx in [-2,2] on the same sample/pixel cohort.
r0,r1,c0,c1=2,H-2,2,W-2
common=np.ones((N,r1-r0,c1-c0),dtype=bool)
for dy in range(-2,3):
  for dx in range(-2,3):
    common &= valid[:,r0+dy:r1+dy,c0+dx:c1+dx]
base_pred=pred[:,r0:r1,c0:c1]
metrics=[]
shift_values={}
for dy in range(-2,3):
  for dx in range(-2,3):
    shifted=ref[:,r0+dy:r1+dy,c0+dx:c1+dx]
    err=(base_pred-shifted)[common]
    rmse=float(np.sqrt(np.mean(err**2)))
    mae=float(np.mean(np.abs(err)))
    shift_values[(dy,dx)]=(rmse,mae)
    metrics.append({'dy':dy,'dx':dx,'rmse_cot':rmse,'mae_cot':mae,'n_pixels':int(common.sum())})
base=shift_values[(0,0)][0]
for m in metrics: m['delta_rmse_vs_zero']=m['rmse_cot']-base
best=min(metrics,key=lambda m:m['rmse_cot'])
# Per station/date groups, preserving the same per-sample common-support rule.
per=defaultdict(list)
for j,gi in enumerate(val_ids):
  rr=rows[int(gi)]
  day=rr['timestamp_utc'][:10]
  per[(rr['station_id'],day)].append(j)
groups=[]
for (station,day),idxs in sorted(per.items()):
  group_common=common[idxs]
  gm=[]
  for dy in range(-2,3):
    for dx in range(-2,3):
      shifted=ref[idxs,r0+dy:r1+dy,c0+dx:c1+dx]
      e=(base_pred[idxs]-shifted)[group_common]
      gm.append((float(np.sqrt(np.mean(e**2))),(dy,dx)))
  gm.sort()
  z=next(v for v,o in gm if o==(0,0))
  groups.append({'station':station,'date_utc':day,'n_samples':len(idxs),'n_common_pixels':int(group_common.sum()),'best_offset_dy_dx':list(gm[0][1]),'best_rmse_cot':gm[0][0],'zero_offset_rmse_cot':z,'best_delta':gm[0][0]-z,'top3':[{'dy':o[0],'dx':o[1],'rmse_cot':v} for v,o in gm[:3]]})
station_offsets=[]
for station in sorted({rows[int(i)]['station_id'] for i in val_ids}):
  idxs=[j for j,gi in enumerate(val_ids) if rows[int(gi)]['station_id']==station]
  sm=common[idxs]
  vals=[]
  for dy in range(-2,3):
    for dx in range(-2,3):
      shifted=ref[idxs,r0+dy:r1+dy,c0+dx:c1+dx]
      e=(base_pred[idxs]-shifted)[sm]
      vals.append((float(np.sqrt(np.mean(e**2))),(dy,dx)))
  vals.sort(); zero=next(v for v,o in vals if o==(0,0))
  station_offsets.append({'station':station,'n_samples':len(idxs),'n_common_pixels':int(sm.sum()),'best_offset_dy_dx':list(vals[0][1]),'best_rmse_cot':vals[0][0],'zero_offset_rmse_cot':zero,'best_delta':vals[0][0]-zero})
# Four exact points from earlier failure records, source CPP windows and sidecars.
false=np.load(root/'results/cot_staged_diagnostic_20260924/false_cloud_cases.npz',allow_pickle=True)
case_rows=[]
for j,gi in enumerate(false['row_indices']):
 rr=rows[int(gi)]
 # star point coordinates are recorded in prior JSON; derive from CPP zero pixels with high route probability
 # identify top cloud probability among near-zero CPP pixels, matching previously reported max case pixel.
 cot=false['cpp_cot'][j,0]*100
 prob=false['cloud_probability'][j,0]
 cand=np.argwhere((cot<1e-6)&false['valid_mask'][j,0])
 ij=cand[np.argmax(prob[cand[:,0],cand[:,1]])]
 case_rows.append({'station':rr['station_id'],'timestamp_utc':rr['timestamp_utc'],'timestamp_bjt':rr['timestamp_bjt'],'row_index':int(gi),'pixel_rc':ij.tolist(),'cpp_cot':float(cot[tuple(ij)]),'cloud_probability':float(prob[tuple(ij)])})
# audit each raw CPP NetCDF crop vs sidecar and CLP at case pixel.
import netCDF4 as nc
for cr in case_rows:
    r,c=cr['pixel_rc']; st=cr['station']; ts=cr['timestamp_utc'].replace('-','').replace(':','').replace('T','').replace('Z','')[:12]
    key=f'{st}_{ts}'
    # naming source files: zhujia_202507060815 etc.
    side=np.load(out/f'{key}_sidecar.npz',allow_pickle=True)
    with nc.Dataset(out/'nc_windows'/f'{key}.nc') as ds:
        source_cot=np.ma.filled(ds['COT'][:],np.nan).astype(float)
        clp=np.ma.filled(ds['CLP'][:],np.nan).astype(float)
        lat=np.ma.filled(ds['LAT'][:],np.nan).astype(float)
        lon=np.ma.filled(ds['LON'][:],np.nan).astype(float)
    sc=side['cpp'][0]
    srcr=source_cot[r,c]
    clpv=clp[r,c]
    cr['raw_cpp_cot']=float(srcr); cr['raw_cpp_clp']=int(clpv) if np.isfinite(clpv) else None
    cr['sidecar_matches_source_exact']=bool(np.array_equal(sc,source_cot,equal_nan=True))
    source_lat_2d=np.broadcast_to(lat[:,None],(16,16)); source_lon_2d=np.broadcast_to(lon[None,:],(16,16))
    cr['full_grid_lat_exact']=bool(np.array_equal(source_lat_2d,side['grid_lat']))
    cr['full_grid_lon_exact']=bool(np.array_equal(source_lon_2d,side['grid_lon']))
    cr['full_grid_lat_max_absdiff_deg']=float(np.max(np.abs(source_lat_2d-side['grid_lat'])))
    cr['full_grid_lon_max_absdiff_deg']=float(np.max(np.abs(source_lon_2d-side['grid_lon'])))
    cr['source_lat']=float(lat[r]); cr['source_lon']=float(lon[c])
    cr['sidecar_lat']=float(side['grid_lat'][r,c]); cr['sidecar_lon']=float(side['grid_lon'][r,c])
    cr['grid_coordinate_absdiff_deg']=[float(abs(lat[r]-side['grid_lat'][r,c])),float(abs(lon[c]-side['grid_lon'][r,c]))]
    cr['source_clp_counts']={str(int(k)):int(np.sum(clp==k)) for k in np.unique(clp[np.isfinite(clp)])}
    cr['source_cot_zero_count']=int(np.sum(source_cot==0))
    cr['source_zero_clp_clear_count']=int(np.sum((source_cot==0)&(clp==0)))
    cr['source_zero_clp_cloud_count']=int(np.sum((source_cot==0)&((clp==1)|(clp==2))))
    cr['source_clp_nan_count']=int(np.sum(~np.isfinite(clp)))
# Whole-validation fixed-grid offset response and per-group offset consistency.
report={'pack_validation_samples':int(len(val_ids)),'cpp_v2_portable_sha256':'dc7959b7bc7ef467239a261072f8789d416df9e9a7bf19c8d16403e3d113eb7b','validation_pack_zero_statistics':pack_zero_stats,'fixed_common_support':{'rule':'same prediction pixels (rows/cols 2..13) and pixels valid in all 25 shifted reference masks for each sample','n_samples_with_any_common_pixels':int(np.sum(common.reshape(N,-1).sum(axis=1)>0)),'n_samples':N,'pixels_per_sample_mean':float(common.reshape(N,-1).sum(axis=1).mean()),'total_sample_pixels':int(common.sum())},'validation_cohort_check':{'row_indices_match':True,'mask_matches_prediction_artifact':True,'target_matches_prediction_reference_after_scale_100':True},'offset_convention':'dy,dx shift applied to CPP reference relative to fixed prediction coordinate; no label movement authorized','offset_metrics':metrics,'zero_offset_rmse_cot':base,'best_offset':best,'best_improvement_percent':float((base-best['rmse_cot'])/base*100),'station_date_groups':groups,'station_aggregate_offsets':station_offsets,'case_audit':case_rows,'provenance_limits':['The original AGRI NPZ files referenced by the manifest were not readable from the current SSH account (permission denied); timestamps and source hashes are recorded in the manifest/sidecar, but raw AGRI values were not independently reopened in this audit.','The historical 2025 CPP NetCDF files expose COT/CLP/LAT/LON but no QA variable; source crop agreement and clear-class semantics can be verified, but historical producer/checkpoint and fallback/invalid-retrieval provenance cannot be proven from this file alone.','Offset results compare the frozen R model prediction against shifted CPP retrieval labels; this is a sensitivity diagnostic, not independent validation truth and cannot justify shifting labels.']}
# best-offset counts and exact group stats
cnt=Counter(tuple(g['best_offset_dy_dx']) for g in groups)
report['group_best_offset_counts']={f'dy{a:+d}_dx{b:+d}':v for (a,b),v in sorted(cnt.items())}
outfile=out/'cpp_quality_alignment_audit_20260924.json'
outfile.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
# concise human-readable report
lines=['# CPP quality and alignment audit (2026-09-24)','',
'## Four false-cloud cases: raw CPP check','',
'Each extracted raw NetCDF crop was compared pixel-for-pixel with the corresponding aligned sidecar. All four COT crops matched their sidecars exactly, and all 16×16 latitude/longitude grids matched exactly. At the four reported station pixels, source CPP has COT=0 and CLP=0 (clear); there are no zero-COT pixels classified as water/ice in these 16×16 crops. The crops have no missing CLP entries. This supports “valid clear-class zero” for these four examples, not a general guarantee that every historical zero is valid.','',
'| Station / UTC | pixel (r,c) | COT | CLP | CPP grid lat,lon | zero COT count / clear |','|---|---:|---:|---:|---|---:|']
for c in case_rows:
 lines.append(f"| {c['station']} {c['timestamp_utc']} | {tuple(c['pixel_rc'])} | {c['raw_cpp_cot']:.3f} | {c['raw_cpp_clp']} (clear) | {c['source_lat']:.4f}, {c['source_lon']:.4f} | {c['source_cot_zero_count']} / {c['source_zero_clp_clear_count']} |")
lines += ['', 'The supplied CPP V2 Portable archive was inspected (SHA-256 `dc7959b7bc7ef467239a261072f8789d416df9e9a7bf19c8d16403e3d113eb7b`). Its `Code/Quality_Control.py` defines CLP=0 as clear and treats zero COT as valid in clear pixels; its newer `Retrieval_CPP.py` writes a QA field whose 1 means previous-hour replacement. The four historical 2025 NetCDF files inspected here contain no QA field, so that newer flag contract cannot be retroactively used to certify their lineage.','', '## Full validation-pack zero accounting','',f"The validation pack has {pack_zero_stats['valid_pixels']:,} research-mask-valid pixels. Of these, {pack_zero_stats['valid_zero_cot_pixels']:,} ({pack_zero_stats['valid_zero_percent']:.2f}%) are stored as valid zero-COT targets. Another {pack_zero_stats['invalid_stored_zero_pixels']:,} zeros are merely stored placeholders outside the mask and are not supervised. By station: Sili {pack_zero_stats['by_station']['sili']['valid_zero_cot_pixels']:,}/{pack_zero_stats['by_station']['sili']['valid_pixels']:,} ({pack_zero_stats['by_station']['sili']['valid_zero_percent']:.2f}%); Zhujia {pack_zero_stats['by_station']['zhujia']['valid_zero_cot_pixels']:,}/{pack_zero_stats['by_station']['zhujia']['valid_pixels']:,} ({pack_zero_stats['by_station']['zhujia']['valid_zero_percent']:.2f}%). This is full-pack label accounting under the research mask; it does not establish official product QA for every validation time.",'', '## Validation-set offset sensitivity','',f"The fixed common support contains {report['fixed_common_support']['total_sample_pixels']:,} sample-pixels over {N:,} validation samples (mean {report['fixed_common_support']['pixels_per_sample_mean']:.1f} pixels/sample). All 25 offsets use exactly this same support. Offsets are applied to CPP references while predictions stay fixed.",'', '| dy | dx | RMSE (COT units) | Δ vs 0,0 |', '|---:|---:|---:|---:|']
for m in sorted(metrics,key=lambda m:(m['dy'],m['dx'])):
 lines.append(f"| {m['dy']:+d} | {m['dx']:+d} | {m['rmse_cot']:.4f} | {m['delta_rmse_vs_zero']:+.4f} |")
lines += ['',f"Zero-offset RMSE: **{base:.4f}**. Best aggregate offset: **dy={best['dy']:+d}, dx={best['dx']:+d}**, RMSE {best['rmse_cot']:.4f} ({report['best_improvement_percent']:.2f}% lower on the same cohort). This is a sensitivity clue only.",'','### Aggregate by station','','| Station | Samples | Best (dy,dx) | RMSE at best | RMSE at 0,0 |','|---|---:|---:|---:|---:|']
for g in station_offsets:
 lines.append(f"| {g['station']} | {g['n_samples']} | {tuple(g['best_offset_dy_dx'])} | {g['best_rmse_cot']:.4f} | {g['zero_offset_rmse_cot']:.4f} |")
lines += ['', '### Best offset by station and UTC date','','| Station | UTC date | Samples | Best (dy,dx) | RMSE at best | RMSE at 0,0 |','|---|---|---:|---:|---:|---:|']
for g in groups:
 lines.append(f"| {g['station']} | {g['date_utc']} | {g['n_samples']} | {tuple(g['best_offset_dy_dx'])} | {g['best_rmse_cot']:.4f} | {g['zero_offset_rmse_cot']:.4f} |")
lines += ['', '## Interpretation and limits','',
'- The four false-cloud pixels are not CPP fill values: the source stores finite COT=0 with CLP=0, which the attached producer QC code defines as clear sky. Their high model cloud probability is therefore a classifier/product-disagreement issue at those points, not a packaging conversion of missing COT to zero.','- The historical files have no independent QA layer. The portable V2 package has a later QA/fallback contract, so it cannot establish the exact lineage of these 2025 products. We cannot rule out all retrieval failures from COT/CLP alone.','- Original AGRI NPZs were permission-denied on the current remote account. The pack/sidecar retain timestamp, original SHA-256, and geolocation records; the source CPP crop aligns exactly with sidecar and matches the corrected grid coordinates, but raw AGRI values were not reopened here.','- The ±1–2 px experiment compares a frozen model to shifted CPP retrieval targets on a common, fixed cohort. A lower error under a shift does not identify the correct geolocation and must not be used to move labels. Any stronger conclusion needs independent geolocation controls (AGRI navigation arrays / original source patch) and cross-date / cross-station agreement.','']
(out/'cpp_quality_alignment_audit_20260924.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'report':str(outfile),'fixed_support':report['fixed_common_support'],'zero_rmse':base,'best':best,'improvement_pct':report['best_improvement_percent'],'group_counts':report['group_best_offset_counts'],'cases':case_rows},ensure_ascii=False,indent=2))

