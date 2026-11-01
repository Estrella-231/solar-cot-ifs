"""Summarize frozen sensitivities and source timing without changing any score."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();r=a.directory/'results';b=a.directory/'payload';rows=list(csv.DictReader((r/'row_diagnostics.csv').open(encoding='utf-8',newline='')));z=np.load(b/'frozen_probe_allval.npz');g=z['gradient_dGHI_dlog1pCOT'];total=g.sum((2,3,4));effect=z['actual']-z['zero_standardized_cot'];repeat=z['actual']-z['repeat_current_cot'];group={}
 for name,mask in [('all',np.ones(len(rows),bool)),('severe_under_drop',np.array([x['severe_under_drop_3seed']=='1' for x in rows])),('helpful_under_up',np.array([x['helpful_under_up_3seed']=='1' for x in rows]))]:
  selected=[x for j,x in enumerate(rows) if mask[j]]
  group[name]=dict(rows=int(mask.sum()),current_COT_lt0p1_rows=sum(float(x['cot_center_current'])<.1 for x in selected),high_kt_ge0p8_rows=sum(float(x['kt'])>=.8 for x in selected),kt_gt1_rows=sum(float(x['kt'])>1 for x in selected),negative_total_gradient_seed_row_fraction=float((total[:,mask]<0).mean()),all_three_seeds_negative_total_gradient_row_fraction=float((total[:,mask]<0).all(0).mean()),mean_total_dGHI_dlog1pCOT=float(total[:,mask].mean()),mean_temporal_total_gradients=g[:,mask].sum((3,4)).mean((0,1)).tolist(),mean_actual_minus_zero_standardized_COT_GHI=float(effect[:,mask].mean()),negative_actual_minus_zero_COT_seed_row_fraction=float((effect[:,mask]<0).mean()),mean_abs_history_repeat_change_GHI=float(np.abs(repeat[:,mask]).mean()))
 cases=json.loads((r/'selected_cases.json').read_text(encoding='utf-8'))['cases'];cr=[]
 for c in cases:
  i=c['validation_index'];cr.append(dict(case_index=c['case_index'],station=c['station'],bjt=c['bjt'],role=c['role'],GHI=c['ghi'],A_mean=c['A_mean'],AC_mean=c['AC_mean'],kt=c['kt'],COT_center=c['cot_center_current'],near_scan5_minus_label15=c['near_scan5_minus_label15'],total_gradient_seed_mean=float(total[:,i].mean()),negative_gradient_seeds=int((total[:,i]<0).sum()),actual_minus_trainmean_COT_GHI=float(effect[:,i].mean()),repeat_current_COT_abs_GHI_change=float(np.abs(repeat[:,i]).mean())))
 with (r/'selected_cases_physical_and_model_summary.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(cr[0]));w.writeheader();w.writerows(cr)
 times=list(csv.DictReader((b/'scan_timing.csv').open(newline='')));timing={}
 for st in ('sili','zhujia'):
  v=np.array([float(x['current_scan_s_relative_target']) for x in times if x['station']==st]);timing[st]=dict(rows=len(v),scan_after_target_min_seconds=float(v.min()),median_seconds=float(np.median(v)),max_seconds=float(v.max()))
 receipt=json.loads((b/'extraction_receipt.json').read_text());frames=receipt['frames'];out=dict(state='PASS_FROZEN_SENSITIVITY_AND_TIMING_SUMMARY',groups=group,scan_timing=timing,all_validation_three_frames_bracket_label=all(x['three_frames_bracket_label']=='1' for x in times),max_saved_input_vs_raw_AGRI_abs_error=max(x['AGRI_3x3_max_abs_error'] for x in frames),max_saved_geometry_vs_nominal_recompute_abs_error=max(x['nominal_geometry_max_abs_error'] for x in frames),max_case_actual_scan_minus_nominal_zenith_deg=max(abs(x['scan_minus_nominal_zenith_deg']) for x in frames),max_case_actual_scan_minus_nominal_cosRAA=max(abs(x['scan_minus_nominal_cosRAA']) for x in frames),max_CPU_prediction_reload_error=max(x['reload_max_abs_GHI'] for x in receipt['models']),all_checkpoints_frozen_and_unchanged=all(x['parameters_frozen'] and x['checkpoint_unchanged'] for x in receipt['models']),full_probe_payload_sha256=hashlib.sha256((b/'frozen_probe_allval.npz').read_bytes()).hexdigest(),test_payloads_read=0,weights_modified=False)
 (r/'frozen_model_summary.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
