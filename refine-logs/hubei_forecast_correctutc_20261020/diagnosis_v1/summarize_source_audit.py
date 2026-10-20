from pathlib import Path
import csv,json,ast
import numpy as np
R=Path(__file__).parent;D=R/'source_probe';rows=list(csv.DictReader((D/'all_validation_key_offsets.csv').open()));assert len(rows)==407 and {int(r['offset_minutes']) for r in rows}=={480};assert {v for r in rows for v in ast.literal_eval(r['target_offset_minutes'])}=={480}
ch=[0,1,2,3,4,5,8,9,10,11,12,13,14];proof=[]
for t in ('202508230900','202511060900'):
 correct=np.load(D/(t+'_offset-8.npz'));wrong=np.load(D/(t+'_offset0.npz'))
 proof.append(dict(declared_BJT=t,actual_saved_image_BJT=t[:8]+'1700',rawUTC_correct_matches_station_npz_max_abs=float(abs(correct['full_crop']-correct['patch16']).max()),rawUTC_same_digit_matches_saved_bank_max_abs=float(abs(wrong['full_crop'][ch]-wrong['saved_bank_truth']).max()),saved_bank_vs_correct_BJT_max_abs=float(abs(correct['full_crop'][ch]-correct['saved_bank_truth']).max())))
events=[]
for init in ('202508230500','202511060500'):
 z=np.load(D/(init+'_with_cot_ghi.npz'))
 for st,(y,x) in {'Niushou':(8,6),'Wangzhai':(9,8),'Mizhuang':(7,10)}.items():
  co=z['canonical_cot'][:,0,y,x];ct=z['saved_truth_cot'][:,0,y,x];cp=z['saved_pred_cot'][:,0,y,x];valid=(z['saved_geometry'][:,2,y,x]>.5)&(z['saved_geometry'][:,0,y,x]>np.cos(np.deg2rad(80)))
  events.append(dict(station=st,declared_init_BJT=init,correct_time_COT_last_frame=float(co[-1]),saved_true_later_COT_last_frame=float(ct[-1]),predicted_later_COT_last_frame=float(cp[-1]),correctly_time_paired_predR_vs_observedR_COT_rmse_this_event=float(np.sqrt(np.mean((cp[valid]-ct[valid])**2))),valid_frames=int(valid.sum())))
result=dict(state='INVALID_FORECAST_TIME_BINDING',validation_sequences=407,offset_minutes=480,local_independent_source_checks=proof,event_COT_diagnostic_only=events,head_input_intervention_state='STOPPED_BEFORE_SCORING_due_to_unmatched_time_keys',same_time_10_percent_result='separate source chain; not invalidated by this forecast-key finding',test_payloads_read=0,buffer_payloads_read=0)
(D/'local_fp64_source_recheck.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
