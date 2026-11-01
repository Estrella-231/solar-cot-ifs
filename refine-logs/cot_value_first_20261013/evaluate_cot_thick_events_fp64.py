import csv, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'refine-logs/cot_value_first_20261013/results/forecast_cot_val_bank_20261014_v1'
z=np.load(OUT/'forecast_cot_val_bank.npz',allow_pickle=False)
profile=json.loads((OUT/'profile.json').read_text(encoding='utf-8'))
assert profile['split']=='val_only' and profile['test_payloads_read']==0
ref=z['cpp_reference_cot'].astype(np.float64); mask=z['cpp_valid_mask'].astype(bool)
names=('SimVP_to_R','Persistence_to_R','Offline_true_future_AGRI_to_R')
pred=[np.expm1(z[k].astype(np.float64)) for k in ('forecast_log1p_cot','persistence_log1p_cot','target_agri_r_log1p_cot')]
regions={'station_pixel_1x1':(slice(8,9),slice(8,9)), 'station_neighbourhood_3x3':(slice(7,10),slice(7,10)), 'full_patch_16x16':(slice(0,16),slice(0,16))}
stations=('sili','zhujia'); threshold=30.0; rows=[]; summaries={}

for region,(rs,cs) in regions.items():
    m=mask[:,:,:,rs,cs]; y=ref[:,:,:,rs,cs]; xs=[x[:,:,:,rs,cs] for x in pred]
    summaries[region]={}
    for si,station in enumerate(stations):
        summaries[region][station]={}
        for method,x in zip(names,xs):
            mm=m[:,si]; yy=y[:,si]; pp=x[:,si]
            thick=mm&(yy>=threshold); forecast=mm&(pp>=threshold)
            hit=int((thick&forecast).sum()); miss=int((thick&~forecast).sum()); false=int((mm&~thick&forecast).sum()); correct=int((mm&~thick&~forecast).sum())
            pod=hit/(hit+miss) if hit+miss else None
            far=false/(hit+false) if hit+false else None
            csi=hit/(hit+miss+false) if hit+miss+false else None
            # Per-window station-neighbourhood onset: first lead with any CPP-supported pixel >=30.
            lead_ref=np.zeros((mm.shape[0],mm.shape[1]),bool); lead_pred=np.zeros_like(lead_ref)
            for i in range(mm.shape[0]):
                for li in range(mm.shape[1]):
                    v=mm[i,li]
                    if v.any():
                        lead_ref[i,li]=bool(np.any(yy[i,li][v]>=threshold))
                        lead_pred[i,li]=bool(np.any(pp[i,li][v]>=threshold))
            ref_event=lead_ref.any(axis=1); pred_event=lead_pred.any(axis=1)
            event_hit=int((ref_event&pred_event).sum()); event_miss=int((ref_event&~pred_event).sum()); event_false=int((~ref_event&pred_event).sum())
            common=ref_event&pred_event
            onset_errors=[]
            peak_errors=[]
            for i in np.where(ref_event)[0]:
                rlead=int(np.flatnonzero(lead_ref[i])[0]); ridx=np.argwhere(mm[i,rlead]&(yy[i,rlead]>=threshold))
                if ridx.size:
                    vals=yy[i,rlead][mm[i,rlead]&(yy[i,rlead]>=threshold)]
                    pos=ridx[int(np.argmax(vals))]
                    peak_errors.append(float(pp[i,rlead][tuple(pos)]-yy[i,rlead][tuple(pos)]))
                if common[i]:
                    plead=int(np.flatnonzero(lead_pred[i])[0]); onset_errors.append((plead-rlead)*15)
            rec={'thick_pixel_threshold_cot':threshold,'pixel_counts':{'hit':hit,'miss':miss,'false_alarm':false,'correct_negative':correct},
                 'pixel_POD':pod,'pixel_FAR':far,'pixel_CSI':csi,'common_cpp_valid_pixels':int(mm.sum()),
                 'local_event_windows':{'reference_event':int(ref_event.sum()),'predicted_event':int(pred_event.sum()),'detected':event_hit,'missed':event_miss,'false_alarm':event_false,
                    'first_event_lead_error_minutes_mean_detected_only':float(np.mean(onset_errors)) if onset_errors else None,
                    'first_event_lead_error_minutes_median_detected_only':float(np.median(onset_errors)) if onset_errors else None,
                    'reference_peak_pixel_amplitude_error_mean_over_reference_event_windows':float(np.mean(peak_errors)) if peak_errors else None}}
            summaries[region][station][method]=rec
            rows.append({'region':region,'station':station,'method':method,'threshold_cot':threshold,'pixels':int(mm.sum()),'hit':hit,'miss':miss,'false_alarm':false,'correct_negative':correct,'POD':pod,'FAR':far,'CSI':csi,
                         'ref_event_windows':int(ref_event.sum()),'pred_event_windows':int(pred_event.sum()),'event_detected':event_hit,'event_missed':event_miss,'event_false_alarm':event_false,
                         'mean_onset_error_min_detected_only':float(np.mean(onset_errors)) if onset_errors else None,'mean_error_at_reference_peak':float(np.mean(peak_errors)) if peak_errors else None})

result={'state':'FP64_THICK_COT_DIAGNOSTIC_PASS','split':'validation_only','test_payloads_read':0,'reference':'CPP COT retrieval product; not independent truth',
        'threshold_cot':threshold,'support':'all methods share CPP-valid pixel mask; predictions and reference use same station/lead/pixel',
        'event_definition':'within a station pixel/neighbourhood/patch, first lead with any supported pixel COT>=30. Descriptive event-signal onset only; not cloud-object tracking.',
        'overlapping_windows_note':'271 validation windows overlap and span only eight target BJT dates; event-window counts and arrival summaries are descriptive, not independent-sample inference.',
        'regions':summaries}
(OUT/'forecast_cot_thick_events_fp64.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
with (OUT/'forecast_cot_thick_events_fp64.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('state',result['state'])
for region in regions:
 for station in stations:
  print(region,station,{m:(r['pixel_POD'],r['pixel_FAR'],r['local_event_windows']['reference_event'],r['local_event_windows']['detected'],r['local_event_windows']['missed'],r['local_event_windows']['false_alarm'],r['local_event_windows']['first_event_lead_error_minutes_mean_detected_only']) for m,r in summaries[region][station].items()})
