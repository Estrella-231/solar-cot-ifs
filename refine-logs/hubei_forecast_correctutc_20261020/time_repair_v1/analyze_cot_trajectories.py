"""Reproduce COT diagnostics; observed-image R is a proxy, CPP a retrieval reference."""
import csv,json,numpy as np
from pathlib import Path
r=Path(__file__).resolve().parent/'runtime_attempt3/fixed_head_diagnosis'
z=np.load(r/'paired_predictions.npz');cm=list(csv.DictReader((r/'cot_metrics.csv').open()));sr=list(csv.DictReader((r/'sequence_diagnostics.csv').open()));out={}
for st in ('Niushou','Wangzhai','Mizhuang'):
 t=z['station']==st;co=z['observed_cot'][t,0,1,1].astype('float64');cp=z['predicted_cot'][t,0,1,1].astype('float64');seq=[s for s in sr if s['station']==st];active=[s for s in seq if float(s['observed_range'])>=3]
 out[st]={'ghi_common_rows':int(t.sum()),'predR_vs_observedR_center_rmse':float(np.sqrt(np.mean((co-cp)**2))),'predR_vs_observedR_center_mae':float(np.mean(abs(co-cp))),'predR_minus_observedR_center_bias':float(np.mean(cp-co)),'eligible_sequences_for_trajectory':len(seq),'active_sequences_observed_range_ge_3':len(active),'active_sequence_std_ratio_median':float(np.median([float(s['predicted_std'])/float(s['observed_std']) for s in active])),'active_sequence_peak_ratio_median':float(np.median([float(s['predicted_peak'])/float(s['observed_peak']) for s in active]))}
 for name in ('predR_vs_CPP','observedR_vs_CPP'):
  a=[s for s in cm if s['station']==st and s['comparison']==name];n=sum(int(s['n']) for s in a);out[st][name]={'overlapping_pixel_lead_records':n,'rmse':float(np.sqrt(sum(int(s['n'])*float(s['rmse'])**2 for s in a)/n)),'bias':sum(int(s['n'])*float(s['bias']) for s in a)/n}
(r/'pooled_cot_and_trajectory_analysis.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
