"""Exact-key CPP validation-reference lookup; never fill missing CPP as clear sky."""
import argparse,json,csv,hashlib
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--diagnostic',type=Path,required=True);a=p.parse_args();base=a.root/'experiments/r_solar_geometry_corrected_20261015';rp=base/'data/paired_common_v1/rows_supervised.csv';rows=list(csv.DictReader(rp.open()));fp=base/'results/paired_v1/corrected/validation_predictions.npz';z=np.load(fp,allow_pickle=False);lookup={}
 for j,i in enumerate(z['array_indices'].astype(int)):
  r=rows[i];assert r['split']=='val';lookup[(r['station'],r['target_time_utc'])]=j
 cases=json.loads((a.diagnostic/'selected_cases.json').read_text())['cases'];out=np.full((len(cases),3,3,3),np.nan,np.float32);mask=np.zeros(out.shape,bool);records=[]
 for ci,c in enumerate(cases):
  for ti,minutes in enumerate((-30,-15,0)):
   t=(datetime.fromisoformat(c['utc'])+timedelta(minutes=minutes)).isoformat(sep=' ');j=lookup.get((c['station'],t));available=j is not None
   if available:
    m=z['mask'][j,0,7:10,7:10].astype(bool);v=z['reference_cot'][j,0,7:10,7:10];m=m&np.isfinite(v);mask[ci,ti]=m;out[ci,ti]=np.where(m,v,np.nan)
   records.append(dict(case_index=ci,station=c['station'],utc=t,offset_minutes=minutes,exact_R_validation_reference_key=available,valid_3x3_pixels=int(mask[ci,ti].sum()),center_CPP_reference=None if not mask[ci,ti,1,1] else float(out[ci,ti,1,1])))
 np.savez_compressed(a.diagnostic/'results_v1/cpp_reference_cases.npz',reference_COT=out,valid=mask)
 report=dict(state='EXACT_CPP_VALIDATION_REFERENCE_JOIN_COMPLETE',records=records,exact_frame_keys=sum(r['exact_R_validation_reference_key'] for r in records),requested_frame_keys=len(records),current_target_center_reference_cases=sum(r['offset_minutes']==0 and r['center_CPP_reference'] is not None for r in records),source_R_rows_sha256=hashlib.sha256(rp.read_bytes()).hexdigest(),source_validation_reference_sha256=hashlib.sha256(fp.read_bytes()).hexdigest(),missing_filled_as_clear=False,reference_is_independent_truth=False,test_payloads_read=0)
 (a.diagnostic/'results_v1/cpp_reference_join.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
