"""Repair CPP keys only; preserve all saved frozen model output tensors."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'refine-logs/cot_value_first_20261013/results/forecast_cot_val_bank_20261014_v1/forecast_cot_val_bank.npz'
OUT=HERE/'repaired_existing_271_windows'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def score(x,y,m):
    a=x[m].astype(float);b=y[m].astype(float);e=b-a
    return dict(n=int(a.size),rmse=float(np.sqrt(np.mean(e**2))),mae=float(np.mean(abs(e))),
                bias=float(np.mean(e)),pearson_r=float(np.corrcoef(a,b)[0,1]))

def main():
    rows=HERE/'sources/rows.csv';original=HERE/'sources/validation_predictions.npz'
    assert sha(rows)=='b270713590f6af793b53df73d27e01f3f62ad8d7e60bb9a8eaace5c43fb2ebcb'
    assert sha(original)=='08c9cc92df549f12412873e9b8e50f308f35f79451693fa10b229a3346c15785'
    assert sha(OLD)=='fd7a431148ebf301140614bdafa8f05e09e99bc5c83a34bb0c456d6cf88b27ac'
    with rows.open(encoding='utf-8',newline='') as f: sup=[r for r in csv.DictReader(f) if r['state']=='SUPERVISED']
    val=[r for r in sup if r['split']=='val']
    with np.load(original,allow_pickle=False) as z: rdata={k:z[k].copy() for k in z.files}
    ids=rdata['sample_indices'].astype(int)
    assert len(ids)==len(val)==1398 and len(sup)==16812
    assert set(ids)=={i for i,r in enumerate(sup) if r['split']=='val'}
    assert all(int(r['array_index'])==i for i,r in enumerate(sup))
    refs=rdata['reference_cot'][:,0];masks=rdata['mask'][:,0].astype(bool)
    saved={int(i):j for j,i in enumerate(ids)}
    correct={(sup[i]['station'],sup[i]['target_time_utc']):(j,i) for i,j in saved.items()}
    wrong={(r['station'],r['target_time_utc']):(saved[i],i) for i,r in enumerate(val) if i in saved}
    with np.load(OLD,allow_pickle=False) as z: bank={k:z[k].copy() for k in z.files}
    shape=bank['cpp_reference_cot'].shape
    newref=np.full(shape,np.nan,np.float32);newmask=np.zeros(shape,bool)
    old_matches=0;examples=[];correct_key_records=0
    for wi in range(shape[0]):
        for si,sta in enumerate(('sili','zhujia')):
            for li,t in enumerate(bank['target_time_utc'][wi].astype(str)):
                key=(sta,t)
                if key in wrong:
                    j,i=wrong[key]
                    assert np.array_equal(bank['cpp_valid_mask'][wi,si,li],masks[j])
                    assert np.array_equal(bank['cpp_reference_cot'][wi,si,li],refs[j],equal_nan=True)
                    old_matches+=1
                    if len(examples)<6 and not any(e['assigned_key']==list(key) for e in examples):
                        examples.append(dict(assigned_key=list(key),actual_CPP_source_key=[sup[i]['station'],sup[i]['target_time_utc']],misused_index=i))
                else:assert not bank['cpp_valid_mask'][wi,si,li].any()
                if key in correct:
                    j,i=correct[key];newref[wi,si,li]=refs[j];newmask[wi,si,li]=masks[j];correct_key_records+=1
    assert old_matches>0 and newmask.any()
    oldmask=bank['cpp_valid_mask'].astype(bool);oldref=bank['cpp_reference_cot'].astype(float)
    bank['cpp_reference_cot']=newref;bank['cpp_valid_mask']=newmask
    rows_out=[]
    for si,sta in enumerate(('sili','zhujia')):
        for key,name in [('forecast_log1p_cot','SimVP forecast AGRI -> R'),('target_agri_r_log1p_cot','observed target AGRI -> R')]:
            y=np.expm1(bank[key].astype(float))[:,si]
            rows_out.append(dict(station=sta,mode=name,old_invalid=score(oldref[:,si],y,oldmask[:,si]),
                                 corrected=score(newref[:,si],y,newmask[:,si])))
    direct=score(rdata['reference_cot'].astype(float),np.expm1(rdata['pred_log1p_cot'].astype(float)),rdata['mask'].astype(bool))
    assert abs(direct['rmse']-9.074)<.001
    OUT.mkdir(exist_ok=True)
    repaired=OUT/'forecast_cot_val_bank.npz';np.savez_compressed(repaired,**bank)
    # No inference/model output changed during label repair.
    with np.load(OLD,allow_pickle=False) as z:
        for key in ('forecast_log1p_cot','target_agri_r_log1p_cot','persistence_log1p_cot','target_time_utc','sequence_ids','dataset_indices'):
            assert np.array_equal(bank[key],z[key])
    report=dict(state='CONFIRMED_CPP_REFERENCE_GLOBAL_VS_VAL_LOCAL_INDEX_BUG',source_bank=str(OLD),
                old_bank_sha256=sha(OLD),new_bank_sha256=sha(repaired),new_bank=str(repaired),
                evidence='Every previously supported bank key matched the reconstructed erroneous val-local join exactly.',
                wrong_key_examples=examples,old_matched_window_station_lead_records=old_matches,
                corrected_key_records=correct_key_records,old_valid_pixel_records=int(oldmask.sum()),
                new_valid_pixel_records=int(newmask.sum()),window_count=shape[0],
                scope='existing 271 forecast windows only; not a re-enumeration of all eligible validation windows',
                changed_arrays=['cpp_reference_cot','cpp_valid_mask'],model_outputs_unchanged=True,
                direct_R_original_validation=direct,paired_metrics=rows_out,test_read=0,
                limitations='Stored target-AGRI R replay uses forecast dataset inputs/geometry; direct R validation is a separate replay/support. Do not equate supports.')
    (HERE/'AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
