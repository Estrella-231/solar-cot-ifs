"""Initial-map visibility of the already frozen 27 dense-reference events."""
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from audit_cot_event_timing import read_csv, csvout, sha, ARMS, STATIONS

BASE=Path(__file__).resolve().parent


def main():
    d=BASE/'results/dense_r_field_event_diagnostic_v1_20261021'
    dense=BASE/'results/dense_observed_v_r_reference_v1_20261021'
    receipt=json.loads((dense/'complete.json').read_text())
    for name in ('cot.npy','mask.npy','rows.csv'):assert sha(dense/name)==receipt['artifact_sha256'][name]
    rr=read_csv(dense/'rows.csv');bytime={r['target_time_utc']:int(r['frame_index']) for r in rr}
    cot=np.load(dense/'cot.npy')[:,:,0];mask=np.load(dense/'mask.npy')[:,:,0]
    predictions=read_csv(d/'canonical_event_predictions.csv')
    windows={r['window_id']:r for r in read_csv(d/'actual_event_timelines.csv')}
    source=BASE/'data/cpp_score_inputs_v1/inference';im=json.loads((source/'inference_manifest.json').read_text())
    assert sha(source/'V/cot_last.npy')==im['artifact_sha256']['V/cot_last.npy']
    last=np.load(source/'V/cot_last.npy',mmap_mode='r')
    allrows=[]
    for p in predictions:
        w=windows[p['window_id']];init=datetime.fromisoformat(w['init_time_bjt'])-timedelta(hours=8)
        sq=int(p['sequence_index']);sti=STATIONS.index(p['station']);field=np.asarray(last[sq,sti,:,0])
        assert np.array_equal(field,np.broadcast_to(field[0],field.shape)), 'cot_last not repeated last historical map'
        key=init.isoformat(' ');status='INITIAL_REFERENCE_MISSING';maximum=None;center=None;drift=None
        if key in bytime:
            ri=bytime[key]
            if mask[ri,sti].all():
                maximum=float(field[0].max());center=float(field[0,8,8]);drift=float(np.abs(field[0]-cot[ri,sti]).max())
                status='CENTER_ALREADY_THICK' if center>=30 else ('NEIGHBOR_THICK_CENTER_NOT' if maximum>=30 else 'NO_THICK_IN_COMPLETE_PATCH')
            else:status='INITIAL_QA_OR_SOLAR_SUPPORT_INCOMPLETE'
        allrows.append({**p,'initial_time_utc':key,'initial_visibility':status,'initial_patch_max':maximum,
            'initial_center':center,'max_abs_last_vs_initial_observed_R':drift})
    out=BASE/'results/initial_cloud_visibility_v1_20261021'
    if out.exists():raise FileExistsError(out)
    out.mkdir();csvout(out/'event_input_visibility.csv',allrows)
    summary={}
    for st in STATIONS:
        summary[st]={}
        for arm in ARMS:
            use=[r for r in allrows if r['station']==st and r['arm']==arm]
            summary[st][arm]={'all_events':dict(Counter(r['initial_visibility'] for r in use)),
                'missed_events':dict(Counter(r['initial_visibility'] for r in use if r['detected']=='False'))}
    result={'state':'COMPLETE_INITIAL_CLOUD_VISIBILITY_DIAGNOSTIC','by_station_arm':summary,
        'source':{'script_sha256':sha(Path(__file__)),'plan_sha256':sha(BASE/'INITIAL_CLOUD_VISIBILITY_DIAGNOSTIC_PLAN_20261021.md'),
            'event_prediction_sha256':sha(d/'canonical_event_predictions.csv'),'inference_manifest_sha256':sha(source/'inference_manifest.json')},
        'training_started':False,'ghi_labels_read':False,'future_observations_used_by_deployable_inference':False,
        'test_payload_opened':False,'buffer_payload_opened':False,'scope':'development diagnostic only; no attribution to off-patch inflow without regional imagery'}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
