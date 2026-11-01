"""Field/trajectory fidelity to dense observed-image R, not GHI or CPP skill."""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from audit_cot_event_timing import (ARMS, STATIONS, STEP, csvout, read_csv,
    sha, runs, select_detection, self_test, peak_metrics, aggregate)
from audit_teacher_forecast_error_decomposition import met

SCALES={'center1':(slice(8,9),slice(8,9)), 'station3':(slice(7,10),slice(7,10)),
        'patch16':(slice(0,16),slice(0,16))}


def changes(p,y,m):
    mm=m[:,1:] & m[:,:-1]
    pp=np.diff(p.astype(np.float64),axis=1)[mm]
    yy=np.diff(y.astype(np.float64),axis=1)[mm]
    assert np.isfinite(pp).all() and np.isfinite(yy).all()
    if not len(pp):return {'n':0,'rmse':None,'mae':None,'pred_sd':None,'reference_sd':None,'sd_ratio':None,'corr':None}
    a,b=float(pp.std()),float(yy.std())
    return {'n':len(pp),'rmse':float(np.sqrt(np.mean((pp-yy)**2))), 'mae':float(np.abs(pp-yy).mean()),
            'pred_sd':a,'reference_sd':b,'sd_ratio':a/b if b>0 else None,
            'corr':float(np.corrcoef(pp,yy)[0,1]) if min(a,b)>0 else None}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();base,out=a.base,a.output;self_test()
    if out.exists():raise FileExistsError(out)
    rd=base/'results/dense_observed_v_r_reference_v1_20261021';receipt=json.loads((rd/'complete.json').read_text())
    assert receipt['state']=='COMPLETE_DENSE_OBSERVED_V_R_REFERENCE'
    for name,h in receipt['artifact_sha256'].items():assert sha(rd/name)==h,name
    prior=json.loads((base/'results/cot_event_timing_diagnostic_v1_20261021/summary.json').read_text())
    assert sha(base/'audit_cot_event_timing.py')==prior['source']['script_sha256'], 'event helpers changed'
    inf=base/'data/cpp_score_inputs_v1/inference';im=json.loads((inf/'inference_manifest.json').read_text())
    assert sha(inf/'inference_manifest.json')==receipt['source']['inference_manifest_sha256']
    vs=base/'data/sequence_support_v1/v_sequences.csv';assert sha(vs)==receipt['source']['v_support_sha256']
    seqs={r['seq_id']:r for r in read_csv(vs)};vindex=read_csv(inf/'V/index.csv');assert len(vindex)==3140
    rows=read_csv(rd/'rows.csv');byrel={r['data_relpath']:int(r['frame_index']) for r in rows}
    targets=[];inits=[];times=[]
    for si,r in enumerate(vindex):
        assert int(r['sequence_index_within_split'])==si
        paths=seqs[r['seq_id']]['data_relpaths'].split('|')
        tt=[datetime.strptime(Path(p).stem,'%Y%m%d%H%M') for p in paths]
        assert all(tt[j+1]-tt[j]==STEP for j in range(23))
        targets.append([byrel[p] for p in paths[8:]]);inits.append(tt[7]);times.append(tt[8:])
    target=np.asarray(targets,np.int64);ref=np.load(rd/'cot.npy')[:,:,0];mask=np.load(rd/'mask.npy')[:,:,0]
    forecasts={}
    for arm in ARMS:
        path=inf/'V'/(arm+'.npy');assert sha(path)==im['artifact_sha256']['V/'+arm+'.npy']
        forecasts[arm]=np.load(path,mmap_mode='r')
    metricrows=[];changerows=[];windows=[]
    for sti,st in enumerate(STATIONS):
        for sq in range(3140):
            ri=target[sq]
            if mask[ri,sti,8,8].all():
                windows.append({'station':st,'sti':sti,'sq':sq,'ri':ri,'init':inits[sq],
                                'times':times[sq],'id':f'{st}_{sq}'})
    # Small per-station slices avoid duplicating all full-field forecast arrays.
    for scale,(rs,cs) in SCALES.items():
        pooled={arm:[] for arm in ARMS};pooly=[];poolm=[]
        for sti,st in enumerate(STATIONS):
            yy=ref[target,sti,rs,cs];mm=mask[target,sti,rs,cs]
            pooly.append(yy);poolm.append(mm)
            for arm in ARMS:
                pp=np.asarray(forecasts[arm][:,sti,:,0,rs,cs],np.float32)
                pooled[arm].append(pp)
                for li in [None]+list(range(16)):
                    p,y,m=(pp,yy,mm) if li is None else (pp[:,li],yy[:,li],mm[:,li])
                    metricrows.append({'reference':'dense_observed_R','scale':scale,'station':st,
                        'lead_minutes':'all' if li is None else 15*(li+1),'arm':arm,**met(p,y,m)})
                if scale in ('center1','station3'):
                    changerows.append({'reference':'dense_observed_R','scale':scale,'station':st,'arm':arm,**changes(pp,yy,mm)})
        yy=np.concatenate(pooly,axis=0);mm=np.concatenate(poolm,axis=0)
        for arm in ARMS:
            pp=np.concatenate(pooled[arm],axis=0)
            for li in [None]+list(range(16)):
                p,y,m=(pp,yy,mm) if li is None else (pp[:,li],yy[:,li],mm[:,li])
                metricrows.append({'reference':'dense_observed_R','scale':scale,'station':'both',
                    'lead_minutes':'all' if li is None else 15*(li+1),'arm':arm,**met(p,y,m)})
            if scale in ('center1','station3'):
                changerows.append({'reference':'dense_observed_R','scale':scale,'station':'both','arm':arm,**changes(pp,yy,mm)})
        del pooled,yy,mm,pp
    daily={}
    for w in windows:
        dates={(t+timedelta(hours=8)).date().isoformat() for t in w['times']}
        if len(dates)!=1:continue
        key=(w['station'],next(iter(dates)))
        if key not in daily or w['init']<daily[key]['init']:daily[key]=w
    dailyrows=[];windowlist=[];events=[];eventpred=[]
    for (st,date),w in sorted(daily.items()):
        y=ref[w['ri'],w['sti'],8,8];patchok=bool(mask[w['ri'],w['sti'],7:10,7:10].all())
        windowlist.append({'station':st,'date_bjt':date,'window_id':w['id'],'sequence_index':w['sq'],
            'init_time_bjt':(w['init']+timedelta(hours=8)).isoformat(' '),'first_target_bjt':(w['times'][0]+timedelta(hours=8)).isoformat(' '),
            'last_target_bjt':(w['times'][-1]+timedelta(hours=8)).isoformat(' '),'patch3_all_valid':patchok})
        for arm in ARMS:
            pp=np.asarray(forecasts[arm][w['sq'],w['sti'],:,0]);p=pp[:,8,8];hi,ph=y>=30,p>=30
            dailyrows.append({'reference':'dense_observed_R','station':st,'date_bjt':date,'window_id':w['id'],'arm':arm,
                'tp':int((hi&ph).sum()),'fp':int((~hi&ph).sum()),'fn':int((hi&~ph).sum()),
                'reference_thick_frames':int(hi.sum()),'prediction_thick_frames':int(ph.sum()),**peak_metrics(p,y),
                'patch3_peak_error':float(pp[:,7:10,7:10].max()-ref[w['ri'],w['sti'],7:10,7:10].max()) if patchok else None})
    sortedri=sorted(range(4025),key=lambda i:rows[i]['target_time_utc']);tt=[datetime.fromisoformat(rows[i]['target_time_utc']) for i in sortedri]
    cover=[]
    for sti,st in enumerate(STATIONS):
        es=runs(tt,ref[sortedri,sti,8,8],mask[sortedri,sti,8,8])
        cover.append({'station':st,'single_frame_runs':sum(e['n']==1 for e in es),'two_frame_events':sum(e['n']>=2 for e in es)})
        for e in es:
            if e['n']<2:continue
            eid=f'{st}_R_{e["start"]:%Y%m%d%H%M}'
            candidates=[w for w in windows if w['station']==st and w['times'][0]<=e['start'] and w['times'][-1]>=e['end']+STEP] if not e['left_censored'] and not e['right_censored'] else []
            w=min(candidates,key=lambda w:w['init']) if candidates else None
            events.append({'event_id':eid,'station':st,'start_time_bjt':(e['start']+timedelta(hours=8)).isoformat(' '),
                'end_time_bjt':(e['end']+timedelta(hours=8)).isoformat(' '),'thick_frames':e['n'],
                'left_censored':e['left_censored'],'right_censored':e['right_censored'],
                'canonical_window':w['id'] if w else '',
                'status':'EVALUABLE' if w else ('CENSORED_REFERENCE' if e['left_censored'] or e['right_censored'] else 'NO_COMPLETE_WINDOW')})
            if w is None:continue
            for arm in ARMS:
                pp=np.asarray(forecasts[arm][w['sq'],sti,:,0,8,8]);pr=runs(w['times'],pp,np.ones(16,bool));d=select_detection(e,pr)
                eventpred.append({'event_id':eid,'station':st,'window_id':w['id'],'sequence_index':w['sq'],'arm':arm,
                    'start_lead_min':int((e['start']-w['init']).total_seconds()/60),'detected':d is not None,
                    'predicted_left_censored':bool(d and d['left_censored']),
                    'arrival_error_min':int((d['start']-e['start']).total_seconds()/60) if d and not d['left_censored'] else None})
    selected=[]
    for st in STATIONS:
        rr=[r for r in dailyrows if r['station']==st and r['arm']=='cot_p_xc']
        if rr:selected.append(max(rr,key=lambda r:abs(r['peak_error'])))
    timeline=[];wb={w['id']:w for w in windows}
    # Includes all evaluable event windows, plus predeclared daily examples.
    ids=sorted({r['window_id'] for r in selected}|{r['window_id'] for r in eventpred})
    for wid in ids:
        w=wb[wid]
        for j in range(16):
            timeline.append({'window_id':wid,'station':w['station'],'lead_minutes':15*(j+1),
                'init_time_bjt':(w['init']+timedelta(hours=8)).isoformat(' '),
                'target_time_bjt':(w['times'][j]+timedelta(hours=8)).isoformat(' '),
                'observed_R':float(ref[w['ri'][j],w['sti'],8,8]),
                **{arm:float(forecasts[arm][w['sq'],w['sti'],j,0,8,8]) for arm in ARMS}})
    stationstats={}
    for st in STATIONS:
        ee=[r for r in events if r['station']==st];stats={'unique_reference_events':len(ee),
            'evaluable_events':sum(r['status']=='EVALUABLE' for r in ee),
            'censored_reference':sum(r['status']=='CENSORED_REFERENCE' for r in ee),
            'no_complete_window':sum(r['status']=='NO_COMPLETE_WINDOW' for r in ee),'arms':{}}
        for arm in ARMS:
            er=[r for r in eventpred if r['station']==st and r['arm']==arm];arrival=np.asarray([r['arrival_error_min'] for r in er if r['arrival_error_min'] is not None],float)
            stats['arms'][arm]={'events':len(er),'detected':sum(r['detected'] for r in er),'missed':sum(not r['detected'] for r in er),
                'predicted_left_censored':sum(r['predicted_left_censored'] for r in er),'arrival_n':len(arrival),
                'arrival_mae_min':float(np.abs(arrival).mean()) if len(arrival) else None,
                'arrival_bias_min':float(arrival.mean()) if len(arrival) else None,
                'daily_window':aggregate([r for r in dailyrows if r['station']==st and r['arm']==arm])}
        stationstats[st]=stats
    out.mkdir(parents=True)
    for name,rs in [('field_metrics.csv',metricrows),('first_difference_metrics.csv',changerows),('fixed_daily_windows.csv',windowlist),
        ('daily_window_metrics.csv',dailyrows),('unique_reference_events.csv',events),('canonical_event_predictions.csv',eventpred),
        ('posthoc_daily_selection.csv',selected),('actual_event_timelines.csv',timeline)]:csvout(out/name,rs)
    result={'state':'COMPLETE_DENSE_R_FIELD_AND_EVENT_DIAGNOSTIC','scope':'development V; learned observed-image R reference, not CPP truth/GHI gain',
        'reference_frames':4025,'forecast_sequences':3140,'complete_center_windows':len(windows),'fixed_daily_windows':len(daily),
        'fixed_daily_windows_by_station':{st:sum(w['station']==st for w in daily.values()) for st in STATIONS},
        'run_coverage':cover,'event_station_stats':stationstats,
        'primary_station3_metrics':[r for r in metricrows if r['scale']=='station3' and r['station']=='both' and r['lead_minutes']=='all'],
        'primary_station3_changes':[r for r in changerows if r['scale']=='station3' and r['station']=='both'],
        'source':{'script_sha256':sha(Path(__file__)),'reference_receipt_sha256':sha(rd/'complete.json'),
            'forecast_manifest_sha256':sha(inf/'inference_manifest.json'),'event_helper_sha256':sha(base/'audit_cot_event_timing.py'),
            'protocol_sha256':sha(base/'DENSE_REFERENCE_INFERENCE_PROTOCOL_20261021.md')},
        'training_started':False,'ghi_labels_read':False,'test_payload_opened':False,'buffer_payload_opened':False,
        'future_observations_used_by_deployable_inference':False,'uncertainty':'descriptive; no new independent confirmation or model selection'}
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
