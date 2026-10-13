#!/usr/bin/env python3
"""Post-hoc validation diagnosis of frozen T16 cases; no training/test reads."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / 'experiments/monthly_R_station16_20261013_v1'
sys.path.insert(0, str(EXP))
from head_candidate_v0 import LocalGHIHead


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(p, np.float64)-np.asarray(y, np.float64))**2)))


def main():
    torch.set_num_threads(2)
    pack = EXP / 'results/ghi_head_pack_new_r_v1'
    out = EXP / 'results/t16_deterioration_forensics_20261014'
    out.mkdir(exist_ok=True)
    manifest_path = EXP / 'results/t16_partial_case_manifest_v1.json'
    cases = json.loads(manifest_path.read_text(encoding='utf-8'))['selected_cases']
    meta = json.loads((pack/'pack_manifest.json').read_text())
    assert meta['test_payload_opened'] is False and meta['state'] == 'COMPLETE'
    for name, digest in meta['array_sha256'].items():
        assert sha(pack/name) == digest, name
    rows = list(csv.DictReader((pack/'rows.csv').open(encoding='utf-8', newline='')))
    assert {r['split'] for r in rows} == {'train', 'val'}
    key = {(r['station'], r['target_time_utc']): i for i, r in enumerate(rows)}
    assert len(key) == len(rows)
    ag = np.load(pack/'agri_raw.npy', allow_pickle=False)
    av = np.load(pack/'agri_valid.npy', allow_pickle=False).astype(bool)
    geo = np.load(pack/'geometry.npy', allow_pickle=False)
    logc = np.load(pack/'cot_log1p.npy', allow_pickle=False)
    x = np.where(av, (ag-np.array(meta['AGRI_norm_mean'], np.float32)[None,:,None,None])/
                 np.array(meta['AGRI_norm_std'], np.float32)[None,:,None,None], 0).astype(np.float32)
    cm, cs = meta['COT_norm_mean'], meta['COT_norm_std']
    c = ((logc-cm)/cs).astype(np.float32)
    y = np.array([float(r['ghi']) for r in rows], np.float64)
    clear = np.array([float(r['clear']) for r in rows], np.float32)
    kt = y/clear
    with np.load(REPO/'refine-logs/cot_value_first_20261013/validation_predictions_monthly_R_seed42.npz', allow_pickle=False) as z:
        ref_full = z['reference_cot'].astype(np.float32)
        rlog_full = z['pred_log1p_cot'].astype(np.float32)
        refvalid_full = z['mask'].astype(bool)
        ref = z['reference_cot'][:, :, 7:10, 7:10].astype(np.float32)
        refvalid = z['mask'][:, :, 7:10, 7:10].astype(bool)
        rlog = z['pred_log1p_cot'][:, :, 7:10, 7:10]
        ridx = z['sample_indices']
    with np.load(REPO/'refine-logs/cot_value_first_20261013/clp_reference_monthly_R_seed42.npz', allow_pickle=False) as z:
        assert np.array_equal(ridx, z['sample_indices'])
        refkeys = list(zip(z['station'].astype(str), z['target_time_utc'].astype(str)))
    refi = {k: i for i, k in enumerate(refkeys)}
    reference_ids = [i for i, r in enumerate(rows) if r['split']=='val' and
                     (r['station'], r['target_time_utc']) in refi and
                     refvalid[refi[(r['station'], r['target_time_utc'])]].all()]
    spatial_common_ids=[i for i in reference_ids if refvalid_full[refi[(rows[i]['station'],rows[i]['target_time_utc'])],:,5:12,5:12].all()]
    bank_ref_max_diff = max(float(np.max(np.abs(logc[i]-rlog[refi[(rows[i]['station'],rows[i]['target_time_utc'])]]))) for i in reference_ids)
    print('BANK_R_REFERENCE_LOGCOT_MAX_ABS',bank_ref_max_diff,flush=True)
    bank_ref_physical_diff = max(float(np.max(np.abs(np.expm1(logc[i])-np.expm1(rlog[refi[(rows[i]['station'],rows[i]['target_time_utc'])]])))) for i in reference_ids)
    # Both artifacts bind the same R checkpoint. Record cross-batch numerical drift;
    # the GHI probes always use the actual bank input, not the validation replay.
    assert bank_ref_max_diff < .01 and bank_ref_physical_diff < .5
    rawpath = REPO/'refine-logs/cot_value_first_20261013/tmp/unique_target_labels_train_val.csv'
    raw5 = {}
    for r in csv.DictReader(rawpath.open(encoding='utf-8-sig', newline='')):
        if r['new_split']=='val' and r['label_status']=='complete_ghi':
            raw5[(r['station'], r['target_time_utc'])] = [float(v) for v in r['raw_ghi_5min_wm2'].split('|')]
    with np.load(out/'solar_geometry_audit.npz',allow_pickle=False) as z:
        spa_by_key={k: g for k,g in zip(zip(z['station'].astype(str),z['target_time_utc'].astype(str)),z['geometry'])}
    saved = {}; replay = {}; interventions = {}; gradients = {}; cp_hash = {}; reload_max = {}
    train_replay={}
    geometry_replay={}; time_donor_available={}
    permutation_outputs = {}; shifted_outputs = {}
    for station in ('sili','zhujia'):
        ids = np.array([i for i,r in enumerate(rows) if r['split']=='val' and r['station']==station])
        tx = torch.tensor(x[ids]); tg = torch.tensor(geo[ids]); tc = torch.tensor(c[ids]); cl = torch.tensor(clear[ids])
        train_ids=np.array([i for i,r in enumerate(rows) if r['split']=='train' and r['station']==station])
        train_tx=torch.tensor(x[train_ids]); train_tg=torch.tensor(geo[train_ids]); train_tc=torch.tensor(c[train_ids]); train_cl=torch.tensor(clear[train_ids])
        local = {int(g): j for j,g in enumerate(ids)}
        ref_local = [local[i] for i in reference_ids if rows[i]['station']==station]
        c_ref = tc.clone()
        for j in ref_local:
            i = ids[j]; c_ref[j] = torch.tensor((np.log1p(ref[refi[(station, rows[i]['target_time_utc'])]])-cm)/cs)
        variants = {'CPP_reference': c_ref,
                    'COT_x0.8': torch.tensor((np.log1p(np.expm1(logc[ids])*.8)-cm)/cs),
                    'COT_x1.2': torch.tensor((np.log1p(np.expm1(logc[ids])*1.2)-cm)/cs)}
        g_spa=tg.clone()
        for j,i in enumerate(ids):
            if (station,rows[i]['target_time_utc']) in spa_by_key:
                g_spa[j]=torch.tensor(spa_by_key[(station,rows[i]['target_time_utc'])])
        for offset in (-15, 15):
            alt = tc.clone(); available = np.zeros(len(ids), bool)
            for j,i in enumerate(ids):
                dt = datetime.fromisoformat(rows[i]['target_time_utc'])+timedelta(minutes=offset)
                other = key.get((station,dt.strftime('%Y-%m-%d %H:%M:%S')))
                if other is not None and rows[other]['split']=='val':
                    alt[j] = torch.tensor(c[other]); available[j] = True
            variants[f'COT_time_{offset:+d}min'] = alt
            for j,i in enumerate(ids):
                time_donor_available[(station,int(i),offset)]=bool(available[j])
        for seed in (42,43,44):
            model_dir = EXP/f'results/ghi_heads_new_r_seed{seed}_v1'/station
            for arm in ('A','AC','AP'):
                model = LocalGHIHead(arm,2)
                cp = model_dir/f'{arm}_best.pt'
                ck = torch.load(cp,map_location='cpu',weights_only=True)
                assert ck['arm']==arm and ck['seed']==seed
                assert sha(cp)==json.loads((model_dir/f'{arm}_validation.json').read_text())['checkpoint_sha256']
                model.load_state_dict(ck['model']);model.eval();cp_hash[(station,seed,arm)]=sha(cp)
                with torch.inference_mode():
                    train_pred=(model(train_tx,train_tg,train_tc if arm=='AC' else None)*train_cl).numpy()
                train_replay[(station,seed,arm)]={int(i):float(p) for i,p in zip(train_ids,train_pred)}
                with torch.inference_mode():
                    base = (model(tx,tg,tc if arm=='AC' else None)*cl).numpy().astype(np.float64)
                arr=np.asarray(np.load(model_dir/f'{arm}_val_prediction_ghi_wm2.npy',allow_pickle=False),np.float64)
                drift=float(np.max(np.abs(base-arr)));assert drift<1
                reload_max[(station,seed,arm)]=drift
                saved[(station,seed,arm)]={int(i):float(p) for i,p in zip(ids,arr)}
                replay[(station,seed,arm)]={int(i):float(p) for i,p in zip(ids,base)}
                with torch.inference_mode():spa_pred=(model(tx,g_spa,tc if arm=='AC' else None)*cl).numpy()
                geometry_replay[(station,seed,arm)]={int(i):float(p) for i,p in zip(ids,spa_pred)}
                if arm=='AC':
                    for name, alt in variants.items():
                        with torch.inference_mode():
                            pred=(model(tx,tg,alt)*cl).numpy().astype(np.float64)
                        interventions[(station,seed,name)]={int(i):float(p) for i,p in zip(ids,pred)}
                    gradc=tc.clone().requires_grad_()
                    output=model(tx,tg,gradc)*cl
                    grad=torch.autograd.grad(output.sum(),gradc)[0].detach().numpy()/cs
                    gradients[(station,seed)]={int(i): g.tolist() for i,g in zip(ids,grad)}
                    chosen_ids=sorted({key[(station,t)] for ca in cases if ca['station']==station for t in ca['future_target_times_utc']}|
                                      {i for i in spatial_common_ids if rows[i]['station']==station})
                    chosen_local=[local[i] for i in chosen_ids]
                    cx=tx[chosen_local];cg=tg[chosen_local];cc=tc[chosen_local];ccl=cl[chosen_local]
                    neighbors=[(r,col) for r in range(3) for col in range(3) if (r,col)!=(1,1)]
                    rng=np.random.default_rng(20261014)
                    for perm_id in range(32):
                        alt=cc.clone();perm=rng.permutation(8)
                        for dst,(r,col) in enumerate(neighbors):
                            sr,sc=neighbors[perm[dst]];alt[:,:,r,col]=cc[:,:,sr,sc]
                        with torch.inference_mode(): pp=(model(cx,cg,alt)*ccl).numpy()
                        permutation_outputs[(station,seed,perm_id)]={i:float(v) for i,v in zip(chosen_ids,pp)}
                    for source,full in [('R',rlog_full),('CPP',ref_full)]:
                        for dy in range(-2,3):
                            for dx in range(-2,3):
                                native=np.stack([full[refi[(station,rows[i]['target_time_utc'])],:,7+dy:10+dy,7+dx:10+dx] for i in chosen_ids])
                                alt=torch.tensor(((np.log1p(native) if source=='CPP' else native)-cm)/cs)
                                with torch.inference_mode():pp=(model(cx,cg,alt)*ccl).numpy()
                                shifted_outputs[(station,seed,source,dy,dx)]={i:float(v) for i,v in zip(chosen_ids,pp)}
        print('REPLAY_AND_COUNTERFACTUAL_COMPLETE', station, len(ids),flush=True)
    case_summaries=[]; detail=[]
    physical=np.expm1(logc[:,0,1,1]); cb=[0,.1,3,10,30,101]; sb=[0,.25,.5,.75,1.01]
    for ci,case in enumerate(cases):
        station=case['station']; ids=[key[(station,ts)] for ts in case['future_target_times_utc']]
        targets=[]
        for lead,i in enumerate(ids,1):
            ri=refi[(station,rows[i]['target_time_utc'])]
            raw = raw5[(station,rows[i]['target_time_utc'])]
            assert abs(np.mean(raw)-y[i])<1e-6
            preds={arm:np.array([saved[(station,s,arm)][i] for s in (42,43,44)]) for arm in ('A','AC','AP')}
            delta=preds['AC']-preds['A']; needed=y[i]-preds['A']; es=(preds['AC']-y[i])**2-(preds['A']-y[i])**2
            assert np.max(np.abs(es-(-2*needed*delta+delta**2)))<1e-6
            row={'case':ci+1,'station':station,'lead_min':lead*15,'target_time_utc':rows[i]['target_time_utc'],
                 'ghi':y[i],'clear':float(clear[i]),'kt':float(kt[i]),'raw_5min':raw,'raw_5min_range':max(raw)-min(raw),
                 'A':float(preds['A'].mean()),'AC':float(preds['AC'].mean()),'AP':float(preds['AP'].mean()),
                 'AC_minus_A':float(delta.mean()),'desired_y_minus_A':float(needed.mean()),
                 'mean_seed_delta_squared_error':float(es.mean()),'seed_delta_squared_errors':es.tolist(),
                 'seed_AC_minus_A':delta.tolist(), 'R_center_COT':float(np.expm1(logc[i,0,1,1])),
                 'CPP_center_COT':float(ref[ri,0,1,1]),'R_CPP_3x3_MAE':float(np.abs(np.expm1(logc[i])-ref[ri]).mean()),
                 'cosSOZ':float(geo[i,0,1,1]),'cosRAA':float(geo[i,1,1,1]),
                 'gradient_GHI_wrt_logCOT':np.mean([gradients[(station,s)][i] for s in (42,43,44)],axis=0).tolist()}
            for name in variants:
                row[name]=float(np.mean([interventions[(station,s,name)][i] for s in (42,43,44)]))
                row[name+'_minus_AC_replay']=float(np.mean([interventions[(station,s,name)][i]-replay[(station,s,'AC')][i] for s in (42,43,44)]))
            for arm in ('A','AC'):
                row[arm+'_SPA_nominal']=float(np.mean([geometry_replay[(station,s,arm)][i] for s in (42,43,44)]))
                row[arm+'_CPU_replay']=float(np.mean([replay[(station,s,arm)][i] for s in (42,43,44)]))
            for offset in (-15,15):
                row[f'COT_time_{offset:+d}min_donor_available']=time_donor_available[(station,i,offset)]
            cot_bin=min(max(int(np.searchsorted(cb,physical[i],side='right'))-1,0),len(cb)-2)
            sun_bin=min(max(int(np.searchsorted(sb,geo[i,0,1,1],side='right'))-1,0),len(sb)-2)
            train_match=np.array([r['station']==station and r['split']=='train' for r in rows])&\
                        (physical>=cb[cot_bin])&(physical<cb[cot_bin+1])&\
                        (geo[:,0,1,1]>=sb[sun_bin])&(geo[:,0,1,1]<sb[sun_bin+1])&(clear>=100)
            support=kt[train_match]
            row['training_condition_count']=int(len(support))
            row['training_condition_median_kt']=float(np.median(support)) if len(support) else None
            row['training_condition_p90_kt']=float(np.quantile(support,.9)) if len(support) else None
            row['observed_kt_training_condition_percentile']=float(np.mean(support<=kt[i])) if len(support) else None
            targets.append(row);detail.append(row)
        yi=y[ids]
        per_seed={}
        for seed in (42,43,44):
            result={arm:rmse(yi,[saved[(station,seed,arm)][i] for i in ids]) for arm in ('A','AC','AP')}
            for name in variants:
                result[name]=rmse(yi,[interventions[(station,seed,name)][i] for i in ids])
            for arm in ('A','AC'):
                result[arm+'_SPA_nominal']=rmse(yi,[geometry_replay[(station,seed,arm)][i] for i in ids])
                result[arm+'_CPU_replay']=rmse(yi,[replay[(station,seed,arm)][i] for i in ids])
                result[arm+'_SPA_minus_CPU_replay']=result[arm+'_SPA_nominal']-result[arm+'_CPU_replay']
            result['CPP_reference_minus_AC_CPU_replay']=result['CPP_reference']-result['AC_CPU_replay']
            result['AC_minus_A']=result['AC']-result['A'];result['CPP_reference_minus_AC']=result['CPP_reference']-result['AC']
            per_seed[str(seed)]=result
        stats={name:float(np.mean([p[name] for p in per_seed.values()])) for name in next(iter(per_seed.values()))}
        harmful=np.array([t['mean_seed_delta_squared_error'] for t in targets])>0
        positive_sse=sum(t['mean_seed_delta_squared_error'] for t in targets if t['mean_seed_delta_squared_error']>0)
        ranked=sorted(targets,key=lambda t:t['mean_seed_delta_squared_error'],reverse=True)
        perm_changes=np.array([[np.mean([permutation_outputs[(station,s,p)][i]-replay[(station,s,'AC')][i] for s in (42,43,44)]) for i in ids] for p in range(32)])
        shifts=[]
        for source in ('R','CPP'):
            for dy in range(-2,3):
                for dx in range(-2,3):
                    value=float(np.mean([rmse(yi,[shifted_outputs[(station,s,source,dy,dx)][i] for i in ids]) for s in (42,43,44)]))
                    shifts.append({'source':source,'dy':dy,'dx':dx,'mean_seed_rmse':value,'minus_AC':value-stats['AC']})
        case_summaries.append({'case':ci+1,'station':station,'init_time_BJT':case['init_time_BJT'],
            'mean_seed_rmse':stats,'per_seed_rmse':per_seed,'harmful_targets':int(harmful.sum()),
            'wrong_correction_direction_targets':int(sum(t['AC_minus_A']*t['desired_y_minus_A']<0 for t in targets)),
            'positive_SSE_top3_fraction':sum(max(t['mean_seed_delta_squared_error'],0) for t in ranked[:3])/positive_sse if positive_sse else 0,
            'top_harmful_targets':ranked[:4],
            'mean_abs_R_CPP_3x3_COT_error':float(np.mean([t['R_CPP_3x3_MAE'] for t in targets])),
            'mean_raw5min_range':float(np.mean([t['raw_5min_range'] for t in targets])),
            'mean_AC_minus_A':float(np.mean([t['AC_minus_A'] for t in targets])),
            'mean_seed_bias':{arm:float(np.mean([t[arm]-t['ghi'] for t in targets])) for arm in ('A','AC','AP')},
            'CPP_reference_mean_abs_prediction_change':float(np.mean([abs(t['CPP_reference_minus_AC_replay']) for t in targets])),
            'permutation_32_mean_abs_GHI_change':float(np.abs(perm_changes).mean()),
            'permutation_32_max_abs_GHI_change':float(np.abs(perm_changes).max()),
            'negative_uniform_logCOT_derivative_targets':int(sum(np.sum(t['gradient_GHI_wrt_logCOT'])<0 for t in targets)),
            'spatial_shift_probes':shifts})
    unique_ids={key[(c['station'], ts)] for c in cases for ts in c['future_target_times_utc']}
    overlaps=[]
    for j,a in enumerate(cases):
        for k,b in enumerate(cases[j+1:],j+1):
            if a['station']==b['station']:
                overlaps.append({'case_pair':[j+1,k+1],'shared_target_count':len(set(a['future_target_times_utc'])&set(b['future_target_times_utc']))})
    full_summary={}
    for station in ('sili','zhujia'):
        ids=[i for i,r in enumerate(rows) if r['split']=='val' and r['station']==station]
        rs=[i for i in reference_ids if rows[i]['station']==station]
        full_summary[station]={'n':len(ids),'reference_3x3_valid_n':len(rs),'rmse_differences':{},'CPP_reference_substitution':{}}
        for name in ('AC','AP'):
            full_summary[station]['rmse_differences'][name+'_minus_A']=float(np.mean([rmse(y[ids],[saved[(station,s,name)][i] for i in ids])-rmse(y[ids],[saved[(station,s,'A')][i] for i in ids]) for s in (42,43,44)]))
        for name in ('A','AC','CPP_reference'):
            full_summary[station]['CPP_reference_substitution'][name]=float(np.mean([rmse(y[rs],[interventions[(station,s,name)][i] if name=='CPP_reference' else saved[(station,s,name)][i] for i in rs]) for s in (42,43,44)]))
        sid=[i for i in spatial_common_ids if rows[i]['station']==station]
        groups={}
        for i in sid:
            day=(datetime.fromisoformat(rows[i]['target_time_utc'])+timedelta(hours=8)).strftime('%Y-%m-%d')
            groups.setdefault(day,[]).append(i)
        spatial_summary=[]
        for source in ('R','CPP'):
            base=float(np.mean([rmse(y[sid],[shifted_outputs[(station,s,source,0,0)][i] for i in sid]) for s in (42,43,44)]))
            for dy in range(-2,3):
                for dx in range(-2,3):
                    value=float(np.mean([rmse(y[sid],[shifted_outputs[(station,s,source,dy,dx)][i] for i in sid]) for s in (42,43,44)]))
                    dates=[]
                    for date,gids in groups.items():
                        delta=float(np.mean([rmse(y[gids],[shifted_outputs[(station,s,source,dy,dx)][i] for i in gids])-
                                            rmse(y[gids],[shifted_outputs[(station,s,source,0,0)][i] for i in gids]) for s in (42,43,44)]))
                        dates.append({'date_BJT':date,'n':len(gids),'delta_vs_same_source_center':delta})
                    spatial_summary.append({'source':source,'dy':dy,'dx':dx,'n':len(sid),'mean_seed_RMSE':value,
                                            'delta_vs_same_source_center':value-base,'per_date':dates})
        full_summary[station]['spatial_shift_common_support']=spatial_summary
        observed_kt_groups=[]
        for lo,hi in [(0,.5),(.5,.9),(.9,float('inf'))]:
            subset=[i for i in ids if clear[i]>=100 and physical[i]>=3 and lo<=kt[i]<hi]
            if not subset:continue
            values={arm:float(np.mean([rmse(y[subset],[saved[(station,s,arm)][i] for i in subset]) for s in (42,43,44)])) for arm in ('A','AC','AP')}
            observed_kt_groups.append({'observed_kt_bin':[lo,hi if np.isfinite(hi) else None],
                                      'R_center_COT_min':3,'clear_GHI_min_wm2':100,'n':len(subset),
                                      'mean_seed_rmse':values,'AC_minus_A':values['AC']-values['A']})
        full_summary[station]['posthoc_target_based_kt_groups']=observed_kt_groups
        train_groups=[]
        for lo,hi in [(0,.5),(.5,.9),(.9,float('inf'))]:
            subset=[i for i,r in enumerate(rows) if r['split']=='train' and r['station']==station and
                    clear[i]>=100 and physical[i]>=3 and lo<=kt[i]<hi]
            if not subset:continue
            values={arm:float(np.mean([rmse(y[subset],[train_replay[(station,s,arm)][i] for i in subset]) for s in (42,43,44)])) for arm in ('A','AC','AP')}
            train_groups.append({'observed_kt_bin':[lo,hi if np.isfinite(hi) else None],'R_center_COT_min':3,
                                 'clear_GHI_min_wm2':100,'n':len(subset),'mean_seed_rmse':values,
                                 'AC_minus_A':values['AC']-values['A']})
        full_summary[station]['posthoc_training_kt_groups']=train_groups
    # Training conditional support: fixed physical centre-COT and solar-cosine bins.
    bins=[]
    for station in ('sili','zhujia'):
        train=np.array([r['split']=='train' and r['station']==station for r in rows])
        for lo,hi in zip(cb[:-1],cb[1:]):
            for slo,shi in zip(sb[:-1],sb[1:]):
                mask=train&(physical>=lo)&(physical<hi)&(geo[:,0,1,1]>=slo)&(geo[:,0,1,1]<shi)
                if mask.any():bins.append({'station':station,'COT_bin':[lo,hi],'cosSOZ_bin':[slo,shi],'n':int(mask.sum()),'kt_p10_p50_p90':np.quantile(kt[mask],[.1,.5,.9]).tolist()})
    report={'state':'COMPLETE_POSTHOC_T16_VALIDATION_DIAGNOSIS','case_manifest_sha256':sha(manifest_path),
            'pack_sha256':sha(pack/'pack_manifest.json'),'input_hashes_verified':True,'bank_vs_R_reference_logCOT_max_abs':bank_ref_max_diff,
            'bank_vs_R_reference_physical_COT_max_abs':bank_ref_physical_diff,
            'max_reload_vs_saved_wm2':max(reload_max.values()),'case_summary':case_summaries,'all_validation':full_summary,
            'case_overlap':overlaps,'case_displayed_rows':64,'unique_station_target_rows':len(unique_ids),
            'train_conditional_bins':bins,'checkpoint_hashes':{'/'.join(map(str,k)):v for k,v in cp_hash.items()},
            'source_sha256':sha(__file__),'solar_geometry_audit_sha256':sha(out/'solar_geometry_audit.json'),'test_payload_opened':False,
            'limits':['post-hoc validation diagnosis; no new training or test','CPP substitution is a frozen-head input diagnostic, not a retrained upper bound','A/AC difference includes independently optimized common trunks','15-minute GHI labels are correct interval averages; endpoint imagery does not describe all intra-interval evolution']}
    (out/'forensic_audit.json').write_bytes((json.dumps(report,indent=2,ensure_ascii=False)+'\n').encode('utf-8'))
    (out/'per_target_diagnostics.json').write_bytes((json.dumps(detail,indent=2,ensure_ascii=False)+'\n').encode('utf-8'))
    scalar=[k for k,v in detail[0].items() if not isinstance(v,list)]
    with (out/'per_target_diagnostics.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=scalar,lineterminator='\n');w.writeheader();w.writerows({k:r[k] for k in scalar} for r in detail)
    log_records=[]
    for seed in (42,43,44):
        for station in ('sili','zhujia'):
            logpath=out/f'metrics_seed{seed}_{station}.jsonl'
            logrows=[json.loads(s) for s in logpath.read_text().splitlines()]
            for arm in ('A','AC','AP'):
                history=[r for r in logrows if r['arm']==arm]
                assert len(history)==30 and [r['epoch'] for r in history]==list(range(30))
                best=min(history,key=lambda r:r['val_ghi_rmse_wm2'])
                log_records.append({'seed':seed,'station':station,'arm':arm,'best_epoch_zero_based':best['epoch'],
                    'train_rmse_at_epoch20':1000*history[20]['train_mse_scaled']**.5,
                    'train_rmse_at_last_epoch':1000*history[-1]['train_mse_scaled']**.5,
                    'best_val_rmse':best['val_ghi_rmse_wm2'],'last_val_rmse':history[-1]['val_ghi_rmse_wm2'],
                    'source_log_sha256':sha(logpath)})
    (out/'optimization_log_audit.json').write_bytes((json.dumps({'scope':'18 train/validation optimization logs, no test; epoch train loss collected during changing weights, not final evaluation',
                                                            'records':log_records},indent=2)+'\n').encode('utf-8'))
    print(json.dumps({'case_summary':[{k:v for k,v in r.items() if k not in ('top_harmful_targets','spatial_shift_probes','per_seed_rmse')} for r in case_summaries],
                      'all_validation':{s:{k:v for k,v in d.items() if k!='spatial_shift_common_support'} for s,d in full_summary.items()},
                      'overlap':overlaps,'unique_rows':len(unique_ids)},indent=2),flush=True)


if __name__=='__main__':main()
