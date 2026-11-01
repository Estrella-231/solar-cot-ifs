from __future__ import annotations

import csv, hashlib, json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/monthly_R_station16_20261013_v1"
PACK = EXP / "results/ghi_head_pack_new_r_v1"
CLP_DIR = ROOT / "refine-logs/cot_value_first_20261013"
R_DIR = CLP_DIR
OUT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    rows=list(csv.DictReader((PACK/'rows.csv').open(newline='',encoding='utf-8')))
    idx_by_key={}
    for i,r in enumerate(rows):
        if r['split']=='val':
            key=(r['station'],r['target_time_utc'])
            if key in idx_by_key: raise ValueError(f'duplicate GHI key {key}')
            idx_by_key[key]=i
    rpred=np.load(R_DIR/'validation_predictions_monthly_R_seed42.npz')
    clp=np.load(CLP_DIR/'clp_reference_monthly_R_seed42.npz')
    if not np.array_equal(rpred['sample_indices'],clp['sample_indices']):raise ValueError('R/CLP sample index mismatch')
    rkeys=[(str(s),str(t)) for s,t in zip(clp['station'],clp['target_time_utc'])]
    pairs=[(j,idx_by_key[k]) for j,k in enumerate(rkeys) if k in idx_by_key]
    missing_keys=[k for k in rkeys if k not in idx_by_key]
    if not pairs: raise ValueError('no common COT/GHI validation keys')
    ri=np.asarray([a for a,b in pairs],int); gi=np.asarray([b for a,b in pairs],int)
    cot=np.load(PACK/'cot_log1p.npy',mmap_mode='r')
    ylog=np.asarray(rpred['pred_log1p_cot'][ri,0],np.float64)
    cmask=np.asarray(clp['clp'][ri,0],int)
    pred_crop=ylog[:,7:10,7:10]
    packed=np.asarray(cot[gi,0],np.float64)
    meta=json.loads((PACK/'pack_manifest.json').read_text())
    # Pack stores physical log1p(COT); mean/std are applied by the head loader.
    pack_consistency=float(np.max(np.abs(packed-pred_crop)))
    if pack_consistency>5e-3:
        raise ValueError(f'GHI COT pack differs from R validation output crop; max log1p discrepancy={pack_consistency}')
    gate_log=np.where(cmask[:,7:10,7:10]==0,0.0,pred_crop)
    gate_cot=((gate_log-float(meta['COT_norm_mean']))/float(meta['COT_norm_std'])).astype(np.float32)

    ag=np.load(PACK/'agri_raw.npy',mmap_mode='r'); av=np.load(PACK/'agri_valid.npy',mmap_mode='r').astype(bool)
    geo=np.load(PACK/'geometry.npy',mmap_mode='r')
    am=np.asarray(meta['AGRI_norm_mean'],np.float32)[None,:,None,None]
    ast=np.asarray(meta['AGRI_norm_std'],np.float32)[None,:,None,None]
    X=np.where(av[gi],(np.asarray(ag[gi])-am)/ast,0).astype(np.float32)
    G=np.asarray(geo[gi],np.float32)
    GH=np.asarray([float(rows[k]['ghi']) for k in gi],np.float64)
    CL=np.asarray([float(rows[k]['clear']) for k in gi],np.float64)
    station=np.asarray([rows[k]['station'] for k in gi])
    date=np.asarray([str((np.datetime64(rows[k]['target_time_utc'].replace(' ','T'),'m')+np.timedelta64(8,'h')).astype('datetime64[D]')) for k in gi])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    X=torch.from_numpy(X).to(device);G=torch.from_numpy(G).to(device)
    C0=torch.from_numpy(((packed-float(meta['COT_norm_mean']))/
                         float(meta['COT_norm_std']))[:,None].astype(np.float32)).to(device)
    C1=torch.from_numpy(gate_cot[:,None]).to(device)
    target=torch.from_numpy(GH.astype(np.float32)).to(device);clear=torch.from_numpy(CL.astype(np.float32)).to(device)
    # Import architecture from the frozen experiment implementation.
    import sys
    sys.path.insert(0,str(EXP))
    from head_candidate_v0 import LocalGHIHead
    outputs={}
    for seed in (42,43,44):
        d=EXP/f'results/ghi_heads_new_r_seed{seed}_v1'
        per={}
        for s in ('sili','zhujia'):
            ckpt=torch.load(d/s/'AC_best.pt',map_location=device,weights_only=False)
            model=LocalGHIHead('AC',2).to(device);model.load_state_dict(ckpt['model']);model.eval()
            sel=np.flatnonzero(station==s)
            pred0=[];pred1=[];pred_saved=[]
            with torch.inference_mode():
                for st in range(0,len(sel),512):
                    ix=sel[st:st+512]; ti=torch.as_tensor(ix,device=device)
                    pred0.append((model(X[ti],G[ti],C0[ti])*clear[ti]).cpu().numpy())
                    pred1.append((model(X[ti],G[ti],C1[ti])*clear[ti]).cpu().numpy())
            p0=np.concatenate(pred0).astype(np.float64);p1=np.concatenate(pred1).astype(np.float64)
            # Existing predictions use all validation rows; map exact pack indices in station-specific order.
            val_station_indices=[k for k,r in enumerate(rows) if r['split']=='val' and r['station']==s]
            subrow_to_local={row_i:j for j,row_i in enumerate(val_station_indices)}
            saved=np.load(d/s/'AC_val_prediction_ghi_wm2.npy').astype(np.float64)
            p_saved=saved[np.asarray([subrow_to_local[int(gi[i])] for i in sel])]
            a_saved=np.load(d/s/'A_val_prediction_ghi_wm2.npy').astype(np.float64)
            p_a=a_saved[np.asarray([subrow_to_local[int(gi[i])] for i in sel])]
            per[s]={"n":len(sel),"saved_reproduction_max_abs_wm2":float(np.max(np.abs(p0-p_saved))),
                    "A_rmse":float(np.sqrt(np.mean((p_a-GH[sel])**2))),
                    "AC_rmse":float(np.sqrt(np.mean((p0-GH[sel])**2))),
                    "AC_oracle_clear_gate_rmse":float(np.sqrt(np.mean((p1-GH[sel])**2))),
                    "AC_to_gated_prediction_mean_abs_change":float(np.mean(np.abs(p1-p0))),
                    "AC_to_gated_ghi_rmse_delta":float(np.sqrt(np.mean((p1-GH[sel])**2))-np.sqrt(np.mean((p0-GH[sel])**2)))}
            outputs.setdefault(str(seed),{})[s]={"rows":sel,"pA":p_a,"p0":p0,"p1":p1,"metrics":per[s]}
        outputs[str(seed)]['metrics']=per

    # Pair bootstrap by BJT target date, keeping station, seed, and arm paired.
    dates=np.unique(date); rng=np.random.default_rng(20261014); boot=[]
    for _ in range(2000):
        sampled=rng.choice(dates,size=len(dates),replace=True)
        mult={d:int(np.sum(sampled==d)) for d in dates}
        diffs=[]
        for seed in ('42','43','44'):
            site_delta=[]
            for s in ('sili','zhujia'):
                z=outputs[seed][s]; sel=z['rows']; weights=np.asarray([mult.get(date[i],0) for i in sel],np.float64)
                if weights.sum()==0:continue
                e0=z['p0']-GH[sel];e1=z['p1']-GH[sel]
                site_delta.append(float(np.sqrt(np.sum(weights*e1**2)/weights.sum())-
                                        np.sqrt(np.sum(weights*e0**2)/weights.sum())))
            diffs.append(float(np.mean(site_delta)))
        boot.append(float(np.mean(diffs)))
    arm_deltas=[outputs[s]['metrics'][st]['AC_to_gated_ghi_rmse_delta'] for s in ('42','43','44') for st in ('sili','zhujia')]
    boot_a=[]
    for _ in range(2000):
        sampled=rng.choice(dates,size=len(dates),replace=True); mult={d:int(np.sum(sampled==d)) for d in dates}; diffs=[]
        for seed in ('42','43','44'):
            site_delta=[]
            for s in ('sili','zhujia'):
                z=outputs[seed][s];sel=z['rows'];weights=np.asarray([mult.get(date[i],0) for i in sel],np.float64)
                if weights.sum()==0:continue
                e1=z['p1']-GH[sel];ea=z['pA']-GH[sel]
                site_delta.append(float(np.sqrt(np.sum(weights*e1**2)/weights.sum())-
                                        np.sqrt(np.sum(weights*ea**2)/weights.sum())))
            diffs.append(float(np.mean(site_delta)))
        boot_a.append(float(np.mean(diffs)))
    gate_minus_a=float(np.mean([outputs[seed][s]['metrics']['AC_oracle_clear_gate_rmse']-
                                 outputs[seed][s]['metrics']['A_rmse'] for seed in ('42','43','44') for s in ('sili','zhujia')]))
    out={"state":"PASS_FIXED_GHI_HEAD_ORACLE_CLEAR_MASK_DIAGNOSTIC",
         "purpose":"non-deployable CPP-reference clear-sky gate intervention on frozen AC checkpoints",
         "n_common_samples":len(ri),"COT_reference_samples_without_GHI_label":len(missing_keys),
         "sites":{s:int(np.sum(station==s)) for s in ('sili','zhujia')},
         "bjt_target_dates":int(len(dates)),"R_crop_vs_packed_COT_max_abs_log1p_difference":pack_consistency,
         "seeds":{s:outputs[s]['metrics'] for s in ('42','43','44')},
         "station_equal_three_seed_mean_AC_to_gate_RMSE_delta_wm2":float(np.mean(arm_deltas)),
         "paired_BJT_date_bootstrap_95CI_station_equal_seed_mean_wm2":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
         "station_equal_three_seed_mean_gated_AC_minus_A_rmse_wm2":gate_minus_a,
         "paired_BJT_date_bootstrap_95CI_gated_AC_minus_A_wm2":[float(np.quantile(boot_a,.025)),float(np.quantile(boot_a,.975))],
         "bootstrap_replicates":2000,"test_payload_opened":False,
         "sources":{"pack_manifest_sha256":sha(PACK/'pack_manifest.json'),"rows_sha256":sha(PACK/'rows.csv'),
                    "prediction_sha256":sha(R_DIR/'validation_predictions_monthly_R_seed42.npz'),
                    "clp_sha256":sha(CLP_DIR/'clp_reference_monthly_R_seed42.npz')},
         "limitations":["Reference CLP mask is unavailable at inference; this is an oracle upper-bound diagnostic, not a deployable system",
                         "Only 1,398 validation keys with CPP reference are evaluated, not all 4,479 GHI validation rows",
                         "Fixed AC heads were trained on original R-COT; changing COT at validation is an input intervention, not retraining",
                         "CPP CLP and COT share a retrieval product lineage"]}
    # Remove internal arrays from serialized result.
    (OUT/'fixed_ghi_oracle_clear_gate_20261014.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k!='sources'},indent=2))

if __name__=='__main__': main()
