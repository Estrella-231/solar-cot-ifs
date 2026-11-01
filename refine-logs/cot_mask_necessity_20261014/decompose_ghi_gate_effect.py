from __future__ import annotations

import csv, json, sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / "experiments/monthly_R_station16_20261013_v1"
PACK = EXP / "results/ghi_head_pack_new_r_v1"
CLP_DIR = ROOT / "refine-logs/cot_value_first_20261013"


def metrics(y, p0, p1):
    e0, e1 = p0-y, p1-y
    d = p1-p0
    return {
        "n": int(len(y)),
        "baseline_rmse": float(np.sqrt(np.mean(e0**2))),
        "gated_rmse": float(np.sqrt(np.mean(e1**2))),
        "rmse_delta": float(np.sqrt(np.mean(e1**2))-np.sqrt(np.mean(e0**2))),
        "baseline_bias": float(e0.mean()),
        "gated_bias": float(e1.mean()),
        "prediction_delta_mean": float(d.mean()),
        "prediction_delta_mae": float(np.abs(d).mean()),
        "sqerr_delta_mean": float(np.mean(e1**2-e0**2)),
        "cross_term_2e0d": float(np.mean(2*e0*d)),
        "correction_square_d2": float(np.mean(d**2)),
        "fraction_improved": float(np.mean(e1**2 < e0**2)),
    }


def main():
    rows = list(csv.DictReader((PACK/"rows.csv").open(newline="", encoding="utf-8")))
    gi_by_key = {(r["station"], r["target_time_utc"]): i for i, r in enumerate(rows)
                 if r["split"] == "val"}
    rp = np.load(CLP_DIR/"validation_predictions_monthly_R_seed42.npz")
    cp = np.load(CLP_DIR/"clp_reference_monthly_R_seed42.npz")
    rkeys = [(str(s), str(t)) for s, t in zip(cp["station"], cp["target_time_utc"])]
    pairs = [(j, gi_by_key[k]) for j,k in enumerate(rkeys) if k in gi_by_key]
    ri = np.asarray([x[0] for x in pairs]); gi = np.asarray([x[1] for x in pairs])
    pred_log = np.asarray(rp["pred_log1p_cot"][ri,0], np.float32)
    clp = np.asarray(cp["clp"][ri,0], int)
    cot_truth = np.asarray(rp["reference_cot"][ri,0], np.float64)
    cot_mask = np.asarray(rp["mask"][ri,0], bool)
    cot0 = np.expm1(pred_log.astype(np.float64))
    cot1 = cot0.copy()
    cot1[clp == 0] = 0.0
    clear3 = (clp[:,7:10,7:10] == 0)
    clear_frac = clear3.mean(axis=(1,2))
    center_clear = clear3[:,1,1]
    meta = json.loads((PACK/"pack_manifest.json").read_text(encoding="utf-8"))
    cot_pack = np.load(PACK/"cot_log1p.npy", mmap_mode="r")
    c0 = np.asarray(cot_pack[gi, 0], np.float32)
    gate_log = pred_log[:,7:10,7:10].copy()
    gate_log[clear3] = 0.
    c1 = ((gate_log-float(meta["COT_norm_mean"]))/float(meta["COT_norm_std"])).astype(np.float32)
    am = np.asarray(meta["AGRI_norm_mean"], np.float32)[None,:,None,None]
    ast = np.asarray(meta["AGRI_norm_std"], np.float32)[None,:,None,None]
    ag = np.load(PACK/"agri_raw.npy", mmap_mode="r")
    av = np.load(PACK/"agri_valid.npy", mmap_mode="r").astype(bool)
    geo = np.load(PACK/"geometry.npy", mmap_mode="r")
    X = np.where(av[gi], (np.asarray(ag[gi])-am)/ast, 0).astype(np.float32)
    G = np.asarray(geo[gi], np.float32)
    y = np.asarray([float(rows[i]["ghi"]) for i in gi], np.float64)
    cs = np.asarray([float(rows[i]["clear"]) for i in gi], np.float64)
    station = np.asarray([rows[i]["station"] for i in gi])
    target_time = np.asarray([rows[i]["target_time_utc"] for i in gi])
    # The frozen AC head predicts kt; restore GHI with the row's clear-sky irradiance.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    X=torch.from_numpy(X).to(device); G=torch.from_numpy(G).to(device)
    C0=torch.from_numpy(((c0-float(meta["COT_norm_mean"]))/float(meta["COT_norm_std"]))[:,None]).to(device)
    C1=torch.from_numpy(c1[:,None]).to(device)
    cs_t=torch.from_numpy(cs.astype(np.float32)).to(device)
    sys.path.insert(0, str(EXP))
    from head_candidate_v0 import LocalGHIHead
    records=[]
    all_outputs={}
    for seed in (42,43,44):
        d=EXP/f"results/ghi_heads_new_r_seed{seed}_v1"
        for st in ("sili","zhujia"):
            ix=np.flatnonzero(station==st)
            ck=torch.load(d/st/"AC_best.pt",map_location=device,weights_only=False)
            model=LocalGHIHead("AC",2).to(device); model.load_state_dict(ck["model"]); model.eval()
            p0=[];p1=[]
            with torch.inference_mode():
                for start in range(0,len(ix),512):
                    q=torch.as_tensor(ix[start:start+512],device=device)
                    p0.append((model(X[q],G[q],C0[q])*cs_t[q]).cpu().numpy())
                    p1.append((model(X[q],G[q],C1[q])*cs_t[q]).cpu().numpy())
            p0=np.concatenate(p0).astype(np.float64); p1=np.concatenate(p1).astype(np.float64)
            all_outputs[(seed,st)]=(ix,p0,p1)
            records.append({"seed":seed,"station":st,"stratum":"all",**metrics(y[ix],p0,p1)})
            # Cloud-mask density is a property of the 3x3 station input, not the full patch.
            bins=[("0",clear_frac==0),("(0,1/3]",(clear_frac>0)&(clear_frac<=1/3)),
                  ("(1/3,2/3]",(clear_frac>1/3)&(clear_frac<=2/3)),
                  ("(2/3,1]",(clear_frac>2/3)&(clear_frac<1)),("1",clear_frac==1)]
            for label, mask in bins:
                sel=ix[mask[ix]]
                if len(sel): records.append({"seed":seed,"station":st,"stratum":f"clear_fraction_{label}",**metrics(y[sel],p0[np.isin(ix,sel)],p1[np.isin(ix,sel)])})
            for label,mask in (("center_clear",center_clear),("center_cloud",~center_clear)):
                sel=ix[mask[ix]]
                if len(sel): records.append({"seed":seed,"station":st,"stratum":label,**metrics(y[sel],p0[np.isin(ix,sel)],p1[np.isin(ix,sel)])})
            for label,mask in (("clear_sky_ghi_<300",cs<300),("clear_sky_ghi_300_600",(cs>=300)&(cs<600)),("clear_sky_ghi_>=600",cs>=600)):
                sel=ix[mask[ix]]
                if len(sel): records.append({"seed":seed,"station":st,"stratum":label,**metrics(y[sel],p0[np.isin(ix,sel)],p1[np.isin(ix,sel)])})
    # Aggregate across models in a way that preserves paired sample rows.
    by_stratum={}
    for rec in records:
        by_stratum.setdefault(rec["stratum"],[]).append(rec)
    # Exact cross-term decomposition averaged over seed/site is the mechanism diagnostic.
    summary=[]
    for name, rs in by_stratum.items():
        summary.append({"stratum":name,"seed_site_equal_n_mean":float(np.mean([r["n"] for r in rs])),
                        **{k:float(np.mean([r[k] for r in rs])) for k in ("baseline_rmse","gated_rmse","rmse_delta","baseline_bias","gated_bias","prediction_delta_mean","prediction_delta_mae","sqerr_delta_mean","cross_term_2e0d","correction_square_d2","fraction_improved")}})
    OUT=HERE/"fixed_ghi_gate_error_decomposition_20261014"
    OUT.mkdir(exist_ok=True)
    fields=list(records[0].keys())
    with (OUT/"stratified_metrics.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
    result={"state":"PASS_ROW_LEVEL_PAIRED_DECOMPOSITION","n_common":len(ri),"sites":{s:int((station==s).sum()) for s in ("sili","zhujia")},
            "center_3x3_clear_fraction_counts":{str(k):int(v) for k,v in zip(*np.unique(clear_frac,return_counts=True))},
            "COT_log1p_norm_mean":meta["COT_norm_mean"],"COT_log1p_norm_std":meta["COT_norm_std"],
            "clear_COT_physical_equivalent_zero_normalized_value":float((0-meta["COT_norm_mean"])/meta["COT_norm_std"]),
            "COT_RMSE_spatial_decomposition_station_equal":{},
            "stratum_seed_site_equal_means":summary,"details_csv":"stratified_metrics.csv",
            "interpretation_note":"The decomposition is for an oracle intervention on frozen AC heads, not a retrained deployable system."}
    for region in ("whole_16x16", "GHI_input_3x3", "outside_GHI_input"):
        site_values=[]
        for st in ("sili","zhujia"):
            pick=station==st
            e0=cot0[pick]-cot_truth[pick]; e1=cot1[pick]-cot_truth[pick]
            m=cot_mask[pick].copy()
            if region=="GHI_input_3x3":
                m=m[:,7:10,7:10]; e0=e0[:,7:10,7:10]; e1=e1[:,7:10,7:10]
            elif region=="outside_GHI_input":
                m[:,7:10,7:10]=False
            vals0=e0[m]; vals1=e1[m]
            site_values.append({"station":st,"n_pixels":int(len(vals0)),"before_rmse":float(np.sqrt(np.mean(vals0**2))),"oracle_gated_rmse":float(np.sqrt(np.mean(vals1**2)))})
        result["COT_RMSE_spatial_decomposition_station_equal"][region]={"site_results":site_values,"mean_before_rmse":float(np.mean([z["before_rmse"] for z in site_values])),"mean_gated_rmse":float(np.mean([z["oracle_gated_rmse"] for z in site_values]))}
    (OUT/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
