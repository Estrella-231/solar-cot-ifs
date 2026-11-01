"""Read-only validation join and frozen-head counterfactuals; never train or read test."""
import argparse, csv, hashlib, importlib.util, json
from pathlib import Path
import numpy as np
import torch

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); root=a.root; out=a.output; out.mkdir(parents=True,exist_ok=False)
    runs=root/'results/paired_gate_ghi_seed42_20261015_attempt1'
    rawpack=root/'data/ghi_head_pack_new_r_v1'; gatepack=root/'data/ghi_head_pack_gate_c1_20261015_attempt1'
    rawbank=root/'results/new_monthly_R_seed42_cot_bank_v1'; gatebank=root/'results/conservative_gate_cot_bank_20261015_attempt1'
    c1=root/'c1_conservative_gate_20261015/output_attempt3'
    rows=list(csv.DictReader((rawpack/'rows.csv').open())); ids=np.array([i for i,r in enumerate(rows) if r['split']=='val'])
    vr=[rows[i] for i in ids]; keys=[(r['station'],r['target_time_utc']) for r in vr]
    assert len(keys)==len(set(keys))==4479
    with np.load(c1/'validation_predictions.npz',allow_pickle=False) as z: ref={k:z[k] for k in z.files}
    refkeys=list(zip(ref['station'].tolist(),ref['target_time_utc'].tolist())); assert len(refkeys)==len(set(refkeys))
    rmap={k:i for i,k in enumerate(refkeys)}; refidx=np.array([rmap.get(k,-1) for k in keys])
    load=lambda p:np.load(p,mmap_mode='r',allow_pickle=False)
    bi=np.array([int(r['bank_index']) for r in vr]); si=np.array([int(r['station_index']) for r in vr])
    raw16=np.array(load(rawbank/'cot_log1p.npy')[bi,si,0]); gate16=np.array(load(gatebank/'cot_log1p.npy')[bi,si,0])
    p16=np.array(load(gatebank/'p_cloud.npy')[bi,si],dtype=np.float32)
    x=np.array(load(rawpack/'agri_raw.npy')[ids]); valid=np.array(load(rawpack/'agri_valid.npy')[ids],bool)
    g=np.array(load(rawpack/'geometry.npy')[ids]); cr=np.array(load(rawpack/'cot_log1p.npy')[ids]); cg=np.array(load(gatepack/'cot_log1p.npy')[ids])
    np.testing.assert_array_equal(raw16[:,None,7:10,7:10],cr); np.testing.assert_array_equal(gate16[:,None,7:10,7:10],cg)
    matched=np.where(refidx>=0)[0]; ri=refidx[matched]
    # C1 audit outputs and the regenerated bank are distinct saved artifacts.
    # Score actual bank inputs; retain their numeric disagreement, never silently substitute C1 values.
    cot_disagreement=np.abs(np.expm1(raw16[matched])-ref['raw_cot'][ri])
    cleared=(raw16>0)&(gate16==0); assert np.all((raw16==gate16)|(gate16==0))
    false=np.zeros_like(cleared); known=np.zeros_like(cleared); clear_ref=np.zeros_like(cleared)
    known[matched]=ref['valid'][ri]
    false[matched]=ref['valid'][ri]&np.isin(ref['clp'][ri],(1,2))&cleared[matched]
    clear_ref[matched]=ref['valid'][ri]&(ref['clp'][ri]==0)&cleared[matched]
    restore=cg.copy(); f3=false[:,None,7:10,7:10]; restore[f3]=cr[f3]
    restore_clear=cg.copy(); c3=clear_ref[:,None,7:10,7:10];restore_clear[c3]=cr[c3]
    restore_known=cg.copy(); k3=known[:,None,7:10,7:10];restore_known[k3]=cr[k3]
    source_x=load(root/'data/pack_trainval_v1/x_raw.npy')
    real_x=np.array(source_x[ref['sample_indices']])
    outdata=dict(station=np.array([k[0] for k in keys]),utc=np.array([k[1] for k in keys]),refidx=refidx,
        raw16_log=raw16,gate16_log=gate16,prob16=p16,cleared=cleared,falseclear=false,known=known,cleared_CPPclear=clear_ref,
        real_x=real_x,ref_clp=ref['clp'],ref_cot=ref['reference_cot'],ref_valid=ref['valid'],
        ref_station=ref['station'],ref_utc=ref['target_time_utc'],ghi=np.array([float(r['ghi']) for r in vr]),
        clear=np.array([float(r['clear']) for r in vr]),geometry=g)
    spec=importlib.util.spec_from_file_location('heads',root/'head_candidate_v0.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    torch.set_num_threads(4); reloads={}
    for station in ('sili','zhujia'):
        ii=np.where(outdata['station']==station)[0]
        for arm,pack in [('raw',rawpack),('gate',gatepack)]:
            meta=json.loads((pack/'pack_manifest.json').read_text()); chk=torch.load(runs/station/arm/'best.pt',map_location='cpu',weights_only=False)
            model=mod.LocalGHIHead('AC',2).eval(); model.load_state_dict(chk['model'])
            xn=np.where(valid[ii],(x[ii]-np.array(meta['AGRI_norm_mean'],np.float32)[None,:,None,None])/np.array(meta['AGRI_norm_std'],np.float32)[None,:,None,None],0).astype(np.float32)
            for treatment,c in [('rawinput',cr),('gateinput',cg),('restore_falsecloud',restore),('restore_CPPclear',restore_clear),('restore_CPPknown',restore_known)]:
                cn=((c[ii]-np.float32(meta['COT_norm_mean']))/np.float32(meta['COT_norm_std'])).astype(np.float32)
                with torch.inference_mode():
                    yp=np.concatenate([model(torch.from_numpy(xn[j:j+512]),torch.from_numpy(g[ii[j:j+512]]),torch.from_numpy(cn[j:j+512])).numpy() for j in range(0,len(ii),512)])
                pred=yp*outdata['clear'][ii].astype(np.float32)
                key=arm+'head_'+treatment
                if key not in outdata: outdata[key]=np.full(len(vr),np.nan)
                outdata[key][ii]=pred
            saved=np.load(runs/station/arm/'val_prediction_ghi_wm2.npy')
            with np.load(runs/station/arm/'val_keys_truth.npz') as z:
                assert z['target_time_utc'].tolist()==outdata['utc'][ii].tolist()
                outdata['ghi'][ii]=z['ghi'].astype(float)
            key=arm+'head_'+arm+'input'
            reloads[station+'_'+arm]=float(np.max(np.abs(outdata[key][ii]-saved)))
            # CPU vs original GPU arithmetic need not be bitwise equal. Counterfactuals
            # use the CPU baseline from this same run; main paired metrics use saved arrays.
            truth=outdata['ghi'][ii]
            rmse_delta=abs(np.sqrt(np.mean((outdata[key][ii]-truth)**2))-np.sqrt(np.mean((saved-truth)**2)))
            reloads[station+'_'+arm+'_rmse_absdiff']=float(rmse_delta)
            assert reloads[station+'_'+arm]<.25 and rmse_delta<.01, reloads
            if arm+'_saved' not in outdata: outdata[arm+'_saved']=np.full(len(vr),np.nan)
            outdata[arm+'_saved'][ii]=saved
    np.savez_compressed(out/'validation_forensics.npz',**outdata)
    meta=dict(state='COMPLETE_READONLY_VALIDATION_EXTRACTION',ghi_val_rows=len(vr),cpp_ref_rows=len(refkeys),
        joined_cpp_ghi_rows=len(matched),test_payloads_read=0,cpu_reload_max_abs_wm2=reloads,
        bank_vs_C1_saved_R_physical_cot_absdiff_mean=float(cot_disagreement.mean()),
        bank_vs_C1_saved_R_physical_cot_absdiff_max=float(cot_disagreement.max()),
        head_source_sha256=sha(root/'head_candidate_v0.py'),result=json.loads((runs/'result.json').read_text()),
        reference_identity='CPP CLP/COT same-product reference, not independent truth',
        oracle_diagnostic='restore only CPP-valid CLP 1/2 pixels actually cleared, using original R values; frozen head and its normalization unchanged',
        real_image_source=str(root/'data/pack_trainval_v1/x_raw.npy'),
        evaluation_mode='same-target observed AGRI; not +15..240-min forecast',
        source_root=str(root))
    (out/'extraction_audit.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2),flush=True)

if __name__=='__main__': main()
