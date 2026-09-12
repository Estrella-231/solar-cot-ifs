"""Independent saved-array verification; does not import the source checker."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path(__file__).resolve().parents[1]
    reportpath=root/'audits/cpp_spatial_payload_20260908_v1_hyperslab.json'
    valuespath=reportpath.with_suffix('.npz')
    out=root/'audits/cpp_spatial_independent_recheck_20260908_v1.json'
    assert not out.exists()
    source=json.loads(reportpath.read_text())
    selectionpath=root/'audits/cot_spatial_reaudit_selection_20260908_v1.json'
    selection=json.loads(selectionpath.read_text())
    assert source['state']=='PASS_12_SOURCE_SIDECAR_PACK_PIXEL_CHAINS'
    assert len(source['samples'])==len(selection['samples'])==12
    checks=[]
    with np.load(valuespath,allow_pickle=False) as z:
        for i,(r,s) in enumerate(zip(source['samples'],selection['samples'])):
            assert int(s['index'])==r['r_pack_index'] and s['station_id']==r['station'] and s['timestamp_utc']==r['timestamp_utc']
            def get(k):return z[f's{i:02d}_'+k]
            lat,lon=get('source_LAT_axis'),get('source_LON_axis')
            y,x=get('target_lat'),get('target_lon')
            # Exhaustive axis search, independently from the runner's searchsorted.
            rows=np.abs(lat.astype(np.float64)[:,None]-y[:,0].astype(np.float64)[None,:]).argmin(0)
            cols=np.abs(lon.astype(np.float64)[:,None]-x[0,:].astype(np.float64)[None,:]).argmin(0)
            assert np.array_equal(np.broadcast_to(lat[rows,None],y.shape),y)
            assert np.array_equal(np.broadcast_to(lon[None,cols],x.shape),x)
            assert np.array_equal(rows,get('sidecar_rows')) and np.array_equal(cols,get('sidecar_cols'))
            assert np.array_equal(np.broadcast_to(rows[:,None],(16,16)),get('derived_rows'))
            assert np.array_equal(np.broadcast_to(cols[None,:],(16,16)),get('derived_cols'))
            cot=get('source_COT')
            assert np.array_equal(cot,get('sidecar_COT'),equal_nan=True)
            finite=np.isfinite(cot)
            assert np.array_equal(finite,get('sidecar_finite'))
            day,soz=get('day'),get('solar_zenith_deg')
            mask=finite&(cot>=0)&(cot<=100)&np.isfinite(day)&(day>.5)&np.isfinite(soz)&(soz>=0)&(soz<80)
            assert np.array_equal(mask,get('sidecar_mask')) and np.array_equal(mask,get('pack_mask'))
            expected=np.where(mask,cot/np.float32(100),np.float32(0)).astype(np.float32)
            assert np.array_equal(expected,get('pack_target'))
            assert int(mask.sum())==r['valid_pixels']
            checks.append({'station':s['station_id'],'split':s['split'],'index':int(s['index']),
                           'coordinate_exact':True,'source_sidecar_COT_exact':True,
                           'mask_exact':True,'scaled_pack_target_exact':True,'valid_pixels':int(mask.sum())})
    final={'state':'PASS_INDEPENDENT_CPP_SAVED_PIXEL_CHAIN','samples':checks,
           'source_report_sha256':sha(reportpath),'source_arrays_sha256':sha(valuespath),
           'selection_file_sha256':sha(selectionpath),'script_sha256':sha(Path(__file__)),
           'prior_checker_shape_failure':'cpp_spatial_independent_checker_shape_failure_20260908.json; explicit broadcasting correction only',
           'scope':'Independent recomputation from saved source/sidecar/pack slices; original NC not reread; producer pixel-coordinate correctness remains unverified.',
           'test_used':False,'GHI_read':False,'training':False}
    out.write_text(json.dumps(final,indent=2)+'\n')
    print(json.dumps({'state':final['state'],'samples':len(checks)}))


if __name__=='__main__':main()
