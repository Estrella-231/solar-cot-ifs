"""Independent NumPy check of saved source patches against actual R pack rows."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    out=a.root/'audits/agri_spatial_independent_recheck_20260908_v1.json'
    assert not out.exists()
    selection=a.root/'audits/cot_spatial_reaudit_selection_20260908_v1.json'
    source=a.root/'audits/agri_spatial_payload_20260908_v1_fd.npz'
    assert sha(source)=='8aa4d94ee55f020a845c717e7d25e41b4a738795d67ae498cbc94ebf50f4d30d'
    samples=json.loads(selection.read_text())['samples']
    packpath=a.root/'data/cot_repaired_pack_20260907_v1/x_raw.npy'
    pack=np.load(packpath,mmap_mode='r',allow_pickle=False)
    report={'state':'IN_PROGRESS','checks':[],'selection_sha256':sha(selection),'source_arrays_sha256':sha(source),
            'R_pack_path':str(packpath),'R_pack_read_scope':'only twelve fixed rows; no full-file hash in this independent check',
            'test_used':False,'training':False,'production_module_imported':False}
    channels=[0,1,2,3,4,5,8,9,10,11,12,13,14]
    with np.load(source,allow_pickle=False) as z:
        for i,s in enumerate(samples):
            original=z[f'original_{i}'];halo=z[f'fd_halo_{i}'];geometry=z[f'fd_geometry_{i}']
            assert np.array_equal(original,halo[:,1:-1,1:-1],equal_nan=True)
            # Independent composition of the documented physical channel contract.
            x=np.empty((16,16,16),np.float32)
            x[:13]=original[channels]
            x[13]=np.cos(np.deg2rad(original[18]))
            x[14]=np.cos(np.deg2rad(original[17]-original[15]))
            x[15]=(original[19]>.5).astype(np.float32)
            assert np.array_equal(x[13:],geometry,equal_nan=True)
            assert np.array_equal(x,pack[int(s['index'])],equal_nan=True)
            alt_equal=[]
            for dr in [-1,0,1]:
                for dc in [-1,0,1]:
                    if dr or dc:
                        if np.array_equal(original,halo[:,1+dr:17+dr,1+dc:17+dc],equal_nan=True):alt_equal.append([dr,dc])
            assert not alt_equal
            report['checks'].append({'pack_row':int(s['index']),'station':s['station_id'],'timestamp_utc':s['timestamp_utc'],
                                     'source_vs_full_exact':True,'geometry_exact':True,'R_actual_row_exact':True,
                                     'eight_nonzero_offsets_all_differ':True})
    report.update(state='PASS_INDEPENDENT_AGRI_SOURCE_AND_R_ROWS',script_sha256=sha(Path(__file__)),
                  scope='Saved FD source arrays and actual R rows independently checked; no second full FD/HPC file replay, no model inference.')
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'state':report['state'],'rows':len(report['checks'])}))


if __name__=='__main__':main()
