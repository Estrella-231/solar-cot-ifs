"""Four fixed station/channel raw-HDF witnesses; no model, test, or GHI."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    assert not (a.output/'report.json').exists(), 'preserve previous output'
    base = Path('/home/Data_Pool_3/chenyi/Auxiliary_data/HuNan_AGRI_Preprocessed')
    grid = base/'grid_static.npz'
    assert sha(grid) == '54a38fcd217a74e0def533217580b48132ef25fee40ed9ed51063e4efdd42c62'
    aux = base/'aux_256/202404/202404020800_1_aux.npz'
    expected = '/home/Data_Pool/data/FY/FY4B/AGRI/4KM/2024/20240402/FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20240402000000_20240402001459_4000M_V0001.HDF'
    with np.load(aux, allow_pickle=False) as z:
        assert str(z['source_hdf'].item()) == expected
        source_size = int(z['source_size_bytes'].item())
        source_mtime = int(z['source_mtime_ns'].item())
        validbits = z['agri_valid_bits'].copy()
    source = Path(expected)
    before = source.stat()
    assert before.st_size == source_size and before.st_mtime_ns == source_mtime
    # The full path is derived from nominal UTC, unlike the BJT auxiliary path.
    fullpath = base/'data/2024/202404/20240402/202404020000.npy'
    assert fullpath.exists(), 'known full AGRI source path unavailable'
    full = np.load(fullpath, mmap_mode='r', allow_pickle=False)
    assert full.shape == (20, 256, 256)
    with np.load(grid, allow_pickle=False) as z:
        gridrow, gridcol = z['row_index'], z['col_index']
    patches = [('sili', (155,171,140,156)), ('zhujia', (65,81,142,158))]
    channels = [(2,1), (13,12)]
    saved, checks, source_meta = {}, [], {}
    with h5py.File(source, 'r') as f:
        assert int(np.asarray(f.attrs['Begin Line Number']).reshape(-1)[0]) == 0
        assert int(np.asarray(f.attrs['Begin Pixel Number']).reshape(-1)[0]) == 0
        for station, bounds in patches:
            r0,r1,c0,c1 = bounds
            rr = gridrow[r0:r1,c0:c1].astype(np.int64)
            cc = gridcol[r0:r1,c0:c1].astype(np.int64)
            saved[station+'_hdf_rows'], saved[station+'_hdf_cols'] = rr,cc
            for channel, index in channels:
                data=f[f'Data/NOMChannel{channel:02d}']; lutdata=f[f'Calibration/CALChannel{channel:02d}']
                block=data[int(rr.min()):int(rr.max())+1,int(cc.min()):int(cc.max())+1]
                dn=block[rr-rr.min(),cc-cc.min()]
                lut=lutdata[:].astype(np.float32)
                dmin,dmax=np.asarray(data.attrs['valid_range']).reshape(-1)
                lmin,lmax=np.asarray(lutdata.attrs['valid_range']).reshape(-1)
                valid=(dn>=dmin)&(dn<=dmax)&(dn<len(lut))&(dn!=int(np.asarray(data.attrs['FillValue']).reshape(-1)[0]))
                physical=np.full((16,16),np.nan,np.float32)
                physical[valid]=lut[dn[valid]]
                valid &= np.isfinite(physical)&(physical>=lmin)&(physical<=lmax)
                physical[~valid]=np.nan
                original=np.array(full[index,r0:r1,c0:c1],copy=True)
                # Preserve the existing full-data visible-channel night-zeroing rule.
                if channel<=6:
                    physical[full[18,r0:r1,c0:c1]>=90]=0
                aux_valid=(validbits[r0:r1,c0:c1]&(1<<index))!=0
                common=np.isfinite(physical)&np.isfinite(original)
                diff=np.abs(physical[common].astype(np.float64)-original[common].astype(np.float64))
                equal=bool(np.array_equal(physical,original,equal_nan=True))
                checks.append({'station':station,'channel':channel,'full_channel_index':index,
                               'valid_pixels':int(valid.sum()),'exact_values_and_nan':equal,
                               'aux_valid_exact':bool(np.array_equal(valid,aux_valid)),
                               'max_abs':float(diff.max()) if diff.size else None})
                prefix=f'{station}_C{channel:02d}_'
                saved.update({prefix+'DN':dn,prefix+'calibrated':physical,prefix+'full_AGRI':original,prefix+'valid':valid,prefix+'aux_valid':aux_valid})
                source_meta[str(channel)]={'data_shape':list(data.shape),'LUT_shape':list(lut.shape),'DN_valid_range':[int(dmin),int(dmax)],'LUT_range':[float(lmin),float(lmax)]}
    after=source.stat()
    assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    np.savez_compressed(a.output/'values.npz',**saved)
    report={'state':'PASS_FIXED_RAW_HDF_TO_FULL_PATCH' if all(c['exact_values_and_nan'] and c['aux_valid_exact'] for c in checks) else 'MISMATCH_FIXED_RAW_HDF_TO_FULL_PATCH',
            'selection':'first train UTC target 2024-04-02T00:00:00Z at both stations; preselected channels C02/C13',
            'source_hdf':str(source),'source_size_bytes':source_size,'source_mtime_ns':source_mtime,
            'source_HDF_hashed_in_full':False,'source_identity_verified_with_aux':True,
            'grid_sha256':sha(grid),'aux_sha256':sha(aux),'full_path':str(fullpath),
            'values_sha256':sha(a.output/'values.npz'),'script_sha256':sha(Path(__file__)),
            'checks':checks,'source_metadata':source_meta,'test_used':False,'training':False,'GHI_read':False,
            'scope':'Four station/channel witnesses only. Proves values follow stored true HDF indices and LUT; not independent absolute geolocation calibration.'}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
