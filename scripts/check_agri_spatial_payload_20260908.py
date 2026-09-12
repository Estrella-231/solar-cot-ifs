"""Fixed 12-row AGRI payload-chain audit. CPU only; never decode GHI/CPP.

fd: original Combined agri versus source full-grid 18x18 halo and auxiliary.
hpc: compare the small FD export to selected R rows and HPC full-grid patches.
All outputs are new, and failures retain their own JSON without overwriting.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import numpy as np

SELECTION_FILE_SHA = 'd2557c7e37d6d55d60a295fb3c7c16a207783cd3928b6e9a8a37f0fb1054a49c'
SELECTION_SHA = 'ae0cbeab4116ba3452f4f0e329b4ca545844c04fa8ad75457e0b6a9c4b50cb2b'
INDICES = [0,1,2,3,4,5,8,9,10,11,12,13,14]
PATCHES = {'sili': [155,171,140,156], 'zhujia': [65,81,142,158]}
GEOMETRY_ABS = 2e-6
GRID_SHA = '54a38fcd217a74e0def533217580b48132ef25fee40ed9ed51063e4efdd42c62'
FD_ROOT = Path('/home/Data_Pool_3/chenyi/Auxiliary_data/HuNan_AGRI_Preprocessed')
HPC_ROOT = Path('/public/home/slfu/ttzhou/Auxiliary_data/HuNan_AGRI_Preprocessed')
PROJECT = Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907')
BUILDER = Path('/home/Data_Pool_3/chenyi/GHI/scripts/legacy_root/build_hunan_station16_samples.py')


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    return sha_bytes(Path(path).read_bytes())


def time_utc(value):
    stamp = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def array_sha(array):
    # dtype/shape accompany the hash, and the numerical representation is preserved.
    value = np.ascontiguousarray(array)
    return dict(shape=list(value.shape), dtype=value.dtype.str, sha256=sha_bytes(value.tobytes()))


def compare(left, right):
    assert left.shape == right.shape
    finite_left, finite_right = np.isfinite(left), np.isfinite(right)
    valid = finite_left & finite_right
    difference = left[valid].astype(np.float64) - right[valid].astype(np.float64)
    return dict(shape=list(left.shape), finite_pairs=int(valid.sum()),
        nonfinite_pattern_equal=bool(np.array_equal(finite_left, finite_right)),
        nan_pattern_equal=bool(np.array_equal(np.isnan(left), np.isnan(right))),
        posinf_pattern_equal=bool(np.array_equal(np.isposinf(left), np.isposinf(right))),
        neginf_pattern_equal=bool(np.array_equal(np.isneginf(left), np.isneginf(right))),
        exact_equal_nan=bool(np.array_equal(left, right, equal_nan=True)),
        max_abs=float(np.abs(difference).max()) if len(difference) else None,
        rmse=float(np.sqrt(np.mean(difference**2))) if len(difference) else None)


def channel_compare(left, right):
    return [dict(channel=i, **compare(left[i], right[i])) for i in range(len(left))]


def physical_summary(array):
    result = []
    for i, channel in enumerate(array):
        finite = channel[np.isfinite(channel)].astype(np.float64)
        result.append(dict(channel=i, finite_count=len(finite), nonfinite_count=int(channel.size-len(finite)),
            minimum=float(finite.min()) if len(finite) else None,
            maximum=float(finite.max()) if len(finite) else None,
            mean=float(finite.mean()) if len(finite) else None))
    return result


def r_raw(agri):
    return np.concatenate([agri[INDICES], np.cos(np.deg2rad(agri[18]))[None],
        np.cos(np.deg2rad(agri[17]-agri[15]))[None], (agri[19] > .5)[None]], axis=0).astype(np.float32)


def metadata(root):
    result = {}
    for name in ('channel_info.json', 'dataset_meta.json'):
        path = root/name
        result[name] = dict(path=str(path), sha256=sha(path), content=json.loads(path.read_text()))
    grid = root/'grid_static.npz'
    result['grid'] = dict(path=str(grid), sha256=sha(grid))
    assert result['grid']['sha256'] == GRID_SHA
    return result


def read_source_patch(root, sample):
    stamp = time_utc(sample['timestamp_utc'])
    path = root/'data'/stamp.strftime('%Y/%Y%m/%Y%m%d/%Y%m%d%H%M.npy')
    record = dict(path=str(path), timestamp_utc=stamp.isoformat(), available=path.is_file())
    if not record['available']:
        return record, {}
    before = path.stat()
    raw = np.load(path, mmap_mode='r', allow_pickle=False)
    assert raw.shape == (20,256,256) and raw.dtype == np.float32
    r0,r1,c0,c1 = PATCHES[sample['station_id']]
    halo = np.array(raw[:,r0-1:r1+1,c0-1:c1+1], copy=True)
    swapped = np.array(raw[:,c0:c1,r0:r1], copy=True)
    assert halo.shape == (20,18,18) and swapped.shape == (20,16,16)
    after = path.stat()
    assert (before.st_size,before.st_mtime_ns) == (after.st_size,after.st_mtime_ns)
    record.update(file_bytes=after.st_size, file_mtime_ns=after.st_mtime_ns,
        full_file_sha_computed=False, full_grid_materialized=False,
        raw_shape=list(raw.shape), raw_dtype=raw.dtype.str, halo=array_sha(halo), swapped=array_sha(swapped))
    return record, dict(halo=halo, swapped=swapped)


def read_aux_patch(root, sample):
    target = time_utc(sample['timestamp_utc']).astimezone(timezone(timedelta(hours=8)))
    candidates = [root/'aux_256'/target.strftime('%Y%m')/(target.strftime('%Y%m%d%H%M')+f'_{day}_aux.npz') for day in (0,1)]
    found = [p for p in candidates if p.is_file()]
    record = dict(expected_BJT=target.isoformat(), exact_candidates=[str(p) for p in candidates], available=bool(found))
    if not found:
        return record, {}
    assert len(found) == 1, 'duplicate aux identity'
    path = found[0]
    before = path.stat()
    r0,r1,c0,c1 = PATCHES[sample['station_id']]
    with np.load(path, allow_pickle=False) as z:
        geometry = np.stack([z[key][r0:r1,c0:c1] for key in ('cosSOZ','cosRAA','day_mask')]).astype(np.float32)
        bits = np.array(z['agri_valid_bits'][r0:r1,c0:c1], copy=True)
        source = {key: str(z[key].item()) for key in ('source_hdf','source_type') if key in z.files}
    after = path.stat()
    assert (before.st_size,before.st_mtime_ns) == (after.st_size,after.st_mtime_ns)
    assert geometry.shape == (3,16,16) and bits.shape == (16,16) and np.isfinite(geometry).all()
    record.update(path=str(path), file_bytes=after.st_size, file_mtime_ns=after.st_mtime_ns,
        source_metadata=source, geometry=array_sha(geometry), valid_bits=array_sha(bits),
        read_scope='only four named auxiliary array members and source identity scalars; NPZ loads full named members before slicing')
    return record, dict(geometry=geometry,bits=bits)


def spatial_alternatives(original, halo, swapped):
    result = []
    for dr in (-1,0,1):
        for dc in (-1,0,1):
            value = halo[:,1+dr:17+dr,1+dc:17+dc]
            result.append(dict(candidate=f'row_offset_{dr:+d}_col_offset_{dc:+d}',
                               **compare(original[INDICES],value[INDICES])))
    center = halo[:,1:17,1:17]
    for name, value in [('transpose',center.transpose(0,2,1)), ('flip_rows',center[:,::-1,:]),
                        ('flip_columns',center[:,:,::-1]), ('flip_both',center[:,::-1,::-1]),
                        ('swap_fullgrid_row_column_origin',swapped)]:
        result.append(dict(candidate=name, **compare(original[INDICES],value[INDICES])))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('fd','hpc'),required=True)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--output-prefix',type=Path,required=True)
    parser.add_argument('--fd-report',type=Path)
    parser.add_argument('--fd-arrays',type=Path)
    args = parser.parse_args()
    output = args.output_prefix.with_suffix('.json')
    arrays_path = args.output_prefix.with_suffix('.npz')
    assert not output.exists() and not arrays_path.exists(), 'new output prefix required'
    assert sha(args.selection) == SELECTION_FILE_SHA
    selection = json.loads(args.selection.read_text())
    assert selection['selection_sha256'] == SELECTION_SHA
    samples = selection['samples']
    assert len(samples)==12 and len({(s['station_id'],s['timestamp_utc']) for s in samples})==12
    assert Counter((s['station_id'],s['split']) for s in samples)=={('sili','train'):3,('sili','validation'):3,('zhujia','train'):3,('zhujia','validation'):3}
    assert all(s['split'] in ('train','validation') for s in samples)
    report = dict(state='IN_PROGRESS', stage=args.stage, started_utc=datetime.now(timezone.utc).isoformat(),
        selection_file_sha256=SELECTION_FILE_SHA, selection_sha256=SELECTION_SHA, script_sha256=sha(__file__),
        threads=2, test_used=False, GHI_CPP_values_decoded=False, models_loaded=False,
        expected_samples=12, samples=[],
        preregistered_tolerances=dict(copied_AGRI='exact_equal_nan with separately equal NaN/+Inf/-Inf patterns',
            R_reconstructed_physical13='exact_equal_nan', R_reconstructed_geometry_absolute=GEOMETRY_ABS,
            auxiliary_geometry='record exact equality and max differences; do not silently substitute aux for raw geometry'),
        spatial_alternatives_policy='fixed 3x3 offsets plus transpose/row flip/column flip/both flip/swapped origin; descriptive only, never select a replacement crop',
        writes_old_data=False, arrays_export={})
    exported = {}
    try:
        root = FD_ROOT if args.stage=='fd' else HPC_ROOT
        report['raw_metadata'] = metadata(root)
        if args.stage=='fd':
            report['Combined_builder'] = dict(path=str(BUILDER), sha256=sha(BUILDER),
                source_provenance_limit='current source snapshot; original builder run did not bind this code SHA into every sample')
        else:
            assert args.fd_report and args.fd_arrays
            fdreport = json.loads(args.fd_report.read_text())
            assert fdreport['selection_sha256']==SELECTION_SHA and len(fdreport['samples'])==12
            assert fdreport['state'] in ('COMPLETE_FD_PAYLOAD_CHECK','COMPLETE_WITH_MAPPING_DIFFERENCES_OR_MISSING_SOURCES')
            assert sha(args.fd_arrays)==fdreport['arrays_export']['sha256']
            fdarrays=np.load(args.fd_arrays,allow_pickle=False)
            report['FD_artifacts']=dict(report_sha256=sha(args.fd_report), arrays_sha256=sha(args.fd_arrays))
            pack=PROJECT/'data/cot_repaired_pack_20260907_v1'
            assert sha(pack/'rows.csv')==selection['rows_csv_sha256']
            pack_hashes=json.loads((pack/'SHA256.json').read_text())
            x=np.load(pack/'x_raw.npy',mmap_mode='r',allow_pickle=False)
            assert x.shape==(18514,16,16,16) and x.dtype==np.float32
            norm=json.loads((pack/'norm.json').read_text())
            assert sha(pack/'norm.json')==pack_hashes['norm.json'] and norm['test_used'] is False
            contract=json.loads((PROJECT/'data/frozen_forecast_trainval_20260907_v1/contract.json').read_text())
            sc=json.loads((PROJECT/'configs/s_frozen_hunan_seed42.json').read_text())
            rc=json.loads((PROJECT/'configs/r_frozen_repaired_seed42.json').read_text())
            assert sc==contract['S'] and rc==contract['R'] and sc['station_patches']==PATCHES
            assert sha(Path(sc['normalization']))==sc['normalization_sha256']
            snorm=json.loads(Path(sc['normalization']).read_text())
            assert snorm['source_indices']==INDICES
            report['reader_sources']={}
            sroot=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/HunanSimVPDDP_20260907/scripts')
            for name in ('hunan_data.py','train_ddp.py'):
                value=sha(sroot/name)
                assert value==contract['imported_source_sha256'][name]
                report['reader_sources'][name]=dict(path=str(sroot/name),sha256=value)
            assert sha(PROJECT/'scripts/cache_frozen_forecast.py')==contract['script_sha256']
            report['R_pack']=dict(path=str(pack),x_raw_recorded_full_SHA=pack_hashes['x_raw.npy'],
                x_raw_full_SHA_recomputed=False,rows_csv_sha256=selection['rows_csv_sha256'],norm_sha256=sha(pack/'norm.json'),
                x_raw_read_scope='12 fixed rows only; saved canonical row hashes below')

        for number,sample in enumerate(samples):
            stamp=time_utc(sample['timestamp_utc'])
            assert stamp==time_utc(sample['timestamp_bjt'])
            bjt=stamp.astimezone(timezone(timedelta(hours=8)))
            lo,hi=(datetime(2024,4,2,tzinfo=bjt.tzinfo),datetime(2025,7,1,tzinfo=bjt.tzinfo)) if sample['split']=='train' else (datetime(2025,7,1,tzinfo=bjt.tzinfo),datetime(2025,10,1,tzinfo=bjt.tzinfo))
            assert lo<=bjt<hi
            row={key:sample[key] for key in ('station_id','timestamp_utc','timestamp_bjt','split','index','selection_slot')}
            row['sample_number']=number
            source,raw=read_source_patch(root,sample)
            auxiliary,aux=read_aux_patch(root,sample)
            row.update(full_source=source,auxiliary=auxiliary)
            if args.stage=='fd':
                path=Path(sample['original_path']);data=path.read_bytes()
                assert sha_bytes(data)==sample['original_sha256']
                with np.load(io.BytesIO(data),allow_pickle=False) as z:
                    assert str(z['station_id'].item())==sample['station_id']
                    assert time_utc(z['timestamp_utc'].item())==stamp
                    assert np.array_equal(z['agri_crop_bounds'],PATCHES[sample['station_id']])
                    original=z['agri'].copy()
                assert original.shape==(20,16,16) and original.dtype==np.float32
                row['original']=dict(path=str(path),sha256=sha_bytes(data),agri=array_sha(original),
                                     source_file_hashed_in_full=True,channel_summary=physical_summary(original))
                exported[f'original_{number}']=original
                if raw:
                    canonical=raw['halo'][:,1:17,1:17]
                    row['original_vs_full_canonical']=compare(original,canonical)
                    row['original_vs_full_per_channel']=channel_compare(original,canonical)
                    row['fixed_spatial_alternatives']=spatial_alternatives(original,raw['halo'],raw['swapped'])
                    exported[f'fd_halo_{number}']=raw['halo']
                if aux:
                    row['raw_geometry_vs_auxiliary']=channel_compare(r_raw(original)[13:],aux['geometry'])
                    exported[f'fd_geometry_{number}']=aux['geometry']
                    exported[f'fd_bits_{number}']=aux['bits']
            else:
                original=fdarrays[f'original_{number}']
                reconstructed=r_raw(original)
                pack_row=np.array(x[int(sample['index'])],copy=True)
                row['R_pack_row']=array_sha(pack_row)
                row['original_reconstructed_vs_R_pack']=channel_compare(reconstructed,pack_row)
                row['R_physical13_exact']=compare(reconstructed[:13],pack_row[:13])
                row['R_geometry_max_abs']=compare(reconstructed[13:],pack_row[13:])
                if raw:
                    canonical=raw['halo'][:,1:17,1:17]
                    row['original_vs_HPC_full_canonical']=compare(original,canonical)
                    row['original_vs_HPC_full_per_channel']=channel_compare(original,canonical)
                    row['fixed_spatial_alternatives']=spatial_alternatives(original,raw['halo'],raw['swapped'])
                    if f'fd_halo_{number}' in fdarrays.files:
                        row['FD_vs_HPC_full_halo']=compare(fdarrays[f'fd_halo_{number}'],raw['halo'])
                if aux:
                    row['raw_geometry_vs_HPC_auxiliary']=channel_compare(reconstructed[13:],aux['geometry'])
                    if f'fd_geometry_{number}' in fdarrays.files:
                        row['FD_vs_HPC_aux_geometry']=compare(fdarrays[f'fd_geometry_{number}'],aux['geometry'])
                        row['FD_vs_HPC_aux_valid_bits']=compare(fdarrays[f'fd_bits_{number}'],aux['bits'])
                    valid=np.stack([(aux['bits'] & (1<<k))!=0 for k in INDICES]) & np.isfinite(original[INDICES])
                    row['S_input_validity']=dict(valid_per_channel=valid.sum((1,2)).tolist(),
                        finite_but_QA_masked_per_channel=(np.isfinite(original[INDICES]) & ~valid).sum((1,2)).tolist())
            report['samples'].append(row)
            print('PAYLOAD_CHECKED',args.stage,number,sample['station_id'],sample['timestamp_utc'],flush=True)

        if args.stage=='fd':
            with arrays_path.open('xb') as stream:
                np.savez(stream,**exported)
            report['arrays_export']=dict(path=str(arrays_path),sha256=sha(arrays_path),
                file_bytes=arrays_path.stat().st_size,array_payload_bytes=sum(v.nbytes for v in exported.values()),
                contents='selected Combined AGRI, FD full 18x18 AGRI halos, selected aux geometry and validity bits only; no GHI/CPP')
            passed=all(r.get('original_vs_full_canonical',{}).get('exact_equal_nan',False) for r in report['samples'])
            report['state']='COMPLETE_FD_PAYLOAD_CHECK' if passed else 'COMPLETE_WITH_MAPPING_DIFFERENCES_OR_MISSING_SOURCES'
        else:
            passed=all(r['R_physical13_exact']['exact_equal_nan'] and
                r['R_geometry_max_abs']['nonfinite_pattern_equal'] and
                r['R_geometry_max_abs']['max_abs'] is not None and r['R_geometry_max_abs']['max_abs']<=GEOMETRY_ABS and
                r.get('original_vs_HPC_full_canonical',{}).get('exact_equal_nan',False) and
                r.get('FD_vs_HPC_full_halo',{}).get('exact_equal_nan',False)
                for r in report['samples'])
            report['state']='PASS_FIXED_12_AGRI_PAYLOAD_CHAIN' if passed else 'COMPLETE_WITH_MAPPING_DIFFERENCES_OR_MISSING_SOURCES'
            report['aux_geometry_scope']='reported separately; payload-chain PASS does not certify raw-derived and auxiliary geometry equivalence'
        report['limits']=['12 rank-selected train/validation source rows only; no claim for all dates.',
            'No raw FY4B HDF DN/LUT recalibration was performed; physical units/order are checked against source metadata and matched arrays.',
            'Full AGRI file identity is exact path plus size/mtime and selected-patch hash, not a new whole-file SHA.',
            'Current Combined builder source snapshot is not an archival per-row build-code hash.',
            'No node-local training cache was read or regenerated. Runtime reader source is hash-bound; matching public raw patches does not certify every prior node-cache value.',
            'Offsets/transforms are fixed diagnostics, not tuned choices or corrected data.']
    except Exception as error:
        report.update(state='ERROR_INCOMPLETE_AGRI_PAYLOAD_CHECK',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        report['finished_utc']=datetime.now(timezone.utc).isoformat()
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('x') as stream:
            json.dump(report,stream,indent=2,allow_nan=False)
        print(report['state'],flush=True)


if __name__=='__main__':
    main()
