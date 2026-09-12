"""Bounded two-LUT read and independent reuse of the four saved DN patches."""
import os
for thread_key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[thread_key] = '2'
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

SOURCE = '/home/Data_Pool/data/FY/FY4B/AGRI/4KM/2024/20240402/FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20240402000000_20240402001459_4000M_V0001.HDF'
AUX = '/home/Data_Pool_3/chenyi/Auxiliary_data/HuNan_AGRI_Preprocessed/aux_256/202404/202404020800_1_aux.npz'
PREFIX = 'agri_spatial_payload_20260908_v1'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def clean(value):
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode('utf-8', errors='strict')
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value

def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')

def save_arrays(path, arrays):
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)

def read_fd(args):
    import h5py
    report_path = args.output.with_suffix('.json')
    array_path = args.output.with_suffix('.npz')
    assert not report_path.exists() and not array_path.exists(), 'refuse existing output'
    with np.load(AUX, allow_pickle=False) as aux:
        aux_identity = {key: clean(aux[key].item()) for key in ('source_hdf', 'source_size_bytes', 'source_mtime_ns')}
    assert aux_identity == {'source_hdf': SOURCE, 'source_size_bytes': 102281546, 'source_mtime_ns': 1712017800000000000}
    source = Path(SOURCE)
    before = source.stat()
    assert (before.st_size, before.st_mtime_ns) == (aux_identity['source_size_bytes'], aux_identity['source_mtime_ns'])
    arrays, metadata = {}, {}
    with h5py.File(source, 'r') as hdf:
        for channel in (2, 13):
            dataset = hdf[f'Calibration/CALChannel{channel:02d}']
            lut = dataset[:]
            assert lut.shape == (4096,) and lut.dtype == np.dtype('float32')
            arrays[f'CALChannel{channel:02d}'] = lut
            entry = {'path': dataset.name, 'shape': list(lut.shape), 'dtype': str(lut.dtype),
                     'attrs': {key: clean(value) for key, value in dataset.attrs.items()},
                     'array_sha256': hashlib.sha256(lut.tobytes(order='C')).hexdigest()}
            if args.include_dn_attrs:
                dn_dataset = hdf[f'Data/NOMChannel{channel:02d}']
                entry['DN_attrs_only'] = {key: clean(dn_dataset.attrs[key]) for key in ('valid_range', 'FillValue')}
            metadata[str(channel)] = entry
    after = source.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    save_arrays(array_path, arrays)
    report = {'state': 'COMPLETE_TWO_LUT_METADATA_READ', 'source_identity': aux_identity,
              'stat_unchanged_after_read': True, 'source_HDF_hashed_in_full': False,
              'aux_path': AUX, 'aux_sha256': sha(AUX), 'arrays_sha256': sha(array_path),
              'script_sha256': sha(__file__), 'metadata': metadata,
              'payload_read': {'LUT_elements': 8192, 'LUT_bytes': sum(a.nbytes for a in arrays.values()),
                               'DN_elements': 0, 'full_AGRI_elements': 0, 'GHI_elements': 0},
              'DN_attrs_only_read': args.include_dn_attrs, 'test_used': False, 'models_loaded': False}
    write_new(report_path, report)
    print(json.dumps(report))

def local_recompute(args):
    base = args.project
    audit = base/'audits'
    report_path = args.output.with_suffix('.json')
    array_path = args.output.with_suffix('.npz')
    assert not report_path.exists() and not array_path.exists(), 'refuse existing output'
    snapshot_path = audit/(PREFIX+'_peer_lut_snapshot.json')
    snapshot = json.loads(snapshot_path.read_text(encoding='utf-8'))
    lut_path = snapshot_path.with_suffix('.npz')
    assert sha(lut_path) == snapshot['arrays_sha256']
    assert snapshot['script_sha256'] == sha(__file__)
    raw_report_path = audit/'cot_raw_hdf_witness_20260908_v1/report.json'
    raw_report = json.loads(raw_report_path.read_text(encoding='utf-8'))
    values_path = raw_report_path.with_name('values.npz')
    raw_script = raw_report_path.with_name('run_script_snapshot.py')
    assert sha(values_path) == raw_report['values_sha256']
    assert sha(raw_script) == raw_report['script_sha256']
    assert snapshot['source_identity'] == {key: raw_report[key] for key in ('source_hdf', 'source_size_bytes', 'source_mtime_ns')}
    assert snapshot['aux_sha256'] == raw_report['aux_sha256']
    combined_path = audit/(PREFIX+'_fd.npz')
    assert sha(combined_path) == '8aa4d94ee55f020a845c717e7d25e41b4a738795d67ae498cbc94ebf50f4d30d'
    arrays, checks = {}, []
    with np.load(lut_path, allow_pickle=False) as luts, np.load(values_path, allow_pickle=False) as values, np.load(combined_path, allow_pickle=False) as combined:
        for channel, index in ((2, 1), (13, 12)):
            lut = luts[f'CALChannel{channel:02d}'].copy()
            arrays[f'CALChannel{channel:02d}'] = lut
            meta = snapshot['metadata'][str(channel)]
            assert meta['DN_attrs_only'], 'complete independent valid reconstruction requires DN metadata'
            dn_range = np.asarray(meta['DN_attrs_only']['valid_range']).reshape(-1)
            fill = int(np.asarray(meta['DN_attrs_only']['FillValue']).reshape(-1)[0])
            lut_range = np.asarray(meta['attrs']['valid_range']).reshape(-1)
            assert dn_range.tolist() == raw_report['source_metadata'][str(channel)]['DN_valid_range']
            assert lut_range.tolist() == raw_report['source_metadata'][str(channel)]['LUT_range']
            for station, sample in (('sili', 0), ('zhujia', 6)):
                key = f'{station}_C{channel:02d}_'
                dn = values[key+'DN'].astype(np.int64)
                assert dn.shape == (16, 16)
                index_valid = (dn >= dn_range[0]) & (dn <= dn_range[1]) & (dn >= 0) & (dn < lut.size) & (dn != fill)
                raw = np.full(dn.shape, np.nan, dtype=np.float32)
                raw[index_valid] = lut[dn[index_valid]]
                valid = index_valid & np.isfinite(raw) & (raw >= lut_range[0]) & (raw <= lut_range[1])
                reconstructed = np.where(valid, raw, np.float32(np.nan)).astype(np.float32)
                original = combined[f'original_{sample}']
                night = original[18] >= 90
                assert not np.any(night), 'fixed target unexpectedly triggers visible night zeroing'
                comparisons = {name: bool(np.array_equal(reconstructed, other, equal_nan=True)) for name, other in {
                    'equals_saved_calibrated': values[key+'calibrated'], 'equals_saved_full': values[key+'full_AGRI'],
                    'equals_independent_Combined': original[index]}.items()}
                comparisons.update({'valid_equals_saved': bool(np.array_equal(valid, values[key+'valid'])),
                                    'valid_equals_aux': bool(np.array_equal(valid, values[key+'aux_valid']))})
                difference = np.abs(reconstructed.astype(np.float64) - original[index].astype(np.float64))
                checks.append({'station': station, 'channel': channel, 'saved_DN_min': int(dn.min()), 'saved_DN_max': int(dn.max()),
                               'DN_FillValue': fill, 'valid_pixels': int(valid.sum()), 'night_zeroing_triggered': False,
                               'max_abs_vs_independent_Combined': float(np.nanmax(difference)), **comparisons})
                arrays[key+'independently_calibrated'] = reconstructed
                arrays[key+'independent_valid'] = valid
    passed = all(all(c[key] for key in ('equals_saved_calibrated', 'equals_saved_full', 'equals_independent_Combined', 'valid_equals_saved', 'valid_equals_aux')) for c in checks)
    save_arrays(array_path, arrays)
    report = {'state': 'PASS_FOUR_SAVED_DN_INDEPENDENT_LUT_RECONSTRUCTION' if passed else 'MISMATCH_SAVED_DN_LUT_RECONSTRUCTION',
              'selection': raw_report['selection'], 'source_identity': snapshot['source_identity'],
              'LUT_read_report': snapshot_path.name, 'LUT_read_report_sha256': sha(snapshot_path),
              'checks': checks, 'new_arrays_sha256': sha(array_path), 'script_sha256': sha(__file__),
              'input_hashes': {'original_HDF_report': sha(raw_report_path), 'saved_DN_values': sha(values_path),
                               'original_HDF_execution_script': sha(raw_script), 'independent_Combined_patches': sha(combined_path)},
              'payload_scope': snapshot['payload_read'], 'source_HDF_hashed_in_full': False,
              'new_DN_read': False, 'new_Combined_read': False, 'test_used': False, 'GHI_read': False, 'models_loaded': False,
              'limits': ['One previously fixed train target and two channels at two stations only.',
                         'Original HDF identity is bound by exact path, size, mtime and auxiliary identity; no whole-HDF hash.',
                         'DN patches come from the prior witnessed read; this step independently rereads LUTs and recalculates from those saved DN values.']}
    write_new(report_path, report)
    print(json.dumps(report))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('fd', 'local'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--project', type=Path)
    parser.add_argument('--include-dn-attrs', action='store_true')
    arguments = parser.parse_args()
    (read_fd if arguments.stage == 'fd' else local_recompute)(arguments)
