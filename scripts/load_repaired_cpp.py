"""Explicit loader for the original Combined sample plus corrected CPP sidecar."""
import hashlib
import io
from pathlib import Path
import numpy as np


def load_repaired_sample(original_path, sidecar_path):
    """Preserve original fields, replace CPP only; never fall back to old CPP."""
    raw = Path(original_path).read_bytes()
    with np.load(sidecar_path, allow_pickle=False) as fix:
        if str(fix['schema_version'].item()) != 'cpp_aligned_sidecar_v1':
            raise ValueError('unrecognized CPP sidecar schema')
        if hashlib.sha256(raw).hexdigest() != str(fix['original_sha256'].item()):
            raise ValueError('original sample differs from repaired provenance')
        with np.load(io.BytesIO(raw), allow_pickle=False) as old:
            for k in ('station_id','timestamp_utc'):
                if old[k].item() != fix[k].item():
                    raise ValueError('CPP sidecar identity mismatch')
            sample = {k: old[k].copy() for k in old.files}
        for k in ('cpp','cpp_valid_mask','cpp_present','cot_reference_mask','cpp_rows','cpp_cols'):
            sample[k] = fix[k].copy()
        # Old hardcoded crop metadata must not survive the repaired view.
        sample.pop('cpp_crop_bounds', None)
        sample['cpp_alignment_schema'] = fix['schema_version'].copy()
        sample['cpp_source_nc'] = fix['source_nc'].copy()
        sample['cpp_contract_sha256'] = fix['contract_sha256'].copy()
    return sample


def cot_target(sample):
    """Masked COT/100 target; invalid zeros are loss placeholders, not clear sky."""
    valid = sample['cot_reference_mask']
    return np.where(valid, sample['cpp'][0]/100.0, 0).astype(np.float32)[None], valid[None]
