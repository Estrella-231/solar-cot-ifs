"""Explicit numerical CPP-reference mask, not a retrieval QA decoder.

The inspected custom NetCDF declares COT Range='0 - 100', FillValue=NaN,
and has no independent QA variable. Geometry thresholds are study choices.
This module does not modify existing caches or certify upstream retrievals.
"""
import numpy as np


def cot_reference_mask(cot, source_finite_mask, day_mask, solar_zenith_deg,
                       *, cot_min=0.0, cot_max=100.0, soz_limit_deg=80.0):
    arrays = [np.asarray(x) for x in (cot, source_finite_mask, day_mask, solar_zenith_deg)]
    if len({a.shape for a in arrays}) != 1:
        raise ValueError('all input arrays must have identical shapes')
    cot, finite_mask, day, soz = arrays
    if finite_mask.dtype != np.bool_:
        raise ValueError('source finite mask must be boolean')
    if not np.array_equal(finite_mask, np.isfinite(cot)):
        raise ValueError('source mask is not the audited numerical-finiteness mask')
    if not (0 <= cot_min < cot_max and 0 < soz_limit_deg <= 90):
        raise ValueError('invalid numerical/geometry policy')
    reasons = {
        'missing_or_nonfinite_cot': ~finite_mask,
        'outside_declared_cot_range': finite_mask & ((cot < cot_min) | (cot > cot_max)),
        'not_day': ~np.isfinite(day) | (day <= 0.5),
        'invalid_solar_geometry': ~np.isfinite(soz) | (soz < 0) | (soz > 180),
        'outside_study_solar_geometry': np.isfinite(soz) & (soz >= soz_limit_deg),
    }
    excluded = np.logical_or.reduce(list(reasons.values()))
    return ~excluded, reasons
