"""Pure contract functions for explicit IFS release times and interval flux.

Release times must come from audited records, never inferred from file mtime.
Intervals are (start,end], UTC-aware, and fluxes are W/m2.
"""
from datetime import datetime
import math


def aware(t):
    if not isinstance(t, datetime) or t.tzinfo is None or t.utcoffset() is None:
        raise ValueError('timezone-aware datetime required')


def accumulated_to_flux(points, units):
    """points: same-cycle (aware valid time, cumulative J/m2) pairs.

    Caller groups and validates cycle/product first. Never silently inserts
    a missing endpoint; intervals preserve the supplied native step spans.
    """
    if units not in {'J m-2', 'J m**-2', 'J m^-2'}:
        raise ValueError('accumulated radiation must use J m-2')
    intervals = []
    for (a, x), (b, y) in zip(points, points[1:]):
        aware(a)
        aware(b)
        seconds = (b-a).total_seconds()
        if seconds <= 0 or not all(math.isfinite(v) for v in (x, y)) or y < x:
            raise ValueError('invalid step, missing accumulation, or cycle reset')
        intervals.append((a, b, (y-x)/seconds))
    return intervals


def interval_mean(intervals, start, end):
    aware(start)
    aware(end)
    if end <= start:
        raise ValueError('non-positive target interval')
    cursor, energy = start, 0.0
    for a, b, flux in sorted(intervals, key=lambda x: x[0]):
        aware(a)
        aware(b)
        if b <= a or not math.isfinite(flux):
            raise ValueError('invalid native interval')
        lo, hi = max(a, start), min(b, end)
        if hi <= lo:
            continue
        if lo != cursor:
            raise ValueError('gap or overlapping native intervals')
        energy += flux*(hi-lo).total_seconds()
        cursor = hi
    if cursor != end:
        raise ValueError('incomplete target coverage')
    return energy/(end-start).total_seconds()


def select_cycle(records, init, targets):
    """Records must share an audited product_contract and native step schema.

    Each record: product_contract, product_id, cycle, release, sha256,
    intervals. Targets: sequence of (start,end). Future/unknown releases
    cannot enter selection. Reports rejection reasons.
    """
    aware(init)
    if not targets:
        raise ValueError('no target intervals')
    contracts = {r['product_contract'] for r in records}
    if len(contracts) > 1:
        raise ValueError('mixed product contracts')
    seen, eligible, rejected = {}, [], []
    for r in records:
        aware(r['cycle'])
        key = r['cycle'], r['product_id']
        signature = (r['sha256'], r.get('release'))
        if key in seen:
            if seen[key] != signature:
                raise ValueError('conflicting duplicate cycle product')
            continue
        seen[key] = signature
        release = r.get('release')
        if release is None:
            rejected.append((r['product_id'], 'missing_release'))
            continue
        aware(release)
        if release < r['cycle']:
            raise ValueError('release precedes cycle')
        if release > init or r['cycle'] > init:
            rejected.append((r['product_id'], 'not_available_at_init'))
            continue
        try:
            values = [interval_mean(r['intervals'], a, b) for a, b in targets]
        except ValueError as e:
            rejected.append((r['product_id'], str(e)))
            continue
        eligible.append((r, values))
    eligible.sort(key=lambda rv: (-rv[0]['release'].timestamp(), -rv[0]['cycle'].timestamp(), rv[0]['product_id']))
    if not eligible:
        return None, rejected
    return eligible[0], rejected
