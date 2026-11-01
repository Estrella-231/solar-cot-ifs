"""Hubei minute observations -> right-ended 15-minute labels.

Only pass permitted train/validation records to this module. Units follow the
legacy project's W/m2 contract; instrument metadata remains separately tracked.
"""
from collections import defaultdict
from datetime import timedelta
import math


def minute_index(timestamp, target):
    start = target - timedelta(minutes=15)
    if not start < timestamp <= target:
        return None
    delta = timestamp - start
    micros = (delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds
    return (micros - 1) // 60000000


def aggregate_window(records, target):
    """Average 15 right-closed minute bins inside (target-15min, target].

    Actual timestamps are filtered before binning. Duplicate timestamps with
    conflicting values invalidate a window; identical duplicates count once.
    Missing, nonfinite and negative values are invalid. Valid zeros stay.
    """
    start = target - timedelta(minutes=15)
    timestamps = {}
    for timestamp, value in records:
        if not start < timestamp <= target:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            return {"valid": False, "reason": "nonnumeric", "ghi": None}
        if not math.isfinite(value) or value < 0:
            return {"valid": False, "reason": "nonfinite_or_negative", "ghi": None}
        if timestamp in timestamps and timestamps[timestamp] != value:
            return {"valid": False, "reason": "conflicting_duplicate", "ghi": None}
        timestamps[timestamp] = value
    bins = defaultdict(list)
    for timestamp, value in timestamps.items():
        # Integer microseconds avoid rounding near an exact minute boundary.
        index = minute_index(timestamp, target)
        assert 0 <= index < 15
        bins[index].append(value)
    if len(bins) != 15:
        return {"valid": False, "reason": "missing_minute_bin", "ghi": None,
                "minute_bins": len(bins)}
    means = [math.fsum(bins[k]) / len(bins[k]) for k in range(15)]
    return {"valid": True, "reason": "complete", "ghi": math.fsum(means) / 15,
            "minute_bins": 15, "unique_records": len(timestamps)}
