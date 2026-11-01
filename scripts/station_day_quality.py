"""Shared station/BJT-day exclusion mask for GHI labels and paired scores."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

POLICY = Path(__file__).resolve().parents[1] / 'docs/GHI_STATION_DAY_EXCLUSIONS_20261019.json'


def ghi_day_keep_mask(stations, *, bjt_dates=None, target_times_utc=None):
    """Return one common method-independent mask; naive UTC strings mean UTC."""
    if (bjt_dates is None) == (target_times_utc is None):
        raise ValueError('Supply exactly one of bjt_dates or target_times_utc')
    station = np.asarray(stations).astype(str)
    if bjt_dates is None:
        dates = []
        for raw in target_times_utc:
            dt = datetime.fromisoformat(str(raw).replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dates.append(dt.astimezone(timezone(timedelta(hours=8))).date().isoformat())
    else:
        dates = [str(d) for d in bjt_dates]
        for d in dates:
            datetime.strptime(d, '%Y-%m-%d')
    if len(dates) != len(station):
        raise ValueError('station/date length mismatch')
    blocked = {(r['station'], r['bjt_date']) for r in json.loads(POLICY.read_text(encoding='utf-8'))['exclusions']}
    return np.array([(s, d) not in blocked for s, d in zip(station, dates)], dtype=bool)
