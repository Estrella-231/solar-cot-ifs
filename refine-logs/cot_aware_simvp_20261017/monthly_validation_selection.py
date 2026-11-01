"""Small NumPy-only helper for baseline-matched four-season model selection."""
from __future__ import annotations

import numpy as np


def validation_score(month_sse: np.ndarray, month_count: np.ndarray) -> tuple[float, dict[str, float]]:
    """Return the unweighted mean of DJF/MAM/JJA/SON pixel RMSEs."""
    month_sse = np.asarray(month_sse, dtype=np.float64)
    month_count = np.asarray(month_count, dtype=np.float64)
    if month_sse.shape != (12,) or month_count.shape != (12,):
        raise ValueError("monthly validation accumulators must have 12 entries")
    if not np.isfinite(month_sse).all() or not np.isfinite(month_count).all():
        raise ValueError("nonfinite validation accumulators")
    if (month_sse < 0).any() or (month_count < 0).any():
        raise ValueError("negative validation SSE/count")
    if np.any((month_count == 0) & (month_sse != 0)):
        raise ValueError("empty validation month must have zero SSE")
    season_of = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    season_sse = np.zeros(4, dtype=np.float64)
    season_count = np.zeros(4, dtype=np.float64)
    for month in range(1, 13):
        season = season_of[month]
        season_sse[season] += month_sse[month - 1]
        season_count[season] += month_count[month - 1]
    if (season_count <= 0).any():
        raise ValueError("each season must have validation support")
    season_rmse = np.sqrt(season_sse / season_count)
    names = ("DJF", "MAM", "JJA", "SON")
    return float(season_rmse.mean()), dict(zip(names, season_rmse.tolist()))
