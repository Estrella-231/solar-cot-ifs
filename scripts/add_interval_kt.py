"""Add deterministic interval clear-sky denominators to the audited GHI base cohort."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    audit = json.loads((a.base/'audit.json').read_text())
    assert audit['state'] == 'BASE_COHORT_VERIFIED'
    assert sha(a.base/'base_cohort.csv') == audit['output_sha256']
    df = pd.read_csv(a.base/'base_cohort.csv')
    clear = np.full(len(df), np.nan)
    for station, group in df.groupby('station'):
        assert group.latitude.nunique() == 1 and group.longitude.nunique() == 1
        stamps = sorted(group.target_time_bjt.unique())
        targets = pd.DatetimeIndex(pd.to_datetime(stamps, utc=True)).tz_convert('Asia/Shanghai')
        loc = pvlib.location.Location(float(group.latitude.iloc[0]), float(group.longitude.iloc[0]),
                                      tz='Asia/Shanghai', altitude=0.0, name=station)
        # Same explicit midpoint quadrature as the Hunan interval baseline.
        values = [loc.get_clearsky(targets-pd.Timedelta(minutes=k), model='ineichen')['ghi'].to_numpy()
                  for k in (12.5, 7.5, 2.5)]
        means = np.mean(values, axis=0)
        assert np.isfinite(means).all() and (means >= 0).all()
        mapping = dict(zip(stamps, means))
        clear[group.index] = group.target_time_bjt.map(mapping)
    valid = np.isfinite(clear) & (clear > 0)
    removed = df.loc[~valid, ['station','init_time_bjt','target_time_bjt','split']].copy()
    removed['reason'] = 'nonpositive_interval_clear_sky'
    removed.to_csv(a.output/'excluded_keys.csv', index=False)
    df['clear_sky_ghi_15min_wm2'] = clear
    df = df.loc[valid].copy()
    df['kt'] = df.observed_ghi_15min_wm2 / df.clear_sky_ghi_15min_wm2
    assert np.isfinite(df.kt).all() and (df.kt >= 0).all()
    error = np.max(np.abs(df.kt * df.clear_sky_ghi_15min_wm2 - df.observed_ghi_15min_wm2))
    assert error < 1e-9
    df.to_csv(a.output/'cohort.csv', index=False)
    counts = df.groupby(['split','station','lead_minutes']).size().reset_index(name='count')
    counts.to_csv(a.output/'counts.csv', index=False)
    report = dict(state='LABEL_COHORT_VERIFIED', base_rows=len(clear), rows=len(df),
                  excluded_nonpositive_clear_sky=len(removed), base_sha256=audit['output_sha256'],
                  source_code_sha256=sha(Path(__file__)), cohort_sha256=sha(a.output/'cohort.csv'),
                  pvlib=pvlib.__version__, clear_sky_model='Ineichen', altitude_assumed_m=0.0,
                  interval_integration='mean of three 5min midpoints target-12.5/-7.5/-2.5min',
                  label='mean(GHI)/mean(clear_sky_GHI)', max_roundtrip_error_wm2=float(error),
                  kt_clipped=False, future_CPP_used_for_filtering=False, test_metrics_computed=False,
                  source_NWP_used=False, forecast_cache='PENDING_FROZEN_SIMVP',
                  IFS_intersection='PENDING_RELEASE_CONTRACT',
                  counts_by_split_station=df.groupby(['split','station']).size().to_dict())
    report['counts_by_split_station'] = {'/'.join(k):int(v) for k,v in report['counts_by_split_station'].items()}
    (a.output/'audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
