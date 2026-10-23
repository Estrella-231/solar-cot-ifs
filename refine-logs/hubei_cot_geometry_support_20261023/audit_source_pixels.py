"""Independent scalar source reads; no interpolation, model, GHI, or test data."""
import json
from pathlib import Path
import netCDF4
import numpy as np

P = Path(__file__).resolve().parent
request = json.loads((P / 'HEIGHT_SOURCE_REQUEST.json').read_text())
centers = {'Niushou': (32.08, 112.04), 'Wangzhai': (32.04, 112.12),
           'Mizhuang': (32.12, 112.20)}
records = []
for row in request['rows']:
    assert row['split'] == 'train'
    with netCDF4.Dataset(row['source']) as ds:
        lat = np.asarray(ds['LAT'][:]); lon = np.asarray(ds['LON'][:])
        record = dict(time_bjt=row['time_bjt'], source=row['source'],
                      variables={k: dict(dimensions=list(v.dimensions), shape=list(v.shape))
                                 for k, v in ds.variables.items()},
                      global_attribute_names=list(ds.ncattrs()),
                      time_named_variables=[k for k in ds.variables if 'time' in k.lower()],
                      station_pixels={})
        for st, (la, lo) in centers.items():
            y = int(np.argmin(abs(lat-la))); x = int(np.argmin(abs(lon-lo)))
            assert abs(float(lat[y])-la) < 1e-4 and abs(float(lon[x])-lo) < 1e-4
            values = {}
            for key in ['CTH', 'CLP', 'COT']:
                scalar = np.ma.asarray(ds[key][y, x], dtype=np.float64)
                value = float(np.ma.filled(scalar, np.nan))
                values[key] = value if np.isfinite(value) else None
            record['station_pixels'][st] = dict(source_row=y, source_col=x,
                                               lat=float(lat[y]), lon=float(lon[x]), values=values)
        records.append(record)
(P/'SOURCE_SCALAR_WITNESS.json').write_text(json.dumps(dict(
    state='DIRECT_SOURCE_SCALAR_READS_COMPLETE', records=records,
    acquisition_time_verified=False, release_time_verified=False,
    note='Filename times are bound; no acquisition-time or release-time claim from filenames.',
    test_payloads_read=0, GHI_labels_read=0), indent=2))
print('DIRECT_SOURCE_SCALAR_READS_COMPLETE', len(records))
