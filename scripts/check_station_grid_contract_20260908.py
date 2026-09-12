"""Independent station/grid check using frozen grid and source schema snapshots."""
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    src = root / 'audits/spatial_reaudit_static_20260908'
    output = root / 'audits/station_grid_contract_20260908_v1.json'
    assert not output.exists(), 'preserve previous outputs'
    schema = json.loads((src / 'station_schema.json').read_text(encoding='utf-8'))
    meta = json.loads((src / 'dataset_meta.json').read_text(encoding='utf-8'))
    conf = json.loads((root / 'configs/s_frozen_hunan_seed42.json').read_text(encoding='utf-8'))
    assert digest(src / 'grid_static.npz') == conf['grid_sha256']
    with np.load(src / 'grid_static.npz', allow_pickle=False) as z:
        lat, lon = [z[k].astype(np.float64) for k in ('lat', 'lon')]
        rr, cc = [z[k].copy() for k in ('row_index', 'col_index')]
    assert lat.shape == lon.shape == rr.shape == cc.shape == (256, 256)
    assert np.all(np.isfinite(lat)) and np.all(np.isfinite(lon))
    assert np.all(np.diff(lat, axis=0) < 0) and np.all(np.diff(lon, axis=1) > 0)
    assert np.array_equal(lat, np.broadcast_to(lat[:, :1], lat.shape))
    assert np.array_equal(lon, np.broadcast_to(lon[:1, :], lon.shape))
    records = []
    for station in schema['stations']:
        slat, slon = station['latitude'], station['longitude']
        lat1, lat2 = np.deg2rad(slat), np.deg2rad(lat)
        hav = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(np.deg2rad(lon-slon)/2)**2
        distance = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))
        nearest = tuple(int(i) for i in np.unravel_index(np.argmin(distance), lat.shape))
        r0, r1, c0, c1 = conf['station_patches'][station['station_id']]
        assert station['agri_bounds'] == [r0, r1, c0, c1]
        assert (r1-r0, c1-c0) == (16, 16)
        center = (r0+8, c0+8)
        assert nearest == center, (station['station_id'], nearest, center)
        records.append({'station': station['station_id'], 'station_latlon': [slat, slon],
                        'bounds': [r0, r1, c0, c1], 'nearest_grid_row_col': list(nearest),
                        'patch_nearest_row_col': [nearest[0]-r0, nearest[1]-c0],
                        'nearest_grid_latlon': [float(lat[nearest]), float(lon[nearest])],
                        'distance_to_nearest_km': float(distance[nearest]),
                        'true_HDF_row_col': [int(rr[nearest]), int(cc[nearest])],
                        'lat_range': [float(lat[r0:r1,c0:c1].min()), float(lat[r0:r1,c0:c1].max())],
                        'lon_range': [float(lon[r0:r1,c0:c1].min()), float(lon[r0:r1,c0:c1].max())],
                        'center_is_exact_nearest': True})
    report = {'state': 'PASS_STATION_GRID_CONTRACT_ONLY', 'region': meta['region'],
              'grid_shape': list(lat.shape), 'crop_from_dataset_meta': meta['crop'],
              'extent_lat': [float(lat.min()), float(lat.max())], 'extent_lon': [float(lon.min()), float(lon.max())],
              'north_to_south': True, 'west_to_east': True, 'stations': records,
              'time_contract': meta['time_conventions'],
              'sources': {str(p.relative_to(root)): digest(p) for p in [src/'grid_static.npz', src/'station_schema.json', src/'dataset_meta.json', root/'configs/s_frozen_hunan_seed42.json', Path(__file__)]},
              'limits': ['Station coordinates use the authoritative project schema, not an independent ground survey.',
                         'This check does not prove original HDF geolocation or actual AGRI/CPP payload alignment; those are separate checks.'],
              'test_used': False, 'training': False}
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
