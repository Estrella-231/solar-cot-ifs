"""Map an AGRI lat/lon patch to this rectilinear CPP NetCDF grid.

Never substitute FY4B raw HDF row_index for this product's LAT/LON axes.
No interpolation is applied: this contract requires an existing grid match.
"""
import numpy as np


def map_cpp_grid(source_lat, source_lon, target_lat, target_lon, tolerance_deg=1e-4):
    lat, lon = np.asarray(source_lat), np.asarray(source_lon)
    y, x = np.asarray(target_lat), np.asarray(target_lon)
    if lat.ndim != 1 or lon.ndim != 1 or y.ndim != 2 or y.shape != x.shape:
        raise ValueError('expected 1D source axes and equally shaped 2D target coordinates')
    if not all(np.isfinite(a).all() for a in (lat, lon, y, x)):
        raise ValueError('nonfinite coordinates')
    if not all(np.all(np.diff(a)>0) or np.all(np.diff(a)<0) for a in (lat, lon)):
        raise ValueError('source axes must be strictly monotonic')
    rows = np.abs(lat[:, None]-y[:, 0][None, :]).argmin(axis=0)
    cols = np.abs(lon[:, None]-x[0, :][None, :]).argmin(axis=0)
    err_lat = float(np.max(np.abs(y-lat[rows, None])))
    err_lon = float(np.max(np.abs(x-lon[None, cols])))
    if max(err_lat, err_lon) > tolerance_deg:
        raise ValueError(f'no coincident CPP grid: errors={err_lat},{err_lon}')
    if len(np.unique(rows)) != len(rows) or len(np.unique(cols)) != len(cols):
        raise ValueError('mapping collapses multiple target pixels')
    return rows, cols, {'latitude_max_error_deg': err_lat, 'longitude_max_error_deg': err_lon}
