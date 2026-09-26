"""Sparse HDF reads and V2-coordinate regional auxiliary interpolation."""
import os
import time
from pathlib import Path

import netCDF4
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr


def resize_crop(array, bounds, nearest=False):
    """Evaluate original 4051-output interpolation only at requested indices."""
    a = torch.as_tensor(np.asarray(array), dtype=torch.float32)
    y0, y1, x0, x1 = bounds
    height, width = a.shape[-2:]
    if nearest:
        ys = torch.floor(torch.arange(y0, y1, dtype=torch.float32) * (height / 4051)).long()
        xs = torch.floor(torch.arange(x0, x1, dtype=torch.float32) * (width / 4051)).long()
        return a[:, ys[:, None], xs[None, :]].numpy()
    # Keep the original PyTorch interpolation kernel exactly. A direct regional
    # bilinear expression differed by ~6e-5 in auxiliary input and failed the
    # strict COT tolerance. This temporary full resize occurs once per hour;
    # only the regional crop survives in the cache. HDF and LCCS remain sparse.
    with torch.inference_mode():
        value = F.interpolate(a[None], size=(4051, 4051), mode="bilinear", align_corners=True)
    return value[0, :, y0:y1, x0:x1].numpy().copy()


class RegionInputs:
    def __init__(self, base, bounds):
        import Retrieval_FY4B_105E as adapter
        import Get_ERA_LCCS as auxiliary
        self.adapter, self.auxiliary = adapter, auxiliary
        self.bounds = bounds
        y0, y1, x0, x1 = bounds
        self.rows = adapter.AGRI_FUNC_LAT_INDEX[y0:y1, x0:x1]
        self.cols = adapter.AGRI_FUNC_LON_INDEX[y0:y1, x0:x1]
        self.saz = adapter._Get_AGRI_SAZ()[y0:y1, x0:x1].astype(np.float32)
        self.native_bounds = (int(self.rows.min()), int(self.rows.max())+1,
                              int(self.cols.min()), int(self.cols.max())+1)
        auxiliary.Convert_High_Resolution = lambda z: resize_crop(z, self.bounds)
        self.era_key = None
        self.era_value = None
        self.lccs_year = None
        self.lccs_value = None
        self.last_sources = {}

    def lccs(self, year):
        if self.lccs_year == year:
            return self.lccs_value
        path = Path(self.auxiliary.LCCS_Dir) / f"MCD12_IGBP_{year}_Global_CMG.nc"
        with xr.open_dataset(path, engine="netcdf4") as ds:
            field = ds["LCCS_IGBP"].sel(LAT=slice(81, -81))
            west = field.sel(LON=slice(24, 180))
            east = field.sel(LON=slice(-180, -174))
            height = west.sizes["LAT"]
            width = west.sizes["LON"] + east.sizes["LON"]
            y0, y1, x0, x1 = self.bounds
            ys = np.floor(np.arange(y0, y1, dtype=np.float32)*(height/4051)).astype(int)
            xs = np.floor(np.arange(x0, x1, dtype=np.float32)*(width/4051)).astype(int)
            if xs.max() >= west.sizes["LON"]:
                raise ValueError("Hunan region unexpectedly intersects longitude wrap")
            value = west.isel(LAT=ys, LON=xs).values.astype(np.float32)[None]
        self.lccs_year, self.lccs_value = year, value
        return value

    def load(self, target):
        started = time.perf_counter()
        from Satellite_Data_Loaders import Calibrate_FY_Channel
        from Satellite_Retrieval import Reorder_ERA_For_FY
        tag = target.strftime("%Y%m%d%H%M%S")
        root = Path(self.adapter.AGRI_DIR) / tag[:4] / tag[:8]
        files = list(root.glob(f"*_{tag}_*_4000M_*.HDF"))
        if len(files) != 1:
            raise ValueError(f"expected unique raw HDF for {tag}, found {len(files)}")
        ry0, ry1, rx0, rx1 = self.native_bounds
        channels = []
        with netCDF4.Dataset(files[0]) as ds:
            for channel in (11, 12, 13, 14, 15):
                raw = ds["Data"][f"NOMChannel{channel}"][ry0:ry1, rx0:rx1]
                lut = ds["Calibration"][f"CALChannel{channel}"][:]
                calibrated = Calibrate_FY_Channel(raw, lut)
                channels.append(calibrated[self.rows-ry0, self.cols-rx0])
        bt = np.stack(channels).astype(np.float32)
        bt[:, np.isnan(self.saz)] = np.nan
        bt = np.concatenate((bt, self.saz[None]), axis=0)
        bt_end = time.perf_counter()
        key = target.replace(minute=0, second=0, microsecond=0)
        if key != self.era_key:
            aux = self.auxiliary
            env = np.concatenate((aux.Get_ATP(target, 81, -81, 186, 24),
                                  aux.Get_RHP(target, 81, -81, 186, 24),
                                  aux.Get_SKT(target, 81, -81, 186, 24),
                                  self.lccs(target.year)), axis=0)
            self.era_key, self.era_value = key, Reorder_ERA_For_FY(env)
        self.last_sources = {"raw_hdf": str(files[0]), "raw_size": files[0].stat().st_size,
                             "hdf_native_bounds": list(self.native_bounds),
                             "era_valid_hour_utc": key.isoformat(), "lccs_year": target.year}
        self.last_timings = {"bt": bt_end-started, "environment": time.perf_counter()-bt_end}
        return bt, self.era_value
