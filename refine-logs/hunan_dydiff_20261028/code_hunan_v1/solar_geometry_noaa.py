"""Geometric solar angles (NOAA/Meeus), UTC input and east-positive longitude.

EOT uses 4 minutes per degree, so the trigonometric result in radians MUST be
converted to degrees before conversion to minutes. No atmospheric refraction.
Reference: https://gml.noaa.gov/grad/solcalc/calcdetails.html
Frozen historical source is retained separately; this module is versioned.
"""
from datetime import timezone
import numpy as np

def _julian_day(dt):
    y, m, d = dt.year, dt.month, dt.day
    h = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    if m <= 2:
        y -= 1
        m += 12
    a = int(y / 100)
    b = 2 - a + int(a / 4)
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5 + h / 24.0

def compute_solar_angles(lat, lon, dt_utc):
    """
    返回:
        soz:      太阳天顶角 (度), 夜间保留 >90°
        soa:      太阳方位角 (度, 0°=北顺时针)
        day_mask: 白天=1, 夜间=0
    """
    if dt_utc.tzinfo is not None:
        dt_utc = dt_utc.astimezone(timezone.utc).replace(tzinfo=None)
    lat, lon = np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)
    if not np.isfinite(lat).all() or not np.isfinite(lon).all() or (abs(lat)>90).any() or (abs(lon)>180).any():
        raise ValueError("Invalid latitude/longitude")
    jd = _julian_day(dt_utc)
    jc = (jd - 2451545.0) / 36525.0

    gmls = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360
    gmas = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    eeo = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)

    gmas_rad = np.deg2rad(gmas)
    sec = (np.sin(gmas_rad) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
           + np.sin(2 * gmas_rad) * (0.019993 - 0.000101 * jc)
           + np.sin(3 * gmas_rad) * 0.000289)
    stl = gmls + sec

    omega_m = 125.04 - 1934.136 * jc
    sal = stl - 0.00569 - 0.00478 * np.sin(np.deg2rad(omega_m))
    moe = 23.0 + (26.0 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60.0) / 60.0
    oc = moe + 0.00256 * np.cos(np.deg2rad(omega_m))
    oc_rad = np.deg2rad(oc)
    decl_rad = np.arcsin(np.sin(oc_rad) * np.sin(np.deg2rad(sal)))

    vary = np.tan(oc_rad / 2.0) ** 2
    gmls_rad = np.deg2rad(gmls)
    eot = 4.0 * (180.0 / np.pi) * (vary * np.sin(2 * gmls_rad)
                 - 2.0 * eeo * np.sin(gmas_rad)
                 + 4.0 * eeo * vary * np.sin(gmas_rad) * np.cos(2 * gmls_rad)
                 - 0.5 * vary * vary * np.sin(4 * gmls_rad)
                 - 1.25 * eeo * eeo * np.sin(2 * gmas_rad))

    utc_min = dt_utc.hour * 60.0 + dt_utc.minute + dt_utc.second / 60.0
    tst = (utc_min + eot + 4.0 * lon) % 1440.0
    ha_rad = np.deg2rad(tst / 4.0 - 180.0)

    lat_rad = np.deg2rad(lat)
    cos_soz = (np.sin(lat_rad) * np.sin(decl_rad)
               + np.cos(lat_rad) * np.cos(decl_rad) * np.cos(ha_rad))
    cos_soz = np.clip(cos_soz, -1.0, 1.0)
    soz = np.rad2deg(np.arccos(cos_soz))

    sin_soz = np.sqrt(np.maximum(1.0 - cos_soz ** 2, 1e-12))
    cos_soa = (np.sin(decl_rad) - np.sin(lat_rad) * cos_soz) / (np.cos(lat_rad) * sin_soz)
    cos_soa = np.clip(cos_soa, -1.0, 1.0)
    sin_soa = -np.cos(decl_rad) * np.sin(ha_rad) / sin_soz
    sin_soa = np.clip(sin_soa, -1.0, 1.0)
    soa = np.rad2deg(np.arctan2(sin_soa, cos_soa)) % 360.0

    day_mask = (soz < 90.0).astype(np.float32)

    return soz, soa, day_mask
