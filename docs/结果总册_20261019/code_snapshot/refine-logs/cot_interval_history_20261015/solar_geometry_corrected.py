"""NOAA-style nominal-UTC geometry; EOT radians are explicitly converted to degrees.
Frozen R and original banks are not modified by this GHI-only geometry module.
"""
from datetime import datetime
from typing import Tuple
import numpy as np

def _julian_day(dt: datetime) -> float:
    year, month, day = dt.year, dt.month, dt.day
    hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    if month <= 2:
        year -= 1
        month += 12
    a = int(year / 100)
    b = 2 - a + int(a / 4)
    return (
        int(365.25 * (year + 4716))
        + int(30.6001 * (month + 1))
        + day
        + b
        - 1524.5
        + hour / 24.0
    )

def compute_solar_angles(
    lat: np.ndarray, lon: np.ndarray, dt_utc: datetime
) -> Tuple[np.ndarray, np.ndarray]:
    """Return solar zenith and azimuth (degrees; azimuth north-clockwise)."""
    jd = _julian_day(dt_utc)
    jc = (jd - 2451545.0) / 36525.0
    gmls = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360.0
    gmas = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    eccentricity = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)

    gmas_rad = np.deg2rad(gmas)
    center = (
        np.sin(gmas_rad) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
        + np.sin(2.0 * gmas_rad) * (0.019993 - 0.000101 * jc)
        + np.sin(3.0 * gmas_rad) * 0.000289
    )
    true_longitude = gmls + center
    omega = 125.04 - 1934.136 * jc
    apparent_longitude = true_longitude - 0.00569 - 0.00478 * np.sin(np.deg2rad(omega))
    mean_obliquity = 23.0 + (
        26.0 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60.0
    ) / 60.0
    obliquity = mean_obliquity + 0.00256 * np.cos(np.deg2rad(omega))
    obliquity_rad = np.deg2rad(obliquity)
    declination = np.arcsin(
        np.sin(obliquity_rad) * np.sin(np.deg2rad(apparent_longitude))
    )

    vary = np.tan(obliquity_rad / 2.0) ** 2
    gmls_rad = np.deg2rad(gmls)
    equation_of_time = 4.0 * np.rad2deg(
        vary * np.sin(2.0 * gmls_rad)
        - 2.0 * eccentricity * np.sin(gmas_rad)
        + 4.0
        * eccentricity
        * vary
        * np.sin(gmas_rad)
        * np.cos(2.0 * gmls_rad)
        - 0.5 * vary * vary * np.sin(4.0 * gmls_rad)
        - 1.25 * eccentricity * eccentricity * np.sin(2.0 * gmas_rad)
    )

    utc_minutes = dt_utc.hour * 60.0 + dt_utc.minute + dt_utc.second / 60.0
    true_solar_time = (utc_minutes + equation_of_time + 4.0 * lon) % 1440.0
    hour_angle = np.deg2rad(true_solar_time / 4.0 - 180.0)

    lat_rad = np.deg2rad(lat)
    cos_soz = (
        np.sin(lat_rad) * np.sin(declination)
        + np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle)
    )
    cos_soz = np.clip(cos_soz, -1.0, 1.0)
    soz = np.rad2deg(np.arccos(cos_soz))

    sin_soz = np.sqrt(np.maximum(1.0 - cos_soz**2, 1e-12))
    cos_soa = (
        np.sin(declination) - np.sin(lat_rad) * cos_soz
    ) / (np.cos(lat_rad) * sin_soz)
    cos_soa = np.clip(cos_soa, -1.0, 1.0)
    sin_soa = -np.cos(declination) * np.sin(hour_angle) / sin_soz
    sin_soa = np.clip(sin_soa, -1.0, 1.0)
    soa = np.rad2deg(np.arctan2(sin_soa, cos_soa)) % 360.0
    return soz.astype(np.float32), soa.astype(np.float32)
